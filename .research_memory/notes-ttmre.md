# TTM-RE 精读笔记（全文版）

> 时间：2026-05-14
> 论文：Gao et al., *TTM-RE: Memory-Augmented Document-Level Relation Extraction*, ACL 2024 Long Paper
> arXiv：2406.05906
> 论文路径：`papers/2024.acl-long.26.pdf`（本地，不入库）
> 代码路径：`codes/TTM-RE/`（本地 clone，不入库）
>
> 本文件是对话里给出的"精读全文"的落盘版。压缩版要点已并入 [`02-papers.md`](./02-papers.md) TTM-RE 条目下的「📖 精读笔记」区块。

---

## 1. 一句话定位

**问题**：DocRE 上"distant supervision 数据 = 大量噪声 + false negative"，以前的方法（ATLOP / SSR-PU / KD-DocRE / DREEAM）即使加上 distant 数据也很难显著涨分，**作者认为这是架构问题，不是数据质量问题**。

**做法**：在 RoBERTa-large + ATLOP 范式之上，**插入一个可学习的 memory 模块（Token Turing Machine）**，对 `<head, tail>` 实体表示做"再加工"，再过 group bilinear + adaptive thresholding 出 logits；loss 用 SSR-PU。

**结果**（Re-DocRED test）：

| Setting | TTM-RE | 最强 baseline | Δ |
|---|---|---|---|
| Human-only | 79.95 | SSR-PU 80.18 | **没赢** |
| Distant-only | 63.00 | SSR-PU 54.46 | **+8.54** |
| Human + Distant | 84.01 | DREEAM 81.67 / SSR-PU 80.52 | **+2~3** |
| ChemDisGene | 53.59 | SSR-PU 48.56 | **+5.03** |
| 极端 19% 标签（H+D） | 66.47 | SSR-PU 54.34 | **+12.13** |

**关键洞察**：TTM 的优势**完全来自大规模噪声数据下的鲁棒学习**——在 fully-supervised + 干净小数据 setting 下相对 SSR-PU 没有优势。

---

## 2. 方法骨架（论文 §3 + 代码对照）

### 2.1 Encoder + 实体表示（沿用 ATLOP）

[`model2.py` `process_long_input`](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L40-L114)：长文档 ≤ 512 token 直接过 RoBERTa；> 512 切成两个 overlapping chunk 分别 encode 再拼回，attention 也加权合并。

[`model2.py` `get_hrt`](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L196-L256)：每个 entity 多个 mention → `logsumexp` 池化得到 `e_emb`；entity-pair 的 attention `ht_att = h_att * t_att` 用来从 sequence 里抽 context `rs`。这就是 ATLOP 的 localized context pooling，**TTM-RE 完全沿用没改**。

### 2.2 TTM 记忆模块（核心）

公式：`Z = Read(M, I) = S_r([M||I])`，其中

- `M ∈ R^(m×d)`：m=200 个**可学习** memory token（`nn.Parameter`，xavier 初始化，**不是从 0 初始化**——论文明说从 0 学不动，因为没有梯度信号）
- `I ∈ R^(2×d)`：head/tail 两个实体表示
- `S_r`：TokenLearner，用 MLP 算 softmax 权重，对 `[M||I]` 做加权和，输出 `r=2` 个 token，即"memory-augmented head/tail"
- 然后再过一层 Transformer encoder（论文消融到 4 层最优）

代码确认 [`model2.py` L297-L317](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L297-L317)：

```python
self.mu_encoder = TokenTuringMachineEncoder(
    process_size=2, memory_size=200, input_dim=emb_size,
    mlp_dim=emb_size, num_layers=args.num_layers
)
# 调用：
encoded = self.mu_encoder(
    torch.cat([hs.unsqueeze(1), ts.unsqueeze(1)], dim=1).unsqueeze(1)
)
hs2 = encoded[:, 0, 0, :]   # memory-augmented head
ts2 = encoded[:, 0, 1, :]   # memory-augmented tail
b1 = (hs2/2 + hs/2).view(-1, emb_size // block_size, block_size)
b2 = (ts2/2 + ts/2).view(-1, emb_size // block_size, block_size)
```

**这里有个论文里没强调但代码里很明显的细节**：最终送 bilinear 的是 **0.5 × 原始 + 0.5 × memory-augmented**，即 memory 做的是"残差修正"而不是替代。这点在论文 §3.3 里没写，是看代码才发现的。

⚠️ 还有一处**疑似 bug**（[`model2.py` L289](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L289)）：tail 用的 `head_extractor`，不是 `tail_extractor`：

```python
ts = torch.tanh(self.head_extractor(torch.cat([ts, rs], dim=1)))
                     # ↑ 这里"应该"是 tail_extractor
```

作者权重是按这种"错误"训练的，复现时**不要去改**——改了权重就废了。

### 2.3 TTM 内部细节

[`ttm.py` `TokenAddEraseWrite`](file:///Users/bytedance/projects/kait/codes/TTM-RE/ttm.py#L66-L126) 还有一套 add/erase 写操作（仿 NTM），但**当前 forward 没用到**——`TokenTuringMachineUnit.forward` 里的 `mem_out_tokens` 计算了但被丢弃，外层只取 `output_tokens`。也就是说**真正用的只有 read 不是 write**，论文 §3.2 也明说"我们不做 write"。

`process_size=2` 意味着 read 出 2 个 token，正好对应 head/tail。这是把 TTM 这个本来用于视频时序的模块**强行降维到"非时序的实体对处理器"**。

### 2.4 Loss：SSR-PU（沿用 Wang et al. 2022b）

[`model2.py` `m_tag='S-PU'`](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L380-L398)：每类独立做 PU + class prior shift 校正。关键超参 `e=1.0`（priors 倍数）、`m=1.0`（margin）、`beta=0`、`gamma=1`。

公式（论文 §3.4 / Appendix F）：

```
R_S-PU(f) = Σ_i [
    (π_i / n_Pi) Σ ℓ(f_i(x_Pi), +1)
  + max(0, [
        (1 / n_Ui) · ((1-π_i) / (1-π_u,i)) · Σ ℓ(f_i(x_Ui), -1)
      − (1 / n_Pi) · ((π_u,i − π_u,i·π_i) / (1-π_u,i)) · Σ ℓ(f_i(x_Pi), -1)
    ])
]
```

其中 `π_u,i = (π_i − π_labeled,i) / (1 − π_labeled,i)`。

**这个 loss 完全没改**，是 SSR-PU 原版照搬。所以 TTM-RE 的 contribution = **「memory 模块」+「两阶段训练 schedule」**，不在 loss。

### 2.5 训练 schedule（关键且容易被忽略）

[`run_roberta_rank.sh`](file:///Users/bytedance/projects/kait/codes/TTM-RE/scripts/run_roberta_rank.sh) 配 `pretrain_distant=4`，对应 [`train2.py` L460-L495](file:///Users/bytedance/projects/kait/codes/TTM-RE/train2.py#L460-L495)：

- **Stage 1**：`pretrain_distant=1`，先在 101k distant 数据上 pre-train 2 epoch（`lr=5e-5`），保存 `pretrain_state_dict.pth`
- **Stage 2**：`pretrain_distant=2`，load pretrain weights，在 3k human-annotated 上 fine-tune 30 epoch（`lr=1e-5`），保存 `finetune_state_dict.pth`
- **Stage 3**：`pretrain_distant=4`，load finetune 权重直接 evaluate

**这个"distant-pretrain → human-finetune"两段式才是 TTM 涨分的关键**，不是单 stage 训练能比的。SSR-PU baseline 默认是直接联合训。

---

## 3. 实验结论（一表看清）

| Setting | TTM-RE 优势？ | 量级 |
|---|---|---|
| Human-only | ❌ 没赢 SSR-PU | F1 ≈ 80 |
| Distant-only | ✅ 大胜 | +8.5 F1 |
| Human + Distant | ✅ 胜 | +2~3 F1 |
| ChemDisGene（生物医学） | ✅ | +5 F1 |
| 极端无标注 19% | ✅ | +12 F1 |
| Top-10 high-freq labels | ✅ 小胜 | +4 |
| All-but-Top-10 long-tail | ✅ | +4.5 |

**ablation**：

- memory size 10 → 200 单调涨（83.20 → 84.01），**作者说还能再大**
- 层数 1 → 4 单调涨（83.56 → 84.01），**作者说也没探到顶**
- DeBERTaV3-large 替 RoBERTa-large **反而掉**（80.56 vs 84.01 在 H+D setting）——加参数 ≠ 涨分，加 memory 才涨

---

## 4. 局限（用户开题 Y 改进可借此切入）

论文自己点到 + 看代码挖出来的：

1. **Memory token 从 normal 初始化**，没用 entity 语义 prior（作者明说"future work 应该研究怎么初始化 memory"——**Y4 的直接入口**）
2. **不写 memory**（write 操作 `TokenAddEraseWrite` 代码里有但没启用）→ memory 是全局静态的，对每个 doc 都一样，没"按文档定制"
3. **TTM 只看 entity-pair 表示，不看 retrieved 文档/示例**——这就是**接 RAG/ICL 的天然口子**：把 retrieved 邻居 doc 的 entity 表示也塞进 memory module
4. **Stage 2 fine-tune 时 memory 是不是 freeze 没明说**——代码里有一行 `model.mu_encoder.memory_tokens.requires_grad_(False)` 但被注释掉了，**实际是 memory 也跟着 fine-tune**
5. Encoder 仍是 RoBERTa-large，**没用 LLM** → 标题包"Retrieval-Augmented"挂 LLM verifier/reranker 是干净空挡
6. **没有任何 retrieval**——名字叫"Memory-Augmented"但其实是 in-model learnable memory，不是真正的 retrieve external context。**这点很关键**：你写"Retrieval-Augmented DocRE"是真填了一个空白
7. 没做 error analysis 区分"哪些预测是 memory 起的作用"——只在 Appendix H 做了一个 case study（Republic of China 那个例子），证据弱

---

## 5. 对 4 个候选 Y 的具体落点（细化到代码行）

| Y | 在代码哪里改 | 工作量 | 期望收益 | 失败兜底 |
|---|---|---|---|---|
| **Y1 替换 retriever** | 现在**没有** retriever，要新增一个：把 distant 训练集中相似 doc 的 entity 表示塞进 `mu_encoder` 的 memory（替换 `nn.Parameter` 为动态 retrieved tokens） | 中（2-4 周） | 长尾关系 +1~2 F1（Top10 之外类） | 写 negative result：external memory 不如 internal learnable memory |
| **Y2 LLM verifier** | 完全外挂：在 [`evaluate` L173-L235](file:///Users/bytedance/projects/kait/codes/TTM-RE/train2.py#L173-L235) 拿到 top-K 候选三元组后，过 LLM 判真伪 | 低（1-2 周） | Precision +2~3、Recall 不变 | 失败也能写"LLM 后处理对 DocRE 无效" |
| **Y3 LLM reranker** | 同 Y2 入口，但是 rerank 不是过滤 | 低 | Top-1 F1 +1 | 同上 |
| **Y4 InfoNCE on memory tokens** | 在 `mu_encoder.memory_tokens` 上加约束：把同关系的 entity-pair 表示拉近 / 不同关系推开（对接你 B 会 InfoNCE 经验） | 低-中 | +0.5~1 F1，但更主要是**故事好讲**（"对比学习 × memory"） | 失败可保留为消融 |

**初步建议**：先 **Y2**，最便宜、最像 RAG、最容易在开题报告里把"LLM"和"verifier"两个词都讲清。Y4 是用户技能最匹配的，可作 Y2 的补充消融。**最终选哪个等 P0 复现跑完再定**。

---

## 6. 复现的几个坑（提前预警）

1. `transformers==4.34.0`、`torch==2.0.1`、`numpy==1.24.3`，老版本，新机器上不能闭眼装
2. [`train2.py`](file:///Users/bytedance/projects/kait/codes/TTM-RE/train2.py#L2) 第 2 行硬编码 `os.environ['TRANSFORMERS_CACHE'] = '/srv/local/data/chufan2/huggingface/'`，**必须改**
3. distant 数据预处理后会 cache 成 `distant_features_roberta-large.pkl`，10w doc 第一次 prepro 慢
4. tail 用 `head_extractor` 的疑似 bug——**不要"修"**，作者权重是这么训的
5. memory 200 × 1024（RoBERTa-large hidden）只占 0.2M 参数，主体显存还是 RoBERTa-large + bilinear，单卡 3090 应该够（论文用 A6000 48G 是为了更大 batch）

---

## 7. 给后续工作的几个观察（非论文原文）

- **TTM-RE 的"Memory-Augmented"是误导性命名**：实际是 in-model learnable 的 prototype tokens，没有 retrieval。这给了"真做 retrieval"的工作合法的差异化空间。
- **作者把上界明显留出来了**：memory size 单调涨没探顶、层数单调涨没探顶、初始化方式自承没研究——开题写"future work" → "本工作改进"映射很顺。
- **两段式训练的成本**：Stage 1 distant 的 101k doc × 2 epoch 是大头，单卡 3090 可能要 1-2 天。复现先单跑 Stage 2（用作者放出的 pretrain 权重）就能直接 reproduce 表 3 的 H+D 数字，**不必从 distant 重训**——这能省 80% 时间。
- **ChemDisGene 是个 hidden gem**：domain shift 实验（生物医学）也涨了 5 F1，开题报告里"方法的领域可推广性"段落可以引这个。
