# 论文池

> 何时更新：新增论文、引用数更新、读完一篇后填观点
>
> 引用数标注：[OA: N] = OpenAlex，[SS: N] = Semantic Scholar

---

## 🥇 底座论文（已锁定，2026-05-11）

### TTM-RE：Memory-Augmented Document-Level Relation Extraction
- **arXiv**：2406.05906（作者：Chufan Gao, Xuan Wang, Jimeng Sun）
- **会议**：ACL 2024 Long Paper（非 Findings，主会）
- **引用**：[SS: 12]
- **代码**：https://github.com/chufangao/TTM-RE ⭐
- **权重**：**已发布预训练权重**（这是选它做底座的关键）
- **数据集**：DocRED / Re-DocRED / ChemDisGene
- **Backbone**：RoBERTa-large
- **核心方法**：Token Turing Machine（TTM）做记忆增强 + PU learning 噪声鲁棒 loss
- **与用户技术栈匹配**：RoBERTa-large + 对比学习 1:1 对口
- **预估显存**：~16G（单卡 3090 足够）
- **与 CLARE 的关系**：CLARE 是句子级 RE + LLM ICL，TTM-RE 是文档级 RE + encoder memory，**任务升级 + 架构换代**，无重叠

**为什么选它做底座**：见 `01-directions.md` "为什么是这个组合" 表。

---

#### 📖 精读笔记（2026-05-14，PDF + 代码对照）

**论文路径**：`papers/2024.acl-long.26.pdf`（本地）
**代码路径**：`codes/TTM-RE/`（本地 clone，不入库）

##### 1. 一句话定位
- **问题**：DocRE 的 distant supervision 数据噪声大、false negative 多，以前的方法（ATLOP/SSR-PU/KD-DocRE/DREEAM）即使加了 distant 数据也不显著涨分。作者认为这是**架构限制**，不是数据质量问题。
- **做法**：在 RoBERTa-large + ATLOP 范式之上，**插入一个可学习的 memory 模块（Token Turing Machine）**，对 `<head, tail>` 实体表示做"再加工"，再过 group bilinear + adaptive thresholding；loss 用 SSR-PU（沿用 Wang 2022b，未改）。
- **结果**（Re-DocRED test）：
  - Human-only：F1 79.95 ≈ SSR-PU 80.18（**没赢**）
  - Distant-only：F1 63.00 vs SSR-PU 54.46（**+8.5**）
  - Human + Distant：F1 84.01 vs DREEAM 81.67、SSR-PU 80.52（**+2~3**）
- **关键洞察**：TTM 的优势**完全来自大规模噪声数据下的鲁棒学习**——干净小数据 setting 下没有优势。

##### 2. 方法细节（论文 §3 + 代码对照）

**Encoder**：完全沿用 ATLOP 的长文档 chunking + localized context pooling（`process_long_input` / `get_hrt`），TTM-RE 一字未改。

**TTM 记忆模块**（核心，但比论文写得朴素）：
- `M ∈ R^(200×1024)`：可学习 memory token，`nn.Parameter`，xavier 初始化（**论文明说从 0 学不动**）
- `Read(M, I) = TokenLearner([M||I])`：MLP softmax 加权和，输出 r=2 个 token，对应 memory-augmented head/tail
- 然后过 1 层 Transformer encoder
- ⚠️ **代码里发现的论文未提细节**：最终送 bilinear 的是 `0.5 × 原始 + 0.5 × memory-augmented`，memory 做的是**残差修正**而不是替代（[model2.py L297-L317](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L297-L317)）
- ⚠️ **疑似 bug**：tail 也用 `head_extractor`（[model2.py L289](file:///Users/bytedance/projects/kait/codes/TTM-RE/model2.py#L289)），作者权重就这么训的，复现时**不要"修"**
- ⚠️ TTM 内部有 `TokenAddEraseWrite`（NTM 风格写操作），代码里有但 forward **没启用** → 实际是 read-only memory，每个 doc 用同一份静态 memory

**训练 schedule（关键且容易忽略）**：
- Stage 1：在 101k distant 上 pretrain 2 epoch（lr=5e-5）
- Stage 2：在 3k human 上 fine-tune 30 epoch（lr=1e-5）
- 这个**两段式**才是涨分关键，不是单 stage 联合训能比的（baseline 默认联合训）

##### 3. 局限（开题 Y 改进的切入口）

1. Memory token 是 normal 初始化，**没用 entity 语义 prior**——作者明说"future work 应改进初始化"
2. **没启用 write 操作** → memory 是全局静态的，对每个 doc 都一样
3. **TTM 只看 entity-pair，不看 retrieved 邻居 doc**——这是**接 RAG/ICL 的天然口子**
4. 名字叫"Memory-Augmented"但**不是真 retrieval**，是 in-model learnable memory → 写"Retrieval-Augmented DocRE"是真填空白
5. Encoder 仍是 RoBERTa-large，**完全没用 LLM** → LLM verifier/reranker 是干净空挡
6. Error analysis 只有 1 个 case study（Appendix H），证据弱

##### 4. 4 个候选 Y 的具体落点

| Y | 在代码哪里改 | 工作量 | 期望收益 | 失败兜底 |
|---|---|---|---|---|
| **Y1 替换 retriever** | 现在没 retriever，要新增：把相似 doc 的 entity 表示动态塞进 memory（替换 `nn.Parameter` 静态 memory） | 中（2-4 周） | 长尾关系 +1~2 F1 | 写 negative result |
| **Y2 LLM verifier** | 外挂：[evaluate L173-L235](file:///Users/bytedance/projects/kait/codes/TTM-RE/train2.py#L173-L235) 拿到 top-K 三元组后过 LLM 判真伪 | 低（1-2 周） | Precision +2~3 | 失败可写"LLM 后处理对 DocRE 无效" |
| **Y3 LLM reranker** | 同 Y2 入口，rerank 不过滤 | 低 | Top-1 F1 +1 | 同上 |
| **Y4 InfoNCE on memory tokens** | 在 `mu_encoder.memory_tokens` 上加 InfoNCE：同关系 entity-pair 拉近 / 不同关系推开（对接 B 会经验） | 低-中 | +0.5~1 F1 | 保留为消融 |

**初步建议**：先 Y2（最便宜、最像 RAG、把"LLM"+"verifier"两个词在开题报告里讲清）；Y4 作为消融补充。**最终选哪个等 P0 复现跑完再定**。

##### 5. 复现坑预警

1. `transformers==4.34.0` / `torch==2.0.1` / `numpy==1.24.3`，老版本，新机器不能闭眼装
2. `train2.py` 第 2 行硬编码 `os.environ['TRANSFORMERS_CACHE'] = '/srv/local/data/chufan2/...'`，**必须改**
3. distant 数据预处理慢，第一次会 cache 成 `distant_features_roberta-large.pkl`
4. tail 用 `head_extractor` 的疑似 bug，**不要"修"**——会废作者权重
5. 论文用 A6000 48G，3090 24G 显存够（memory 才 200×1024 ≈ 0.2M 参数）但 batch 要小

---

## 🥈 候选底座清单 v1（2026-05-11 评估，备查）

| # | 论文 | 年/会议 | 引用 | 代码 | 数据集 | Backbone | 显存 | 复现风险 | 选择结果 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **TTM-RE** | ACL 2024 长文 | SS:12 | [chufangao/TTM-RE](https://github.com/chufangao/TTM-RE) + 权重 | DocRED/Re-DocRED | RoBERTa-large | ~16G | **极低** | ✅ **已选** |
| 2 | DEEIA | ACL Findings 2024 | — | [LWL-cpu/DEEIA](https://github.com/LWL-cpu/DEEIA) | RAMS/WikiEvents/MLEE/ACE05 | RoBERTa | ~16G | 低 | ❌（EAE 路线被排除） |
| 3 | CsEAE | 2024 / arXiv | OA:1 | [simon-p-j-r/CsEAE](https://github.com/simon-p-j-r/CsEAE) | RAMS/WikiEvents/MLEE | RoBERTa + LLM | ~16G | 中 | ❌（EAE） |
| 4 | AutoRE | ACL 2024 Demo | SS:34 | [THUDM/AutoRE](https://github.com/THUDM/AutoRE) | Re-DocRED | Mistral/Llama-7B + QLoRA | ~24G | 中 | ⚠️（可作 baseline） |
| 5 | LMRC | 2024 / arXiv | SS:11 | [wisper12933/LMRC](https://github.com/wisper12933/LMRC) | DocRED/Re-DocRED | LLM (ICL) | 推理为主 | 中 | ⚠️（作为 Y3 reranker 参考） |
| 6 | HD-LoA | ACL 2024 长文 | SS:11 | [hzzhou01/HD-LoA-Prompting](https://github.com/hzzhou01/HD-LoA-Prompting) | RAMS + DocEE | 纯 prompting | 低 | 低 | ❌（无训练，故事弱） |
| 7 | Context-Guided LP | AAAI 2024 | — | [kracr/document-level-relation-extraction](https://github.com/kracr/document-level-relation-extraction) | DocRED/Re-DocRED | RoBERTa + KG | ~16G | 中 | ⚠️（backup） |
| 8 | KeyEE | BDMA 2024 | SS:11 | [OStars/KeyEE](https://github.com/OStars/KeyEE) | ACE05 / ERE | T5 | ~24G | 高（LDC 数据） | ❌ |

---

## ⭐ 9 篇必读论文（截至 2026-05-11 引用排名）

| # | 简称 | arXiv | 年/会议 | OA | SS | 已下载 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | LMRC | 2408.13889 | 2024 / arXiv | 3 | **11** | ❌ | DocRE 两阶段范式起点，引用网络最活跃 |
| 2 | ULTRA + LEAFER | 2401.13218 | 2024 / ACL Findings | 4 | (429) | ❌ | DocEAE 必引基线，Bloomberg 出品 |
| 3 | LegalCore | 2502.12509 | 2025 / arXiv | 0 | 2 | ❌(失败) | 数据集，3 个月被引 2 次很快 |
| 4 | CsEAE | 2411.05895 | 2024 / arXiv | 1 | (429) | ✅ | DocEAE 小+大模型协作 |
| 5 | KnowRA | 2501.00571 | 2024 / arXiv | 0 | 1 | ❌ | DocRE + 知识增强 |
| 6 | Triggers Needed? | 2411.08708 | 2024 / arXiv | 0 | (429) | ❌ | DocEE 范式反思 |
| 7 | ThinkTwice | 2601.18395 | 2026-01 / arXiv | 0 | (429) | ❌ | 采样+选择，太新无引用很正常 |
| 8 | GenExtract | 2603.02909 | 2026-03 / arXiv | 0 | (429) | ❌ | 多智能体 + 零样本 DocEAE |
| — | ECB++ | — | — | — | — | — | 名字在 arXiv 无对应论文，应是数据集别名 |

## ✅ 用户已下载的论文（在 `/Users/bytedance/projects/kait/.arxiv_papers/`）

| arXiv | 简称 | 是否精读 | 关键观点摘录 |
|---|---|---|---|
| 2411.05895 | CsEAE | ❌ | DocEAE 小+大模型协作 |
| 2508.00757 | GLiDRE | ❌ | GLiNER 风格 bi-encoder 用于 DocRE |
| 2511.08143 | RelPrior | ❌ | "关系作为先验"——LMRC 两阶段范式的对偶 |

## 🔥 LMRC 的 11 篇引用论文（按价值分档，2026-05-11 拉取）

### 第一档：直接接续 DocRE × LLM 主线（7 篇，开题必看）

| 论文 | arXiv/DOI | 时间/会议 | 跟进点 |
|---|---|---|---|
| RelPrior | 2511.08143 | 2025-11 / arXiv | 关系作为先验注入，对偶 LMRC |
| **DTPE** | 10.1109/icassp55912.2026.11460694 | 2026 / **ICASSP 2026** | 文档树解析 + LLM 数据精修 |
| **Coarse-to-Fine** | 10.1109/icassp55912.2026.11461490 | 2026 / **ICASSP 2026** | 低资源 DocRE 的粗到细两阶段 |
| **MAUM** | 10.1109/IJCNN64981.2025.11228956 | 2025 / IJCNN | **U-Mamba 替代 Transformer** + 记忆增强 |
| Two-Stage Loss + Anaphor | 10.1109/IJCNN64981.2025.11228050 | 2025 / IJCNN | 两阶段 loss + 指代消解 |
| Correction & Completion | 10.1109/ICAACE65325.2025.11019681 | 2025 / ICAACE | LLM 当后处理器（**Y2 verifier 的直接 motivation**） |
| **Hallucination-Resistant RE** | 2508.14391 | 2025-08 / arXiv | LLM 幻觉抑制，Y2 motivation |

### 第二档：相关但不直接 follow（1 篇）

| 论文 | 关系 |
|---|---|
| Can an LLM Induce a Graph? (2510.03611, ICKG 2025) | 分析 LLM 长文档下的"记忆漂移"，**可作为"DocRE 长文档问题仍存在"的引证素材** |

### 第三档：领域应用顺手引（3 篇，开题不用看）

- LLMs-Based Documents Classification (UBMK 2025)
- Brazilian Judicial Healthcare NER (ISCC 2025)
- EarthSE (2505.17139)

## 🔧 工具与 API 备注

- **OpenAlex**：不限流，覆盖正式期刊/会议引用，但**滞后 6-12 个月**，arXiv 预印本之间的引用拿不到。**裸 `search` 查询会跨学科污染，见 `03-trends.md` 顶部的口径警告**
- **Semantic Scholar**：覆盖 arXiv 引用，但**公开 API 严格限流（~100 次/5min/IP）**
- **建议**：申请免费的 SS API Key（[这里](https://www.semanticscholar.org/product/api#api-key-form)）后限流升至 ~1000 次/min

