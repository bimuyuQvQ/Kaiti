# 双尺度检索环境向量条件化查询改写：可执行实验方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-24
- Verification Status: UNVERIFIED
- Version Label: env_rewriter_code_plan_v1

## 1. Experiment Overview

- **Title**：基于全局知识库环境向量与局部检索反馈向量的查询改写
- **Objective**：验证检索到的文档集合能否形成对查询改写有增量价值的连续向量表示，并在固定检索器下提高未见知识库上的检索效果。
- **Type**：ETL + retrieval benchmark + QLoRA-SFT + statistical analysis
- **主任务**：模型根据对话历史和检索环境，自由生成一个新的检索查询。
- **明确排除**：
  - 不做 `rewrite / lastturn / all_questions` 三选一；
  - 不做全量微调；
  - 不做 GRPO、PPO 等强化学习；
  - 第一阶段不更新检索器参数；
  - 第一阶段不同时改变检索器和知识库。

### 1.1 核心研究问题

**RQ1：局部反馈是否有效？**

初始查询检索出的 Top-k 文档集合，经冻结编码器聚合成局部反馈向量后，能否帮助模型生成比普通 query-only rewriter 更好的自由查询？

**RQ2：全局知识库画像是否有额外价值？**

对固定探针返回的文档集合进行聚合得到的全局环境向量，能否在局部反馈向量之外继续提高检索效果？

**RQ3：是否能泛化到未见知识库？**

环境编码器和改写模型在训练时未见目标知识库，只根据目标库的无标签文档和探针结果构建环境向量时，能否获得稳定增益？

**RQ4：向量压缩是否优于直接文本反馈？**

将 Top-k 标题和片段直接放入提示词，通常信息更完整。向量条件化是否能在明显减少输入 token 和延迟的同时保持或提高检索效果？

### 1.2 主要假设与停止条件

| 编号 | 假设 | 通过条件 | 失败后的解释 |
|---|---|---|---|
| H0 | 自由查询候选中存在可学习增益 | best-of-8 相对基线平均 nDCG@10 ≥ +0.03 | 候选生成器不足，暂不训练 |
| H1 | 局部文档集合向量有用 | `Local-Vector` 相对 `Query-only SFT` ≥ +0.01，且 95% CI 下界 > 0 | 当前文档聚合不能提供可靠反馈 |
| H2 | 全局知识库向量有额外价值 | `Global+Local` 相对 `Local-Vector` ≥ +0.005 | 只能声称 relevance feedback，不能声称知识库感知 |
| H3 | 可泛化到未见知识库 | 至少 3/4 个 MTRAG 库提高，且任一库下降不超过 0.01 | 环境表示只记忆训练库或不稳定 |
| H4 | 向量压缩有成本优势 | 性能不劣于文本反馈 0.002，且反馈输入 token 至少减少 50% | 直接文本反馈更合理，不保留 soft-token 机制 |
| H5 | 环境向量不是伪标签或身份泄露 | shuffled-vector 不优于 query-only；corpus-ID 在未见库上无效 | 若负对照也提高，训练或评测存在泄漏 |

只有 H0 通过才进入 SFT。只有 H1 通过才继续做 H2。只有 H2 通过，论文主张中才保留“知识库环境向量”。

## 2. 方法定义

### 2.1 普通查询改写基线

给定对话历史或用户问题 \(H\)，普通改写器生成初始查询：

\[
q_0 = G_0(H)
\]

其中 \(G_0\) 是冻结的初始改写器，使用固定提示、`temperature=0` 和固定 checkpoint。
所有 B3–B8 方法共享离线缓存的同一个 \(q_0\) 及其 Top-5，不能各自生成不同的初检查询。
否则比较会混入初始检索质量差异。

`Query-only SFT` 只接收 \(H\)，不接收检索结果、知识库 ID 或环境向量；它与环境方法使用
完全相同的训练目标 \(q^*\)。

### 2.2 全局知识库环境向量

为所有知识库使用同一组 \(P=64\) 个固定探针查询。对知识库 \(K\)：

\[
D_{j,K} = R_K(p_j), \quad j=1,\ldots,P
\]

每个探针返回 Top-5 文档。使用冻结的 `BAAI/bge-base-en-v1.5` 编码：

\[
e_{j,i}=E(d_{j,i}),\qquad e^q_j=E(p_j)
\]

每个探针的返回集合先聚合为：

\[
u_j =
\operatorname{Mean}_{i=1}^{5}
\left[
e_{j,i};
\cos(e^q_j,e_{j,i});
\operatorname{rank}_i;
\widetilde{s}_{j,i}
\right]
\]

再使用小型 DeepSets 编码器：

\[
z_K =
\rho\left(
\frac{1}{P}\sum_{j=1}^{P}\phi([e^q_j;u_j])
\right)
\]

默认配置：

- BGE embedding：768 维，冻结；
- \(\phi\)：`Linear → GELU → Linear`，输出 256 维；
- \(\rho\)：`Linear → GELU → LayerNorm`，输出 256 维；
- 全局向量 \(z_K\)：256 维；
- 探针 Top-k：5；
- 检索分数在每个探针内部做 z-score，避免不同知识库分数量纲泄露。

离线阶段保存的是冻结 BGE 产生的逐探针张量以及 rank/score，而不是随机初始化
DeepSets 产生的最终 \(z_K\)。\(\phi\) 和 \(\rho\) 在 SFT 阶段训练。

### 2.3 查询相关的局部反馈向量

用基线查询 \(q_0\) 检索：

\[
D_0=R_K(q_0)=\{d_1,\ldots,d_5\}
\]

文档向量通过 query-aware attention 聚合：

\[
a_i =
\operatorname{softmax}
\left(
w^\top\tanh(W_qE(q_0)+W_dE(d_i)+W_rr_i+W_ss_i)
\right)
\]

\[
z_{\text{local}}=\sum_i a_iW_vE(d_i)
\]

默认配置：

- Top-k：5；
- 输出：256 维；
- 输入包含文档 embedding、归一化 rank、归一化检索分数；
- 不输入 qrels、gold passage 标识或答案；
- BGE 冻结，只训练 attention pooling 层。

### 2.4 Soft-token 注入

分别将全局和局部向量投影成 4 个 soft tokens：

\[
P_K(z_K)\in\mathbb{R}^{4\times d_{\text{model}}}
\]

\[
P_L(z_{\text{local}})\in\mathbb{R}^{4\times d_{\text{model}}}
\]

模型输入排列：

```text
[GLOBAL_SOFT_1 ... GLOBAL_SOFT_4]
[LOCAL_SOFT_1  ... LOCAL_SOFT_4]
<用户问题与对话历史>
```

实现要求：

- 使用 Hugging Face `inputs_embeds`；
- soft-token 位置的 label 统一设为 `-100`；
- attention mask 和 position IDs 覆盖 soft tokens；
- base LM 使用 LoRA/QLoRA；
- 环境编码器和 soft-token projector 正常训练；
- BGE 与 BM25 冻结；
- 输出格式只包含：

```text
<rewrite>自由文本查询</rewrite>
```

允许模型输出与 \(q_0\) 相同的查询，相当于隐式 KEEP，但不增加离散动作分类头。

### 2.5 训练目标

主目标是查询文本的 token-level SFT：

\[
\mathcal L_{\text{rewrite}}
=-\log p_\theta(q^*\mid H,z_K,z_{\text{local}})
\]

第一版不增加 DPO 和 RL。若主实验通过，可增加一个辅助效用预测头：

\[
\mathcal L =
\mathcal L_{\text{rewrite}}
+0.2\mathcal L_{\text{utility}}
\]

其中 utility 只预测候选相对 \(q_0\) 是 `improve / neutral / harm`。辅助头属于后续消融，不进入首轮训练。

## 3. 数据设计

### 3.1 角色划分

| 角色 | 数据 | 用途 |
|---|---|---|
| 训练 | RAGBench 中具有 qrels 的训练问题，按子数据集保留 corpus 边界 | 构造 8k–12k 条 SFT 样本 |
| 开发 | 从训练 corpus 中按 query 分层划分 10% | 早停、超参数和候选阈值 |
| 主测试 | MTRAG FiQA、Cloud、Govt、ClapNQ，共 777 条 | 未见知识库泛化 |
| 辅助测试 | BEIR NFCorpus、SciFact、TREC-COVID，确保不与训练重叠 | 检查是否只适配 MTRAG |
| 负对照 | 打乱环境向量与 corpus 的对应关系 | 检查环境信息是否真实有效 |

RAGBench 使用官方数据入口：

```text
https://huggingface.co/datasets/galileo-ai/ragbench
```

固定使用以下 11 个 config：

```text
covidqa
delucionqa
emanual
expertqa
finqa
hagrid
hotpotqa
msmarco
pubmedqa
tatqa
techqa
```

每个 config 最多确定性抽取 1,000 个 train queries，少于 1,000 时全部使用，目标总量约
8k–11k。每个 config 独立构建一个 corpus：对其中全部 context/passages 按规范化文本
SHA256 去重，以原始支持 passage 为 relevant document，沿用主基线论文对 RAGBench 的
retrieval adaptation 方式。不能把 11 个 config 再合并成单一 corpus，否则无法训练和
检验知识库条件。

### 3.2 数据泄漏规则

1. MTRAG 四库的 qrels 只能用于最终评测，不能用于：
   - 选择候选；
   - 训练环境编码器；
   - 调阈值；
   - 选择 checkpoint。
2. 全局环境向量允许读取目标知识库的无标签文档，因为部署时知识库本来可见。
3. 固定探针不能来自 MTRAG 测试问题。
4. 候选生成时不能向教师模型提供 gold 文档。
5. 训练目标 \(q^*\) 只能从训练集 qrels 选择。
6. 同一 query 的所有候选必须位于同一数据划分，不能跨 train/dev。
7. 文档 embedding 缓存按 `corpus_sha256 + encoder_revision` 命名，避免索引混用。

### 3.3 固定探针集合

首版使用 64 个固定、非目标测试问题的英文探针，覆盖：

- 人物、地点、组织、时间事实；
- 定义和术语；
- 操作步骤；
- 原因和机制；
- 比较；
- 数字和金融；
- 法律/政策；
- 医疗/科学；
- 产品和故障排查；
- 多实体关系。

探针文件：

```text
experiments/environment_conditioned_rewriter/data/probes/probes_v1.jsonl
```

每条包含：

```json
{"probe_id": "p0001", "text": "...", "category": "definition"}
```

探针文件一旦开始正式实验即冻结，并记录 SHA256。

## 4. SFT 标签构造

### 4.1 生成自由查询候选

对每个训练问题生成 8 个自由文本候选。候选生成策略只用于增加搜索覆盖，不是最终模型的动作空间：

- 2 个低温自然改写；
- 2 个关键词/术语密集改写；
- 2 个文档式或陈述式改写；
- 2 个基于初检 Top-5 标题/短片段的反馈改写。

生成模型优先使用服务器已缓存的：

```text
reasonrag/Qwen2.5-7B-Instruct-ReasonRAG
```

采样配置：

```yaml
temperature: 0.8
top_p: 0.95
max_new_tokens: 128
num_candidates: 8
```

候选集合必须额外加入：

- 原始问题；
- 基线 \(q_0\)；
- 去重后的教师候选。

### 4.2 检索评分与目标选择

对每个候选使用同一 BM25 索引计算：

- nDCG@10；
- Recall@10；
- MRR@10。

目标选择：

1. 以 nDCG@10 最大为主；
2. nDCG 并列时选择 Recall@10 更高者；
3. 仍并列时选择更短的查询；
4. 如果最佳候选相对 \(q_0\) 的 nDCG 增益小于 0.02，则目标设为 \(q_0\)；
5. 所有候选得分和选择原因写入 JSONL，不只保留最终目标。

### 4.3 best-of-N 可学习空间门槛

训练前先计算：

| 上界 | 候选数 |
|---|---:|
| baseline | 1 |
| best-of-2 | 2 |
| best-of-4 | 4 |
| best-of-8 | 8 |

只有当 best-of-8 相对 baseline：

- 宏平均 nDCG@10 至少 +0.03；
- 至少 20% 查询有 ≥0.05 的增益；
- 增益不只来自单一 corpus；

才进入 SFT。否则先修复候选生成，不训练模型。

此外，所有方法的初始查询统一由冻结 \(G_0\) 生成并缓存到：

```text
artifacts/initial_queries/<split>.jsonl
```

每条记录必须包含 `query_id`、`corpus_id`、`q0`、生成模型 revision、prompt SHA256 和
解码配置。环境模型不得在训练时重新生成不同的 \(q_0\)。

## 5. 第一阶段实验矩阵

第一阶段固定 BM25，只改变知识库。

| ID | 方法 | 输入环境 | 是否训练 soft tokens | 作用 |
|---|---|---|---|---|
| B0 | Raw Question | 无 | 否 | 最低基线 |
| B1 | Baseline Rewriter | 历史/问题 | 否 | 原始模型零样本 |
| B2 | Query-only SFT | 历史/问题 | 否 | 公平训练基线 |
| B3 | Text Feedback SFT | Top-5 标题+每篇 80 tokens | 否 | 信息最完整的强基线 |
| B4 | Corpus-ID SFT | 训练库 ID embedding | 是 | 检查记忆效应；不能处理未见库 |
| B5 | Global-Vector SFT | \(z_K\) | 是 | 只看知识库总体画像 |
| B6 | Local-Vector SFT | \(z_{\text{local}}\) | 是 | 只看当前检索反馈 |
| B7 | Global+Local SFT | \(z_K+z_{\text{local}}\) | 是 | 主方法 |
| B8 | Shuffled Global+Local | 错配向量 | 是 | 必做负对照 |
| U1 | best-of-8 | 使用 qrels 事后选候选 | 不部署 | 候选集合上界 |

### 5.1 公平性控制

- B2–B8 使用完全相同的训练 query 和目标 \(q^*\)；
- 训练 epoch、LoRA rank、优化器和解码参数相同；
- B3 的文本片段来自与 B6/B7 相同的 Top-5；
- 所有方法最终只输出一个自由文本查询；
- 所有查询使用同一 BM25 索引重新检索；
- 不把训练时 best-of-8 的候选列表提供给测试模型。

## 6. 第二阶段：RAMP-like 多检索器扩展

仅当第一阶段 H1 或 H2 通过后执行。

### 6.1 检索器

- BM25；
- Contriever；
- `BAAI/bge-base-en-v1.5`。

### 6.2 环境表示

增加三类对比：

1. `Retriever-ID`：离散检索器 ID；
2. `RAMP-like`：同一 corpus 上固定 probes 的返回文档 ID Jaccard profile；
3. `Semantic Environment Vector`：不依赖共享文档 ID的文档内容集合向量。

### 6.3 关键划分

- seen retriever + seen corpus；
- unseen retriever + seen corpus；
- seen retriever + unseen corpus；
- seen retriever 和 corpus，但 unseen 组合；
- unseen retriever + unseen corpus。

第二阶段的主要目的不是追求总体最高分，而是判断：

> 语义文档集合向量是否比 RAMP 的文档 ID 重叠 profile 更适合跨知识库环境。

## 7. 模型与训练配置

### 7.1 当前服务器条件

已确认：

- PyTorch 2.4.0；
- CUDA 12.1；
- `torch.cuda.is_available() == True`；
- PyTorch 可见 8 张 GPU；
- 已缓存 `BAAI/bge-base-en-v1.5`；
- 已缓存 `reasonrag/Qwen2.5-7B-Instruct-ReasonRAG`；
- `/data1` 约剩余 2.8 TB。

异常：

- `nvidia-smi` 当前报告 NVML driver/library mismatch；
- PyTorch 仍可见 8 张 GPU，但正式训练前必须修复或至少验证单卡矩阵计算、显存分配和保存 checkpoint；
- NVML 修复前不能依赖 `nvidia-smi` 做监控，临时使用 PyTorch 显存日志。

### 7.2 QLoRA 配置

```yaml
model: reasonrag/Qwen2.5-7B-Instruct-ReasonRAG
quantization: nf4
compute_dtype: bfloat16
double_quant: true
max_input_tokens: 1024
max_output_tokens: 128
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
gradient_checkpointing: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
effective_batch_size: 32
epochs: 3
learning_rate_lora: 0.0002
learning_rate_environment_encoder: 0.001
weight_decay: 0.01
warmup_ratio: 0.05
lr_scheduler: cosine
max_grad_norm: 1.0
evaluation_steps: 250
save_steps: 250
save_total_limit: 2
seed: 20260724
```

每个运行默认占用一张 24 GB GPU。先单卡 smoke test，再并行运行 B2、B3、B5、B6；
B7 在这些运行正常后启动。首轮所有方法只跑一个种子，最终 B2、B3、B6、B7 中最强的两种补跑
3 个种子。

## 8. 代码结构与接口

计划新增：

```text
experiments/environment_conditioned_rewriter/
├── README.md
├── experiment_plan.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── candidate_generation.yaml
│   ├── train_query_only.yaml
│   ├── train_text_feedback.yaml
│   ├── train_global.yaml
│   ├── train_local.yaml
│   └── train_global_local.yaml
├── data/
│   ├── probes/probes_v1.jsonl
│   └── manifests/
├── src/env_rewriter/
│   ├── validate_environment.py
│   ├── prepare_data.py
│   ├── build_bm25.py
│   ├── generate_initial_queries.py
│   ├── generate_candidates.py
│   ├── score_candidates.py
│   ├── build_global_profiles.py
│   ├── build_local_feedback.py
│   ├── environment_encoder.py
│   ├── model.py
│   ├── collator.py
│   ├── train.py
│   ├── evaluate.py
│   └── analyze.py
└── tests/
```

大体积数据、索引、embedding 和 checkpoint 全部位于服务器：

```text
/data1/home/lmy/datasets/env_rewriter/
/data1/home/lmy/Kaiti/experiments/environment_conditioned_rewriter/artifacts/
```

不提交 Git，也不通过 SCP 传输 GB 级文件。小型配置、日志摘要和结果表通过 Git 或小文件 SCP。

## 9. 精确执行顺序与命令

以下命令是实现后的统一入口；每一步必须通过对应成功标准才能进入下一步。

### Step 0：环境健康检查

```bash
cd /data1/home/lmy/Kaiti
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
/data1/home/lmy/miniconda3/envs/retriever/bin/python \
-m env_rewriter.validate_environment \
--output experiments/environment_conditioned_rewriter/artifacts/env_check.json
```

成功标准：

- 8 张 GPU 被 PyTorch 枚举；
- 至少一张 GPU 完成 4096×4096 bf16 矩阵乘法；
- BGE 和 Qwen tokenizer/model config 可从本地缓存加载；
- 输出目录可写；
- 剩余磁盘 ≥1 TB。

### Step 1：准备数据清单

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
python -m env_rewriter.prepare_data \
--config experiments/environment_conditioned_rewriter/configs/data.yaml \
--output experiments/environment_conditioned_rewriter/artifacts/data_manifest.json
```

成功标准：

- query、qrels、corpus ID 完整对齐；
- 每个 qrel 文档都存在；
- train/dev/test corpus 无意外交叉；
- 输出每个源文件 SHA256；
- MTRAG 测试 qrels 未进入训练目标文件。

### Step 2：构建 BM25 索引

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
python -m env_rewriter.build_bm25 \
--manifest experiments/environment_conditioned_rewriter/artifacts/data_manifest.json \
--output-root experiments/environment_conditioned_rewriter/artifacts/indexes
```

成功标准：

- 每个 corpus 的索引文档数等于 manifest；
- 随机抽查 20 个 query，返回 doc ID 全部有效；
- 同一索引重复运行 Top-10 完全一致。

### Step 3：冻结并缓存统一初始查询

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.generate_initial_queries \
--manifest experiments/environment_conditioned_rewriter/artifacts/data_manifest.json \
--model reasonrag/Qwen2.5-7B-Instruct-ReasonRAG \
--local-files-only \
--temperature 0 \
--output-root experiments/environment_conditioned_rewriter/artifacts/initial_queries
```

成功标准：

- 每个 train/dev/test query 恰好一个 \(q_0\)；
- 输出不为空且 `<rewrite>` 可解析；
- 生成模型 revision、prompt SHA256 和解码参数完整；
- 同一命令抽查 50 条重复运行完全一致；
- 后续 B3–B8 只读取缓存，不重新生成 \(q_0\)。

### Step 4：生成并评分候选

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.generate_candidates \
--config experiments/environment_conditioned_rewriter/configs/candidate_generation.yaml \
--initial-queries experiments/environment_conditioned_rewriter/artifacts/initial_queries/train.jsonl \
--split train \
--output experiments/environment_conditioned_rewriter/artifacts/candidates/train.jsonl
```

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
python -m env_rewriter.score_candidates \
--manifest experiments/environment_conditioned_rewriter/artifacts/data_manifest.json \
--candidates experiments/environment_conditioned_rewriter/artifacts/candidates/train.jsonl \
--output experiments/environment_conditioned_rewriter/artifacts/labels/train_scored.jsonl \
--summary experiments/environment_conditioned_rewriter/artifacts/labels/best_of_n_summary.json
```

硬门槛：H0 通过，否则停止训练并检查候选多样性。

### Step 5：构建全局和局部向量

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.build_global_profiles \
--manifest experiments/environment_conditioned_rewriter/artifacts/data_manifest.json \
--probes experiments/environment_conditioned_rewriter/data/probes/probes_v1.jsonl \
--encoder BAAI/bge-base-en-v1.5 \
--local-files-only \
--output-root experiments/environment_conditioned_rewriter/artifacts/global_profiles
```

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.build_local_feedback \
--manifest experiments/environment_conditioned_rewriter/artifacts/data_manifest.json \
--queries experiments/environment_conditioned_rewriter/artifacts/labels/train_scored.jsonl \
--output-root experiments/environment_conditioned_rewriter/artifacts/local_feedback
```

表示健康检查：

- 对保存的冻结 BGE 逐探针张量做确定性 mean-pooling reference；
- 用两组互斥的 32 probes 构建同一环境 reference，平均 cosine ≥0.80；
- 同一环境的 probe-half 相似度应高于不同环境平均相似度至少 0.10；
- embedding 不含 NaN/Inf；
- 打乱文档后 profile 明显变化；
- 结果文件记录 encoder revision 和 probe SHA256。

若稳定性不通过，不训练 rewriter，先调整 probe 或 pooling。

### Step 6：32 样本端到端 smoke test

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.train \
--config experiments/environment_conditioned_rewriter/configs/train_global_local.yaml \
--max-train-samples 32 \
--max-steps 20 \
--output-dir experiments/environment_conditioned_rewriter/artifacts/smoke/global_local
```

成功标准：

- loss 有限且总体下降；
- LoRA、environment encoder、projector 均有非零梯度；
- BGE 和量化 base LM 无梯度；
- checkpoint 能保存并重新加载；
- 同一输入加载前后生成完全一致；
- soft-token attention mask、labels 和 position IDs 单元测试通过。

### Step 7：正式 SFT

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.train \
--config experiments/environment_conditioned_rewriter/configs/train_query_only.yaml
```

其余 B3、B5、B6、B7、B8 使用对应配置运行。每个训练作业：

- 硬超时 24 小时；
- 每 60 秒记录 process alive、loss、学习率、PyTorch allocated/reserved memory；
- 连续 3 次评估无改进只报警，不自动停止；
- OOM、NaN 或非零退出不自动重试。

### Step 8：未见知识库评测

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
CUDA_VISIBLE_DEVICES=0 python -m env_rewriter.evaluate \
--manifest experiments/environment_conditioned_rewriter/artifacts/data_manifest.json \
--models-root experiments/environment_conditioned_rewriter/artifacts/models \
--split mtrag_test \
--output experiments/environment_conditioned_rewriter/artifacts/evaluation/mtrag_predictions.jsonl \
--summary experiments/environment_conditioned_rewriter/artifacts/evaluation/mtrag_summary.json
```

测试解码固定：

```yaml
do_sample: false
temperature: 0
max_new_tokens: 128
num_return_sequences: 1
```

### Step 9：统计分析

```bash
PYTHONPATH=experiments/environment_conditioned_rewriter/src \
python -m env_rewriter.analyze \
--predictions experiments/environment_conditioned_rewriter/artifacts/evaluation/mtrag_predictions.jsonl \
--bootstrap-samples 10000 \
--seed 20260724 \
--output-dir experiments/environment_conditioned_rewriter/artifacts/analysis
```

## 10. 指标与统计方案

### 10.1 主指标

- nDCG@10；
- Recall@10；
- MRR@10。

主检验：

```text
Global+Local SFT vs Query-only SFT
```

使用逐查询配对差值、10,000 次 bootstrap，并同时报告：

- 均值差；
- 95% CI；
- win / tie / loss；
- 各 corpus 单独结果；
- corpus 宏平均。

### 10.2 必须单独报告的子集

1. **Initial-failure subset**：\(q_0\) 的 Recall@10 = 0；
2. **Initial-success subset**：\(q_0\) 已命中相关文档；
3. **大幅受益**：nDCG 增益 ≥0.05；
4. **有害改写**：nDCG 下降 ≥0.05；
5. 查询长度四分位；
6. 每个知识库。

这样可以区分：

- 模型是否真正救回失败检索；
- 还是只在已经检索成功的文档中复制词语；
- 是否通过大量冒险换取少数大增益。

### 10.3 成本指标

- 每个问题检索次数；
- BGE 编码时间；
- rewriter 延迟；
- 输入文本 token；
- soft token 数；
- 峰值显存；
- 索引与 embedding 缓存大小。

### 10.4 多重比较

预注册的唯一主比较是 B7 vs B2。其他比较标记为次要或探索性。对 B3、B5、B6、B8
相对 B2 的多重比较使用 Holm 校正；不只报告未经校正的最好结果。

## 11. Expected Outputs

| Output | Path | Format | Success Criterion |
|---|---|---|---|
| 环境检查 | `artifacts/env_check.json` | JSON | 所有硬门槛通过 |
| 数据清单 | `artifacts/data_manifest.json` | JSON | ID、qrels、SHA 完整 |
| 候选数据 | `artifacts/candidates/train.jsonl` | JSONL | 每 query 至少 6 个去重候选 |
| 候选评分 | `artifacts/labels/train_scored.jsonl` | JSONL | 每候选有三项 IR 指标 |
| best-of-N | `artifacts/labels/best_of_n_summary.json` | JSON | H0 判定完整 |
| 全局 profile | `artifacts/global_profiles/*.npz` | NPZ | 无 NaN，包含版本元数据 |
| 局部反馈 | `artifacts/local_feedback/*.npz` | NPZ | 与 query ID 一一对应 |
| QLoRA adapter | `artifacts/models/<run>/` | safetensors | 可重载并生成 |
| 逐查询评测 | `artifacts/evaluation/mtrag_predictions.jsonl` | JSONL | 777 条齐全 |
| 统计摘要 | `artifacts/analysis/summary.json` | JSON | CI、逐库结果完整 |
| 论文表格 | `artifacts/analysis/main_table.csv` | CSV | B0–B8 指标齐全 |

## 12. 单元测试清单

至少覆盖：

1. BEIR/MTRAG 数据读取与 ID 对齐；
2. qrels 文档存在性；
3. BM25 重复检索确定性；
4. candidate 去重与 `<rewrite>` 解析；
5. best-of-N tie-break；
6. BGE embedding 缓存键；
7. DeepSets 对文档顺序置换不敏感；
8. local attention 权重和为 1；
9. soft-token shape、mask、position IDs；
10. soft-token labels 为 `-100`；
11. 冻结参数无梯度；
12. LoRA、projector、environment encoder 有梯度；
13. shuffled-vector 映射确实跨 corpus 打乱；
14. 测试 qrels 未出现在训练文件；
15. checkpoint 保存/加载一致；
16. nDCG、Recall、MRR 与已知 toy example 一致。

## 13. 资源与预计时间

| 阶段 | 资源 | 预计时间 |
|---|---|---:|
| 数据下载与校验 | CPU/网络 | 2–6 小时 |
| BM25 索引 | CPU/RAM | 1–3 小时 |
| 8 候选生成，约 10k queries | 1×3090，7B vLLM | 6–14 小时 |
| 候选检索评分 | CPU | 2–6 小时 |
| 全局/局部 BGE embedding | 1×3090 | 1–4 小时 |
| 单个 QLoRA-SFT | 1×3090 | 4–10 小时 |
| 首轮 5 个必要训练运行 | 可用 4–5 张 GPU 并行 | 约 10–20 小时墙钟时间 |
| MTRAG 全评测 | 1×3090 + CPU 检索 | 每模型 1–3 小时 |
| 最终 3 seeds | 2 个入选模型 | 约 12–30 小时 |

预计新增磁盘：

- 原始语料与索引：100–300 GB；
- BGE embedding：10–80 GB，取决于训练 corpus 数量；
- 候选与逐查询结果：5–20 GB；
- 每个 QLoRA adapter/checkpoint：2–8 GB；
- 总预算控制在 500 GB 内。

## 14. 风险与降级方案

### 风险 1：best-of-8 空间不足

处理：

- 检查教师查询是否高度重复；
- 增加温度和提示多样性；
- 增加基于初检文档的反馈候选；
- 最多扩展到 best-of-16；
- 仍不足则停止，不训练。

### 风险 2：全局 profile 不稳定

处理：

- probes 从 64 增加到 128；
- Top-k 从 5 增加到 10；
- 使用 robust mean 或 attention pooling；
- 检查收益是否仅来自 corpus ID；
- 若仍不稳定，删除全局向量，只保留局部反馈。

### 风险 3：局部向量不如文本反馈

如果 B3 明显优于 B6/B7：

- 论文主方法改为语义检索反馈 SFT；
- 向量压缩作为失败消融；
- 不强行保留 soft-token 机制。

### 风险 4：只在已经命中 gold 的样本上提升

如果 initial-failure subset 无提升：

- 当前方法只是重述已有证据，不能声称修复检索失败；
- 增加 query–document mismatch 监督；
- 或把研究范围收缩为低成本反馈压缩，不声称召回改善。

### 风险 5：未见知识库失败

如果 seen corpus 有效、unseen corpus 无效：

- 检查 profile 是否编码 corpus ID；
- 增加 profile dropout；
- 用更多训练 corpus；
- 将结论限定为轻量知识库适配，而非零样本泛化。

### 风险 6：7B QLoRA 不稳定或太慢

降级：

- 使用 Qwen 1.5B–3B 指令模型；
- 保持 BGE、环境编码器、数据和评测不变；
- 先完成 Query-only、Local、Global+Local 三个必要运行；
- Text Feedback 和多种 projector 作为后续消融。

## 15. 最小可毕业版本

若时间紧张，只完成以下闭环：

1. BM25 + 四个 MTRAG 知识库；
2. 一组冻结的 64 probes；
3. BGE 文档集合向量；
4. Query-only SFT；
5. Text Feedback SFT；
6. Local-Vector SFT；
7. Global+Local SFT；
8. Shuffled-vector 负对照；
9. nDCG@10、Recall@10、逐库和 failure-subset 分析；
10. 一次 7B QLoRA 训练和最终两个方法的 3 seeds。

这个版本足以形成一章完整的硕士论文：

- 有明确基线；
- 有连续环境表示；
- 有自由查询生成；
- 有检索监督；
- 有未见知识库测试；
- 有负对照和消融；
- 即使方法失败，也能给出“全局画像、局部反馈和文本反馈谁真正有效”的实证结论。

## 16. 主要来源

- RAMP: Adapting One Agent to Multiple Retrievers via Behavioral Probing
  https://openreview.net/pdf/97882921bc667f2c7e987352922262b5bca09e40.pdf
- Understanding the Behaviors of Environment-aware Information Retrieval
  https://arxiv.org/pdf/2606.16817
- RAGBench 官方数据集
  https://huggingface.co/datasets/galileo-ai/ragbench
- RAGBench 官方论文
  https://arxiv.org/abs/2407.11005
