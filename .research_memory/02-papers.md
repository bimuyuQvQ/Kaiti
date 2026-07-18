# 论文池

> 何时更新：新增论文、引用数更新、读完一篇后填观点
>
> 引用数标注：[OA: N] = OpenAlex，[SS: N] = Semantic Scholar

---

## 🆕 ETC/CURA 近邻工作（2026-07-17，当前主线）

> 以下论文已通过官方会议页、OpenReview 或作者 arXiv 页面核对；本轮用于新颖性边界，不代表均已下载 PDF。详细方法定位见 `08-etc-cura.md`。

- **ETC**：*Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG*（AAAI 2026）。当前底座；论文公式与开源代码存在差异，alpha 按数据集调节。
- **DRAGIN**：*Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models*（ACL 2024）。已讨论动态 when/what，查询使用注意力 QFS；ETC 查询机制主要沿用它。
- **SKR**：*Self-Knowledge Guided Retrieval Augmentation for Large Language Models*（Findings of EMNLP 2023）。明确观察到检索知识有时会降低原回答质量；问题级 knowledge boundary 不是新空白。
- **Self-RAG**：*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*（ICLR 2024）。已联合学习 Retrieve、Relevant、Supported、Useful 等反思信号。
- **CRAG**：*Corrective Retrieval Augmented Generation*（arXiv:2401.15884）。检索后质量评估、纠正和无关内容过滤；普通 post-retrieval filter 不能作为主要创新。
- **Adaptive Retrieval Without Self-Knowledge?**（ACL 2025）。系统比较 35 种自适应检索/不确定性方法，指出检索可能引入无关信息，同时强调效率评测。
- **Pandora’s Box or Aladdin’s Lamp**（ACL 2025）。构建 RAG 噪声分析，显示噪声有有益与有害类型；不能把“噪声”简单等同于负效应。
- **D²-RAG**：*Dual-Decision Retrieval-Augmented Generation via Multi-Dimensional Uncertainty and Utility-Aware Decoding*（Findings of ACL 2026）。已做是否检索与噪声上下文处理的双决策，构成直接新颖性威胁。
- **S2G-RAG**：*Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA*（ACL 2026）。已做证据充分性、结构化 gap 和下一查询，且压缩多轮噪声；“gap query”不能单独作为贡献。
- **QuCo-RAG**：*Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation*（Findings of ACL 2026）。直接批评 entropy/logit 等内部信号因失校准和自信错误而不可靠，是 ETC 代理信号诊断的重要强基线。
- **CUE-R**：*Beyond the Final Answer in Retrieval-Augmented Generation*（arXiv:2604.05467）。通过 REMOVE/REPLACE/DUPLICATE 干预研究单条证据效用；CURA 必须突出生成中间状态、skip/多查询动作和跨时间校准，而非泛称“首次反事实效用”。
- **GRIP**（ACL 2026）与 **ReaLM-Retrieve**（SIGIR 2026 接收预印本）：已分别通过控制 token、上下文 bandit/QueryGen 联合何时与查询；CURA 不能声称首次联合 when/what。

---

## 🆕 agent 自演进 / PRM 方向论文池（2026-06-10 新增，对应新主候选研究内容二）

> 用 2026-06 web 检索核到，未精读。研究内容二（步骤级归因/PRM）的起点。

> **2026-06-10 已下载到 `papers/`（gitignore，不入库）+ 已核验 arXiv 真实存在**：PRM 综述、Who&When、CAR、ECHO 四篇 PDF 在本地。

**先读 4 篇（定盘）**：
- **PRM 综述**：*A Survey of Process Reward Models: From Outcome Signals to Process Supervisions* (arXiv:2510.08049, SJTU Weinan Zhang 组)。✅已核验+下载。PRM 全景：生成过程数据→训 PRM→test-time scaling / RL 闭环。**注意：PRM 重心是 reasoning(math/code)，agent 只是它列的应用之一（math/code/text/multimodal/robotics/agents），不是 PRM 的主场。**
- **Who&When**：*Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems* (ICML 2025, **arXiv:2505.00212**)。✅已核验+下载。**真正的归因 benchmark**，LLM-judge 步骤级准确率仅 ~14%（反水硬难度的源头）。⚠️ 上轮误把 ECHO(2510.04886) 当成 Who&When，已纠正。
- **Causal Agent Replay (CAR)** (arXiv:2606.08275, CMU 单作者短文)。✅已核验+下载。用 do(·) 干预+重放做失败步骤归因，反驳 LLM-judge 归因。开源。
- **ECHO**：*Where Did It All Go Wrong? Hierarchical Multi-Agent Error Attribution* (arXiv:2510.04886)。✅已核验+下载。Who&When 上的层次化归因方法。

**CAR related-work 里挖到的同方向（未下载，待按需）**：AgenTracer (arXiv:2509.03312, counterfactual replay 标注失败轨迹)、Ma et al. *Automatic Failure Attribution & Critical Step Prediction via Causal Inference* (arXiv:2509.08682)。
**AgentPRM**：仅核到 WWW 2026 ACM 收录 (dl.acm 10.1145/3774904.3792551)，**arXiv 号未独立核验**，别当真号引用，要用先查。

### 2026-06-10 推荐方向定调 + 综述/基线（已核验+下载到 `papers/`）

> **AI 最终推荐方向**：agent「步骤级过程奖励 / 自我纠错」——核心是**训判别式 verifier/PRM（对比学习，复用 B 会手艺）+ 推理时用**，**不做全程 RL**（RL 不稳，违背"省事毕业"）。次选：tool-use 可靠性（ToolACE-8B/xLAM/Hammer 全开权重 + BFCL V4，最稳最干净基线）。理由表见本轮对话/`04-decisions`。

**综述**：
- `SURVEY_SelfEvolvingAgents_TMLR2026_arXiv2507.21046.pdf`（TMLR 2026，方向大图，✅核验+下载）
- `PRM_Survey_2025_arXiv2510.08049.pdf`（方法族，已有）

**基线**（按"权重>代码、26>25、无代码不要"排序）：
- **PRIME** (arXiv:2502.01456, 清华/上海AI Lab)：2025，✅**开权重**(Eurus-2-7B-PRIME)+数据+代码(`PRIME-RL/PRIME`)。杀手锏=**只用结果标签在线训 PRM，免逐步标注**。偏 math/code。→ 首选起点。`BASELINE_PRIME_*.pdf` 已下。
- **StepPO** (arXiv:2604.18401, AgentR1/StepPO)：**2026**，⚠️开代码无权重。把 agent RL 提到**步骤级信用分配**(step-level GAE)，基于 veRL+vLLM。→ 最新+最 agentic，第二。`BASELINE_StepPO_*.pdf` 已下。
- **空白即机会**：目前无"2026+开权重+专做 agent 过程奖励"的现成模型 → 论文空间在此。
- **评测 benchmark**：ProcessBench(推理) / Who&When(agent 失败归因) / BFCL V4 Agentic(tool-use)。
- **次选(tool-use)基线**：ToolACE-8B(开权重, Team-ACE@HF)、xLAM、Hammer；榜 BFCL V4 Agentic(2025-07)。

**背景/扩展（按需）**：
- *A Survey of Self-Evolving Agents* (TMLR 2026)：what/when/how/where to evolve，"自演进"招牌的综述靠山。
- agent memory 三段式（Storage→Reflection→Experience）：arXiv:2605.06716、2603.07670、2512.16301（adaptation: post-training/memory/skills）。Ch2"知识提炼/经验沉淀"靠山。
- *Conformal Agent Error Attribution* (arXiv:2605.06788)：带覆盖保证的归因区间，进阶。
- 约束化 workflow（对应研究内容一/三，备用）：**MermaidFlow-CF** (openreview INX9FhqbUM) / MermaidFlow / *Agentic AI Architectures & Evaluation* (arXiv:2601.12560)。flow engineering / 显式图约束执行。
- **公开 benchmark**（研究内容二可用，不依赖指控数据）：Who&When、ProcessBench、AgentProcessBench、PRM800K。

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
**全文版精读笔记**：[`notes-ttmre.md`](./notes-ttmre.md)（含完整公式、扩展观察、复现坑预警）

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

---

## 🔥 2025-2026 DocRE 顶会论文清单（2026-05-27 调研，回应"方向是否过时"）

> **调研动机**：导师担心 TTM-RE (2024) 之后没人在 DocRE 这条线了。
> **结论**：方向**完全没死**——2025 顶会主会 + Findings 累计 ~13 篇 DocRE。

### 顶会主会 / 长文（4 篇，直接驳斥"过时"担忧）

| # | 简称 | 会议 | 切入点 | 代码 | 对我们的影响 |
|---|---|---|---|---|---|
| 1 | **DRELL** | **NAACL 2025 Long** | LLM as refiner（task distribution + probability fusion）。比已有 LLM 方法 +25.2% F1，SOTA | [Drasick/Drell](https://github.com/Drasick/Drell) ⭐3 | 🔴 **抢了 Y2/Y3 路线**——单纯做"LLM verifier"已没新意 |
| 2 | **SciNLP** | **EMNLP 2025 Main** | 科学文献领域 entity + RE 新 benchmark | ? | 🟡 可考虑作为第二章的领域扩展 |
| 3 | **CaDRL** | **COLING 2025 Main** | 上下文感知可微规则学习 | [aclanthology.org/2025.coling-main.551](https://aclanthology.org/2025.coling-main.551/) | 🟡 规则路线，跟我们方向不冲突 |
| 4 | **Re2-DocRED** | **EACL 2026 Long** | Joint Entity-Relation Extraction，用 LLM + reasoning + schema 增强 Re-DocRED（+27% triplets） | [klassessg/re2-docred](https://github.com/klassessg/re2-docred) | 🟢 **方案 C（联合抽取）的直接 motivation + 新增强数据集** |

#### 📖 Re2-DocRED 精读纠正（2026-06-03，读 §1/§3/§4/§5）

> ⚠️ **纠正一个之前记忆库 + `目前搞法以及todo.md` 里的根本性误解**：Re2-DocRED **不是 JERE 方法论文，是数据集论文**。第二章不能说"用 Re2-DocRED 的 JERE baseline"。

- **它的真正卖点（§4 contribution）**：**SiftingLogic**——一个**免训练的 LLM 标注 pipeline**，专门补回 Re-DocRED/DocGNRE 里漏标的**假阴性（False Negative）三元组**。
  - Stage 0：小 LLM 检索相关关系 R* + NER 抽 entity
  - Stage 1：按"关系→兼容 head/tail 实体类型"映射表生成候选三元组 T1
  - Stage 2：小 LLM 用**实体级约束（entity-level constraints，源自 Wikidata 关系定义）**+ 关系 verbalization 逐条校验
  - Stage 3：大 LLM 严格复验 → T3；再 5 人人工验证（Krippendorff α=0.88）+ inverse/co-occurring 规则扩展 → T4
  - 产出：Re2-DocRED 数据集（Re-DocRED +27% triplets / DocGNRE test +49.89% / REDFM 中文 +109.8%）
- **它自己不提出 JERE 模型**。§5 的"JERE baseline"是**拿现成模型来跑**，用于证明"补全数据后旧模型 recall 掉"：
  - **AutoRE (Xue et al. 2024)** —— 被称为 "state-of-the-art document-level JERE model"，**LLM-based**
  - **TaG (Zhang et al. 2023)**、**REBEL (Huguet Cabot & Navigli 2021)**
  - 这三个才是**第二章真正要打的 baseline**；数字很低（AutoRE 在新 test 上 P≈47-67 / R≈30-50），任务远未解决 → 有空间但 baseline 不弱、且是 LLM-based。
- **对方案 C/C' 的影响**：
  1. 第二章对照口径要改成"对标 AutoRE/TaG/REBEL（AutoRE = LLM-based SOTA）"，不是"对标 Re2-DocRED 的方法"。
  2. **两章卖点的真正交汇点是"假阴性 FN"**：TTM-RE 从模型侧扛 FN（SSR-PU + memory），Re2-DocRED 从数据侧补 FN（SiftingLogic）。`目前搞法` 里的"任务递进 + InfoNCE 平移"串法没串到卖点，FN 主线才是更硬的串联。
  3. Re2-DocRED 的"entity-level constraints"是可借用的现成思想（第二章 entity 侧约束/校验），且能避开 DRELL（DRELL refine 关系，这里约束实体）。
- **数据/代码 gate**：数据集已开源（github.com/klassessg/re2-docred）。✅ C' 的"数据集能不能用"这关基本过（仍需 clone 确认格式 + 是否含 mention/coref 标注供端到端用）。

#### 📌 AutoRE（第二章主对照 baseline，2026-06-03 下载 + 调研）

- **出处**：Xue, Zhang, Dong, Tang，**ACL 2024 Demo track**（arXiv 2403.14888，清华）
- **代码**：https://github.com/THUDM/AutoRE ⭐ ｜ **PDF**：`papers/AutoRE_LLM_DocRE_2024_ACL_Demo.pdf`（不入库）
- **地位**：**LLM-based 文档级 RE 的 SOTA / 标准对照**——Re2-DocRED §5 和 EP-RSR(NAACL 2025 Findings) 都拿它当 baseline。第二章 JERE 真正要打的就是它（+ TaG / REBEL）。
- **核心方法 RHF（Relation-Head-Facts）**：分三步生成 ——（1）判文档出现哪些关系 →（2）每个关系找头实体 →（3）每个(关系,头)抽完整三元组。底座 **Mistral-7B + QLoRA**（PEFT），端到端生成式。
- **卖点**：不假设关系候选已知、不假设实体已知（贴近真实场景，区别于 ATLOP/TTM-RE 这类"实体已标好"的 encoder 方法）；Re-DocRED 上 LLM-based SOTA（比 TaG dev/test +10.03/9.03%）。
- **关键局限（对我们有利）**：LLM-based 方法整体**仍打不过 fine-tuned 小模型 SOTA**（EP-RSR/Re2-DocRED 均印证）→ 印证"DocRE 上 encoder 仍是主力，LLM 是追赶者"。

#### 🧵 两章卖点串联主线（2026-06-03 确立）：**对抗 DocRE 的假阴性/噪声监督**

> 第一章基线 TTM-RE 的卖点本就是"噪声/FN 鲁棒"（SSR-PU + memory）；第二章数据集 Re2-DocRED 的卖点是"补 FN"。两章应统一在 FN 主线下，而非"用了同一底座"的工程联系。

| | 设定 | 噪声/FN 来源 | 创新点 | 接续关系 |
|---|---|---|---|---|
| 第一章 | entity 已知（Re-DocRED）| **关系级 FN**（pair 对，关系没标）| TTM-RE entity-pair 表示上加 **FN 感知对比**（不盲推疑似 FN 的 pair）| —— |
| 第二章 | entity 未知（Re2-DocRED）| **+ 实体级 FN**（真实体被漏召回→它所有关系全丢，pipeline 永久不可恢复）| 同套 FN 鲁棒学习 + **用关系信号回流召回被漏掉的实体** + 吃 Re2-DocRED 补回的标签，对标 AutoRE | 同一 FN 鲁棒原则从关系级扩到实体级 |

> ⚠️ **2026-06-03 概念纠正（用户戳出）**：不要把第二章卖点说成"对 error propagation 鲁棒"——错误传播是 pipeline vs 端到端的轴，端到端本就为消除传播而生，说"对传播鲁棒"自相矛盾。FN（数据/监督问题）与 error propagation（架构问题）是两个独立轴，勿混。
>
> ⚠️⚠️ **2026-06-03 二次纠正（更重要，用户再戳）**：上一条把"实体级 FN"说成第二章招牌也夸大了，两点站不住：
> 1. **Re2-DocRED 补的是关系不是实体**：+27% 全是 triplet（Table 1/3），靠 inverse/co-occurring 关系规则 + 实体级约束校验三元组；Stage 0 的 NER 只为生成候选，**它不补漏标实体**。所以 Re2-DocRED 的卖点 = 关系 FN，**不是实体 FN**。
> 2. **"漏实体→关系全丢"现象存在，但在 DocRED 家数据上可能很小**：DocRED 实体是命名实体+Wikidata 链接、相对好认，标准 DocRE 还给 gold 实体+coref；真正难的是关系（长尾/跨句）。实体漏召回大概率不是主误差源。
> - **结论**：实体 FN 目前是**硬凑的桥，非文献既有结论**，别当第二章招牌（会被问"问题多大、谁说过、数据呢"）。
> - **第二章稳的招牌 = 关系 FN 在 JERE 难设定（实体未知）下的迁移 + 吃 Re2-DocRED 补回的关系标签**。
> - **判据（P0 复现后做）**：在 Re2-DocRED 跑 JERE baseline，分解漏掉的 triplet 里"实体没识别出来" vs "实体对了关系判错"的占比。实体漏召回占比大 → 实体 FN 可升级为招牌；占比小（更可能）→ 老实用关系 FN 迁移当招牌。
>
> ⚠️⚠️⚠️ **2026-06-03 三次纠正（用户再戳"补干净"）**：不要说 Re2-DocRED"把数据侧 FN 补干净了"——论文自己强调 FN 是顽疾、补不完（说 DocGNRE "remains incomplete, substantial FN persisting"，它只是再补一层）。它的补法有天花板：规则只覆盖 inverse/co-occurring 关系、LLM 有召回上限+幻觉、人工只留 5/5 全票（偏保守）。+27% 是相对量，无 ground truth 说残余 FN=0。
> - **正确表述**：Re2-DocRED = 目前公开数据里 **FN 最少的版本，但残余 FN 仍在**。
> - **修正后的对照设计（不依赖"补干净"假命题）**：把第一/二章看成 **FN 完整度的梯度**——Ch1 Re-DocRED（FN 更多）vs Ch2 Re2-DocRED（FN 更少）。问题：数据 FN 越被补全，模型侧 FN 鲁棒的收益是变小（主要靠数据）还是稳定（模型侧正交不可替代）。
>
> 📌 **元判断**：核心骨架（关系 FN 主线 + 跨"实体已知/未知"和"FN 多/少"两轴做对照）是稳的；但"叠加 vs 冗余""实体 FN""补干净"这些招牌级精确表述**全都还没数据支撑，不能当结论讲**。开题阶段只讲骨架，细节统一答"这是开题后第一批诊断实验要测的"。别在无数据时精雕话术（已被用户连续戳塌 3 处：对传播鲁棒 / 实体 FN / 补干净）。

### Findings（5 篇，比 B 会高半档）

| # | 简称 | 会议 | 切入点 | 代码 | 对我们的影响 |
|---|---|---|---|---|---|
| 5 | **GREP** | ACL 2025 Findings | 全局关系 + entity pair reasoning + 辅助任务"先预测所有可能关系" | [yanyi74/GREP](https://github.com/yanyi74/GREP) ⭐2 | 🟡 中等冲击——抢了 entity pair reasoning，但跟我们 retriever/对比学习路线不直接撞 |
| 6 | **AMTL** | ACL 2025 Findings | Adaptive Multi-Threshold Loss（plug-in loss，解决长尾问题）。在 TTM-RE 等模型上一致提升 | [xhm-code/AMTL](https://github.com/xhm-code/AMTL) ⭐3 | 🔴 **对方案 B' 重大冲击**——AMTL 已经把长尾 loss 改进做了，且明确说能 plug 到 TTM-RE 上。我们做方案 B' 必须找 AMTL 没覆盖的子问题 |
| 7 | **ET-MIER** | EMNLP 2025 Findings | Entity Type-guided 关键 mention 识别 + 证据检索 | [NEU-IDKE/ET-MIER](https://github.com/NEU-IDKE/ET-MIER) | 🟡 中等冲击——抢了 entity type 路线 |
| 8 | **EP-RSR** | NAACL 2025 Findings | Entity Pair-guided LLM-based DocRE（EPRF 范式） | [LookingYu/EP-RSR](https://github.com/LookingYu/EP-RSR) | 🟡 中——但明确说 "LLM-based DocRE still lags behind small models at SOTA" |
| 9 | **GLiM** | ACL 2025 Findings | Graph Transformer + LLM（生物医学 DocRE） | ? | 🟢 跟我们方向不冲突 |

#### 🆕 2026-06-03 补充：SOTA 时间线纠正 + 为什么基线锚在 2024 可辩护

> 用户质疑"TTM-RE / AutoRE 都是 2024 的，2026 难道没有更新 SOTA"。WebSearch 确认结论：

- **有更新的，但 DocRE 是慢车道（~13 篇/年），2024 基线不丢人。** 2025-2026 新工作：
  - **EP-RSR**（NAACL 2025 Findings）：LLM-based，**比 AutoRE +7.42 F1（DocRED）→ AutoRE 已非 LLM-SOTA**
  - **DRELL**（NAACL 2025）：LLM refiner，声称整体 SOTA
  - **AMTL**（ACL 2025 Findings）：plug-in 长尾 loss，**plug 在 TTM-RE 上能涨 → TTM-RE 仍是 2025 别人改进的活底座**
  - **DOREMI**（2026, Knowledge-Based Systems）：长尾去噪数据集
  - **DocKS-RAG**（2025/2026 OpenReview poster）：LLM + 文档级知识图谱 + **RAG** + hybrid-prompt tuning → ⚠️ **已占"Retrieval-Augmented DocRE"包装位，用户原标题策略需避开/区分**
- **为什么仍锚 2024 基线（可辩护）**：
  1. 第一章 TTM-RE = **可复现改进底座**（唯一公开权重），不是当绝对 SOTA 比；新 SOTA 进相关工作 + 对照即可
  2. 第二章 AutoRE = **Re2-DocRED（EACL 2026）自己用的 baseline**，跟随其评测协议天然合理；2026 论文仍用 2024 AutoRE 反证子方向节奏慢
- **待办**：DocKS-RAG / EP-RSR / DRELL 还没下载精读，需确认它们占的地盘（尤其 DocKS-RAG 的 RAG 包装重叠）。

### 其他主流会议（4 篇）

| # | 简称 | 会议 | 切入点 |
|---|---|---|---|
| 10 | MAUM | IJCNN 2025 | U-Mamba 替代 Transformer + 记忆增强 |
| 11 | Two-Stage Loss + Anaphor | IJCNN 2025 | 两阶段 loss + 指代消解 |
| 12 | DTPE | ICASSP 2026 | 文档树解析 + LLM 数据精修 |
| 13 | Coarse-to-Fine | ICASSP 2026 | 低资源 DocRE 的粗到细两阶段 |

### 🚨 关键发现

**1. 东北大学 Fu Zhang & Jingwei Cheng 团队是当前最活跃的 DocRE 团队**
- 2025 一年发了 **4 篇 DocRE**（DRELL @ NAACL Main / GREP @ ACL Findings / AMTL @ ACL Findings / ET-MIER @ EMNLP Findings / EP-RSR @ NAACL Findings = 实际 5 篇）
- 风格：每年从不同角度切入同一个 DocRE 任务，路线包括 loss 改进、LLM 协作、entity type、entity pair-level reasoning
- 暗示：DocRE 是个**有人 actively investing 的方向**，但**人少**（一年大部分论文出自同一团队），适合冷门方向做毕业论文

**2. 关于"小众 ≠ 过时"的对照**

| 方向 | 2025 顶会论文/年 | 卷度 | 毕业友好度 |
|---|---|---|---|
| RAG | ~240 | 🔴 红海 | 答辩追问深 |
| ICL | ~115 | 🔴 红海 | 撞车风险高 |
| **DocRE** | **~13** | 🟡 **冷门但活着** | ✅ **甜区** |
| EAE | ~7 | 🟢 已死 | 答辩会被问"为什么不做更前沿的" |

### 🎯 对我们改进点 Y 的重新评估（2026-05-27）

| Y | 原计划 | 2025 现状 | 是否还可做 |
|---|---|---|---|
| Y1 替换 retriever | TTM-RE 静态 memory 换 retrieved-doc memory | 没人做过 retrieval-augmented TTM-RE | ✅ 仍可做 |
| Y2 LLM verifier | TTM-RE + LLM 判真伪 | ❌ DRELL 已做（NAACL 2025），且做得更精致 | ❌ **不能直接做** |
| Y3 LLM reranker | TTM-RE top-K + LLM rerank | ❌ 同上，DRELL 的 probability fusion 已覆盖 | ❌ **不能直接做** |
| Y4 InfoNCE on memory | 在 TTM memory tokens 上加对比学习正则 | 没人做过 | ✅ 仍可做 |
| **Y5 长尾 retrieval** | （新增）retrieval-based long-tail DocRE | AMTL 做 loss 角度，**retrieval 角度空白** | ✅ **新机会** |
| **Y6 联合抽取扩展** | （新增）从 RE 扩到 NER + coref + RE | Re2-DocRED (EACL 2026) 提供新数据 | ✅ **新机会** |

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


---

## 2026-07-09 精读补充：LogicRAG（2508.06105v2）

**论文**：*You Don’t Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures*，arXiv:2508.06105v2，本地路径：`papers/acl2026/2508.06105v2.pdf`。

**一句话定位**：LogicRAG 是一种 **query-time / inference-time 的 GraphRAG 替代方案**：不为整个语料预构建知识图，而是对每个输入问题动态构造“查询逻辑依赖图”（Query Logic Dependency Graph），用子问题 DAG 来调度多步检索与生成。

**核心流程**：
1. 用 LLM 将复杂 query 分解为子问题集合 `P={p1,...,pn}`；
2. 用 LLM 判断子问题之间的逻辑先后关系，构造 DAG `G=(V,E)`；
3. 对 DAG 做拓扑排序，得到依赖一致的推理/检索顺序；
4. 按拓扑顺序贪心解决子问题，前序子问题答案作为后续检索条件；
5. 对同一拓扑 rank 的子问题做 unified query，减少重复检索（graph pruning）；
6. 用 rolling memory 对历史检索证据和中间答案做摘要压缩，控制上下文长度（context pruning）；
7. 最后基于所有中间答案和 memory compose 最终答案。

**关键观察**：
- 传统 GraphRAG 的图是 corpus-level graph，构图成本高、更新慢、且固定图未必适配当前 query。
- LogicRAG 的图是 query-level reasoning graph，图的用途从“组织知识库”变成“组织当前问题的推理计划”。
- 论文还指出 agentic RAG 容易出现“hesitation”：模型反复生成相似子查询。作者用 sampling without replacement 强制推进子问题批次，降低 token 成本。

**实验结果摘要**：
- 数据集：HotpotQA、2WikiMultiHopQA、MuSiQue，各取 1000 条验证集问题。
- Baseline：Zero-shot LLM、Vanilla RAG、KGP、G-retriever、RAPTOR、GraphRAG、LightRAG、HippoRAG、HippoRAG2。
- LogicRAG 在三组多跳 QA 上整体优于 baseline：例如 2WikiMQA string accuracy 64.7%，明显高于 HippoRAG2 的 50.0%；MuSiQue 30.4/37.5（Str/LLM Acc）也超过 HippoRAG2。
- 2WikiMQA 查询时效率：LogicRAG 平均 9.83s / 1777.9 tokens；比 VanillaRAG 慢，但比多个图方法省 token，且无离线构图成本。

**方法性质判断（重要）**：
- 这是 **prompt + 系统流程 + 检索调度 + 上下文管理** 的工作。
- 没有训练新模型，没有改 LLM 底层结构，没有新 embedding 模型，也没有可学习模块。
- 子问题分解、依赖边判断、统一查询生成、rolling memory 摘要、新子问题发现、最终 compose 基本都依赖 LLM prompting。
- 论文有算法符号（DAG、拓扑排序、rolling memory、merge/query rank），但数学深度主要是流程形式化，不是模型结构或优化目标。

**对本毕设的启发**：
- 可以作为“非 RL Agentic RAG / 推理时结构化检索”的重要相关工作和可借鉴模块。
- 它证明“动态推理结构 + 检索调度”本身能涨点，适合拿来设计 retrieval scheduling、子问题依赖建模、context pruning 的 baseline/ablation。
- 但如果直接复刻 LogicRAG 作为毕业论文一章，风险是：创新会被看成“纯 prompt 编排”，缺少训练、模型模块或更硬的优化目标。
- 更稳的用法：把 LogicRAG 当作流程骨架或对照，在其上增加**可训练/可量化的组件**，例如子问题依赖图质量评估器、检索步骤选择器、过程偏好模型、DPO/SimPO 训练的数据构造，或面向中文/专业场景的结构化检索诊断。

**可尝试方向**：
- 将 LogicRAG 的 DAG 规划与 ReasonRAG/ProRAG 的过程监督思想结合：不是只 prompt 出 DAG，而是对“子问题分解、依赖边、检索推进、上下文保留”这些步骤构造偏好数据或奖励信号。
- 做一个“训练增强版 LogicRAG”：保留 DAG + rolling memory 框架，但引入可训练的 verifier/ranker/policy 来判断子问题是否足够、依赖是否正确、是否需要继续检索。
- 论文叙事上可以作为反例/动机：纯 inference-time 流程有效但偏软，因此我们的贡献需要落在“流程结构 + 可训练过程监督/评价器”的结合上。
