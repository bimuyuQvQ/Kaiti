# 当前待办与下次切入点

> 何时更新：每轮对话结束时刷新（保留近 3 次）

---

## 🎯 当前阶段（2026-06-18 重调）：离线过程监督 for Agentic RAG

**背景**：2026-06-18 确认 8×3090 无法跑 online agentic RL（师姐：128 卡 × 3 天/轮）。ProRAG 不再作为可训练基线。RE/EE/NER 彻底放弃。

**新方向（路 A）**：**离线过程监督**（Offline Process Supervision for Multi-hop RAG）
- 基线：SFT-only agentic RAG（Qwen3-8B + 结构化推理格式微调）
- 改进：MCTS 构造离线过程偏好数据 → DPO / RFT 训练
- 全程无 online RL，8×3090 完全可行
- 参考论文：ReasonRAG（arXiv:2505.14069）、ProRAG Stage 1-3

**ProRAG 的新用途**：
- 发布权重（`bmbgsj/ProRAG`）作为**上界参考**（inference only，不训练）
- 其 Stage 1-3 的设计作为**方法参考**

**方向待确定事项**：
1. 具体贡献点：数据构造质量 / PRM 设计 / 训练目标选哪个环节
2. GPT-4o 标注成本能否接受（ReasonRAG 的 MCTS 对比对需要 LLM 打标签）
3. 与导师对齐：接受"不做 RL"方案吗

---

### ✅ TODO（按优先级排列）

#### 🔧 环境 & 数据（实验室服务器，进行中）
- [x] **【P0】** 修复 torch CUDA 版本问题（已跑通推理）
- [x] **【P0】** 下载 wiki18_100w.jsonl + QA 数据集
- [x] **【P0】** 建索引（bge_Flat.index 已完成）
- [~] **【P0】** 跑通 ReasonRAG 推理（hotpotqa 已跑到约 3000/7405，被 OOM kill 中断）→ **需从头重跑**（FlashRAG 不支持断点续跑，全跑完才写盘）
- [~] **【P0】** 复现论文推理指标（HotpotQA）：当前 `em=33.21 / f1=43.96`，低于论文 `38.4/48.9`，需做配置对齐排查（iter/topk/评测 split/检索一致性）。
  - 已在 `inference.py` 增加 `--gpu_id` 与 `--run_tag`，支持多卡并行跑不同评测集/参数组合。
- [~] **【P1】** **SimPO 全量微调进行中**（2026-06-24）：ZeRO-3 + 8 GPU（0-7）+ chunked logps + lowmem DS config。
  - 已定位并修复 OOM 根因：`dpo/trainer.py` 在 `concatenated_forward` 里把 `labels` 传进了 `model.forward`，触发 Qwen2 内部 CE loss，导致额外显存峰值（约 +2.3GB）并在 rank1 OOM。
  - 修复：forward 时移除 `labels`，仅保留 logits 路径，再由 SimPO/DPO trainer 自己算偏好损失。
  - 8 卡 + `cutoff_len=2048`：`/data1/home/lmy/Kaiti/logs/simpo_full_z3_8gpu_2048.log`，在 rank1 再次 OOM 后退出。
  - 8 卡 + `cutoff_len=1536`：`/data1/home/lmy/Kaiti/logs/simpo_full_z3_8gpu_1536.log`，已稳定跑到 step 11（step10：loss=0.9776，reward accuracy=55%），当前运行中。
  - 检查点目录：`/data1/home/lmy/Kaiti/baselines/ReasonRAG-main/saves/qwen2.5-7b-instruct/full/dpo`

#### ⚠️ 训练显存红线（2026-06-23 踩坑记录，2026-07-01 更新）
- ❌ **ZeRO-3 + sigmoid DPO 全量**：每卡 22GB baseline + lm_head all-gather，OOM
- ❌ **DPO+LoRA 1–4 卡 ZeRO-2/3**：step 0 OOM（每卡 ~22GB 顶满）
- ❌ **ZeRO-3 + offload_param + offload_optimizer**：CPU 内存炸，触发 OOM-killer
- ❌ **Liger Kernel + DPO**：transformers 4.57 下 `.logits` 变 None，与 ZeRO-3 不兼容
- ✅ **DPO+LoRA+ZeRO-3（8 卡）已验证**（2026-07-01）：5 step 冒烟全过，峰值 **~8.8GB/卡**，~120s/step，无 OOM
  - 配置：`qwen_dpo_lora_z3_smoke.yaml` + `ds_z3_lowmem.json` + NCCL 三件套（`P2P_DISABLE/IB_DISABLE/SHM_DISABLE=1`）
  - 必修补丁：`dpo/trainer.py` 去掉 forward 的 labels（CE 峰值）；LoRA+torch2.11 scheduler callback（`zip()` 报错，非 OOM）
  - 日志：`logs/dpo_lora_z3_smoke_8gpu_v5.log`；权重：`saves/qwen2.5-7b-instruct/lora/dpo_z3_smoke/`
- ✅ **SimPO 全量+ZeRO-3（8 卡）**：也可跑，但比 DPO+LoRA 更吃显存/更慢；`cutoff_len=1536` 曾稳定到 step 20+

#### 🚨 系统内存红线（2026-06-25 新增，强制）
- 任何高内存任务（尤其 `build_extend_index.py`）运行前必须先评估 OOM 风险。
- `build_extend_index.py` 已增加 **CUDA OOM 自适应降批**（`START_BATCH_SIZE=64`，失败自动减半到 `MIN_BATCH_SIZE=8`）与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，用于避免 24GB 卡在首批编码时 OOM。
- `build_extend_index.py` 已新增 `--device` 参数（如 `--device cuda:0`）并在启动时 `torch.cuda.set_device` 显式绑定卡，避免默认设备漂移导致加载阶段 OOM。
- `build_extend_index.py` 的合并阶段已修复：不再使用仅适用于 IVF 的 `faiss.merge_into`，改为 `IndexFlatIP` 兼容的 `reconstruct_n + add`；支持 `--merge_only` 直接复用已生成的 `index_part_*.index` 产出最终索引。

#### 🧠 方向 & 贡献点（讨论）
- [ ] **【P0，讨论】** 精读 ReasonRAG 方法节，搞清楚 MCTS 数据构造 + DPO 训练的细节，找改进白区
- [ ] **【P0，讨论】** 确定具体贡献点，起草一句话 pitch
- [ ] **【P1，问导师】** 导师是否接受"不做 online RL"方案；GPT-4o 标注费用是否有支持
- [ ] **【之后】** 根据确认的贡献点起草开题骨架

> ⚠️ **已确认**：毕设目标是**在 ReasonRAG 基础上改进**（情况二），需自己复现训练流程，不是只跑 inference 取数字。

---

## 🧪 2026-06-24 复现偏差排查（HotpotQA）

**最新观测**：
- `hotpotqa` 评测结果：`EM 0.3321 / F1 0.4396`（显著低于论文 Table 2 的 `38.4 / 48.9`）。
- 运行配置来自：`output/hotpotqa_2026_06_24_05_21_/.../config.yaml`。

**已确认的关键偏差（有证据）**：
1. **语料与论文不一致**：论文 Appendix E.1 明确写了“在 wiki18 上并入 PopQA/HotpotQA/2Wiki 相关内容做增强语料”；当前推理仍使用 `indexes/wiki18_100w.jsonl` + `indexes/bge_Flat.index`。
   - 2026-06-24 定量检验：`context` 标题覆盖率仅 **67.47%**（81,095/120,195），`context` 100-word chunk 精确覆盖率仅 **0.035%**（58/165,738）。
2. **增强语料当前无效**：`wiki18_100w_extend.jsonl` 与 `wiki18_100w.jsonl` 行数完全相同（`21,015,324`），且 `bge_Flat_wiki_extend.index` 不存在，说明未形成可用增强检索链路。
3. **当前热点评测不是论文主设定**：最新一次是 `retrieval_topk=5`（论文主文/附录主设定是 top-3）；这会带来可比性偏差。
4. **checkpoint 一致性待确认**：当前本地模型卡显示 `dpo_v16 / dpo_mcts_rag_v8` 自动导出信息，需核验是否与论文发布的最终 `ReasonRAG` checkpoint 完全一致。

**下一步（按优先级）**：
- [ ] 先做**严格可比复现**：固定论文设定（top-3、同 split、同 backbone），并记录输出 `config.yaml` 作为证据。
- [ ] 修复/重建增强语料流程：重新生成 `wiki18_100w_extend.jsonl`（保证行数增加），并构建对应 `bge_Flat_wiki_extend.index`。
- [ ] 校验 checkpoint 来源：确认是否为论文 release 权重，必要时拉取官方权重重跑一轮 HotpotQA。

---

### 📂 已下载论文（`papers/`）

| 文件名 | 用途 |
|---|---|
| `2505.14069v3.pdf`（ReasonRAG） | 参考 baseline，代码有但 GPT-4o 依赖重 |
| `2601.21912v1.pdf`（ProRAG） | **主底座**，精读方法节 |
| `2602.22576v1.pdf`（Search-P1） | 参考，了解 outcome-only RL 上限 |
| `BASELINE_IG-Search_arXiv2604.15148.pdf` | 已排除（无代码），保留读方法思路用 |
| `BASELINE_TreePS-RAG_arXiv2601.06922.pdf` | 已排除（无代码），保留读方法思路用 |
| `BASELINE_PRIME_implicit_PRM_arXiv2502.01456.pdf` | 参考隐式 PRM 方法 |
| `BASELINE_StepPO_2026_arXiv2604.18401.pdf` | 参考 step-level PO |
| `papers/tool-use/`（若干） | 兜底保留 |

---

## 🚫 不要做的事

- ❌ 不要推荐 RE / EE / NER 方向（永久放弃）
- ❌ 不要以 ProRAG 作为"可训练基线"（online RL 在 8×3090 不可行）
- ❌ 不要推荐需要 online RL rollout 的方案（GRPO/PPO + agentic rollout 均排除）
- ❌ 不要推荐没有开源代码的论文作为底座候选
- ❌ 不要在 Windows 笔记本上跑 GPU 任务——没卡
- ❌ 不要用 OpenAlex 裸 `search` 查询出趋势数字——口径污染严重

---

## 🗄️ 历史阶段（已过期）

### 2026-06-10 中间阶段：tool-use / 函数调用可靠性
> 被 2026-06-15 ProRAG 方向取代。相关 PDF 保留在 `papers/tool-use/`。

### 2026-06-03 前：DocRE × TTM-RE
> 导师 2026-06-10 建议换方向后降级为兜底。

---

## 2026-07-07 更新：基于新训练对比重置下一步计划

### 当前状态
- 已完成 `DPO 全量微调` 与 `DPO + LoRA` 的对比。
- 观测结果：`DPO + LoRA` 约低 `3` 个准确率点。
- 结论：暂停 RL 路线，论文主线调整为非 RL Agentic RAG。

### 新的待办（按优先级）
- [P0] 写出非 RL Agentic RAG 的 1 页方法规格：
  - 过程奖励构造拆解
  - 不涉及 RL 优化的 rollout/search 改造
  - 训练路径限定为 `LoRA-DPO` 或 `全量 SimPO`
- [P0] 冻结基线短名单（2026 优先，必须开源代码，优先有权重），并确定 1-2 个可改造底座。
- [P1] 设计最小实验矩阵（小/中/全量）以适配当前算力预算。
- [P1] 更新开题第一节叙事，明确说明“排除 RL”的依据（算力成本与迭代效率）。

---

## 2026-07-09 更新：LogicRAG 读后补充

### 新增判断
- LogicRAG 可作为“非 RL Agentic RAG / 动态推理结构检索”的重要参考，但它本身是纯 inference-time workflow，没有训练和模型结构改动。
- 如果后续继续做类似方向，不能只做 DAG 分解、拓扑排序、rolling memory 这类流程复刻；需要补上可训练组件、优化目标或严格诊断评测，否则作为硕士毕设一章偏软。

### 待办补充
- [P0] 在“非 RL Agentic RAG 的 1 页方法规格”里加入一节：**与 LogicRAG 的区别**，明确我们的硬贡献不是 prompt 编排，而是训练/评价/偏好数据中的哪一环。
- [P0] 设计一个候选方案：`LogicRAG-style DAG planner + process supervision verifier/ranker`，列出哪些步骤可训练、需要什么标签、最小实验怎么做。
- [P1] 把 LogicRAG 加入 baseline/related work 表：作为“无训练、动态结构化检索”的代表，用来对比训练增强方案是否有额外收益。
