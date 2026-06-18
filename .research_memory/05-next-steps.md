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

- [ ] **【P0，讨论】** 精读 ReasonRAG 方法节，搞清楚其离线流水线细节，找到 ProRAG Stage 1-3 + ReasonRAG 的差异和白区
- [ ] **【P0，讨论】** 确定具体贡献点，起草一句话 pitch
- [ ] **【P1，问导师】** 导师是否接受"不做 online RL"方案；GPT-4o 标注费用是否有支持
- [ ] **【P1，实验室服务器】** 用 ProRAG 发布权重跑原论文 benchmark inference（复现数字，作为上界参考）
- [ ] **【之后】** 根据确认的贡献点起草开题骨架

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
