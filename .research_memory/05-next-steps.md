# 当前待办与下次切入点

> 何时更新：每轮对话结束时刷新（保留近 3 次）

---

## 🎯 当前阶段（2026-06-15 锁定）：过程监督 RL for Agentic RAG，底座 = ProRAG

**方向**：过程监督强化学习 for Agentic RAG，搜索 agent 多跳推理。

**策略 A+B**：
- **A（底座）**：**ProRAG**（arXiv:2601.21912，`lilinwz/ProRAG`，HF 权重 `bmbgsj/ProRAG` Qwen3-8B，MIT 协议）
- **B（冷场景）**：中文 / 领域多跳检索 QA（英文已饱和，中文是真空白；Qwen3 原生支持中文，天然对接）

**排除候选（无代码）**：
- IG-Search (2604.15148)：无代码无权重 ❌
- TreePS-RAG (2601.06922)：无代码无权重 ❌

**执行顺序（确认）**：
1. **Phase 0（立即可做）**：下载 ProRAG 权重，在原论文 benchmark 复现数字
2. **Phase 1**：切换到中文多跳 QA 数据，跑 ProRAG baseline，error analysis
3. **Phase 2**：针对弱点小改 + 消融

---

### ✅ TODO（按优先级排列）

- [ ] **【P0，实验室服务器】** 克隆 ProRAG 代码 + 装环境
  ```bash
  git clone https://github.com/lilinwz/ProRAG
  conda create -n prorag python=3.13.11
  # 按 README 装 vllm==0.11.0、flash-attn 等
  ```
- [ ] **【P0，实验室服务器】** 下载 `bmbgsj/ProRAG`（Qwen3-8B）HF 权重
- [ ] **【P0，实验室服务器】** 在原论文 benchmark（HotpotQA / 2WikiMultiHopQA / MuSiQue）跑推理，复现论文 EM 数字（允许 ±1-2 点误差）
- [ ] **【P1，确认数据】** 查中文多跳 QA 数据集可用性（CMuSiQue / CRAG / DRCD / 自建？）
- [ ] **【P1，实验室服务器】** ProRAG 在中文数据上跑 zero-shot baseline，做 error analysis
- [ ] **【之后】** 根据 error analysis 定具体改进点，回来讨论并起草开题骨架

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

- 不要再讨论"要不要换方向"——方向已最终锁定
- 不要推荐没有开源代码的论文作为底座候选
- 不要在 Windows 笔记本上跑 GPU 任务——没卡
- 不要用 OpenAlex 裸 `search` 查询出趋势数字——口径污染严重

---

## 🗄️ 历史阶段（已过期）

### 2026-06-10 中间阶段：tool-use / 函数调用可靠性
> 被 2026-06-15 ProRAG 方向取代。相关 PDF 保留在 `papers/tool-use/`。

### 2026-06-03 前：DocRE × TTM-RE
> 导师 2026-06-10 建议换方向后降级为兜底。
