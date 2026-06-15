# 当前待办与下次切入点

> 何时更新：每轮对话结束时刷新（保留近 3 次）

---

## 🎯 当前阶段（2026-06-15 最终锁定）：过程监督 RL for Agentic RAG，A+B 策略

> 决策链：DocRE→agent项目→PRM/verifier→tool-use→**过程监督 RL for Agentic RAG（2026-06-15 最终）**。完整记录在 `04-decisions.md`。

**方向**：过程监督强化学习 for Agentic RAG，搜索 agent 多跳推理。接受做 RL。

**策略 A+B**：
- **A（方法底座）**：选 **IG-Search（arXiv:2604.15148）或 TreePS-RAG（arXiv:2601.06922）** 作为便宜过程奖励底座（无 MCTS、无外部 LLM 评委、只用结果标签 / 模型自身信号）。两者都在 8×3090 上可跑。
- **B（冷场景）**：把底座搬到**中文 / 领域多跳检索 QA**（现有工作全在英文 HotpotQA 那套，中文设定是真空白）。

**执行顺序（已确认）**：
1. **Phase 0**：在原论文 benchmark 上复现底座数字（±1-2 点）→ 代码跑通、数字对上
2. **Phase 1**：切到冷场景数据，跑 baseline，做 error analysis，找弱点
3. **Phase 2**：针对弱点小改 + 消融实验

逻辑：先复现才有干净的"我的改动让它从 X 涨到 Y"对照；复现过程同步吃透代码结构。

**领域密度（2026-06-15 查证）**：这条线 12 个月内至少 8 篇，月级迭代，"隐式/便宜过程奖励"已是红海。故不拼新颖度，拼"首次在 X 场景下"。

**已下载的核心论文**（`papers/`）：
- 底座候选：IG-Search(2604.15148)、TreePS-RAG(2601.06922) — **待下载**
- 已有：ReasonRAG(2505.14069)、ProRAG(2601.21912)、Search-P1(2602.22576)
- 兜底保留：PRM综述(2510.08049)、tool-use 系列（`papers/tool-use/`）

### 下一动作（按顺序）

1. **【立即】下载底座候选 PDF**：IG-Search(2604.15148) + TreePS-RAG(2601.06922)，确认 GitHub 开源情况
2. **选定底座**（读两篇方法节 + 查 GitHub 活跃度，选代码更易复现的）
3. **Phase 0**：在实验室服务器上跑通底座，复现原论文数字
4. **确认中文多跳数据集**（MuSiQue-zh / DRCD / CMuSiQue / CRAG 等，需查可用性）
5. **Phase 1**：切场景，baseline error analysis
6. 回来定具体改进点 + 开题骨架

---

## 🗄️ 历史阶段（已过期，保留供参考）

### 2026-06-10 中间阶段：tool-use / 函数调用可靠性

> 已被 2026-06-15 A+B 策略取代。相关 PDF 保留在 `papers/tool-use/`。

核心文件：`xLAM2`、`ToolACE`、`Hammer`、`UniToolCall`、`SimpleTool`（全在 `papers/tool-use/`）

### 2026-06-03 前：DocRE × TTM-RE（已降级为兜底）

> 导师 2026-06-10 建议换 agent 方向后降级。DocRE 全套 PDF 保留。
> P0 TTM-RE 复现 checklist / P1-P2 改进点详细计划已存在本文件历史版本中。

---

## 🚫 不要做的事

- 不要再讨论"要不要换方向"——方向已最终锁定
- 不要在 Windows 笔记本上跑 GPU 任务——没卡
- 不要用 OpenAlex 裸 `search` 查询出趋势数字——口径污染严重，见 `03-trends.md`
- 不要再推新方向候选，除非用户主动提出
