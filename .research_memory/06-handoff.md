# 🚪 Agent 交接文件（必读，2026-05-11）

> **如果你是第一次进入这个项目的 AI agent，从这里开始读。**
> **本文件目标**：让你在 5 分钟内拿到完整上下文，直接进入工作状态。

---

## 1. 这个项目是干什么的

帮一位**硕士生准备开题 + 写毕业论文**。
最终输出是一份能让用户**毕业**的硕士论文（不是顶会发表）。

## 2. 用户身份与目标（一句话版本）

- 北京理工大学硕士，已发 1 篇 B 会（"水 B 会"，他自己评价）
- 毕业要求已满足，**不再投会议**
- 就业方向：**央国企**，不读博、不去大厂、不搞学术
- 核心目标：**能毕业，工作量可控**

## 3. 已锁定的毕业论文方向（不要再讨论是否换）

```
方向：    DocRE（文档级关系抽取）
底座：    TTM-RE（chufangao/TTM-RE，ACL 2024 长文，作者发布预训练权重）
benchmark: DocRED + Re-DocRED
backbone: RoBERTa-large + LLM 7B 外挂
形态：    encoder 底座 + LLM verifier/reranker 外挂
包装：    论文标题往 "Retrieval-Augmented Document-Level Relation Extraction" 靠
```

详细决策链见 [`04-decisions.md`](./04-decisions.md)，方向对比见 [`01-directions.md`](./01-directions.md)。

## 4. 关键约束（绝对不要碰的红线）

| 红线 | 原因 |
|---|---|
| ❌ 不要延续 B 会的 CLARE / PCE / AIM 思路 | PCE 是师姐做的（不是用户的工作），AIM 鲁棒性不足（+0.2 F1，调参产物） |
| ❌ 不要换方向到 EAE / 纯 ICL / RAG 学术 | 已排除，理由都在 `04-decisions.md` |
| ❌ 不要在 Windows 笔记本上跑 GPU 任务 | 没卡，所有 GPU 任务在实验室 Linux 服务器（8×3090，SSH 连接） |
| ❌ 不要用 OpenAlex 裸 `search` 查询给"趋势"结论 | 跨学科污染严重，见 `03-trends.md` 顶部口径警告 |
| ❌ 不要写代码占位符（如 `YOUR_API_KEY`、`<base_url>`） | 用户规则，所有需要环境值的地方主动用 `AskUserQuestion` 问 |

## 5. 当前工作机器

| 角色 | 机器 | 用途 |
|---|---|---|
| 调研/写作/讨论 | 个人 Windows 笔记本 | AI 对话、记忆库更新、git push |
| 训练/推理/GPU | 实验室 Linux 服务器（8×3090，SSH） | clone TTM-RE 代码，跑训练，跑推理 |
| 同步媒介 | git@github.com:bimuyuQvQ/Kaiti.git | 两个环境共享记忆库 + 代码 |

> 历史环境（已废弃）：字节 macOS 机器（用户即将离职，不再用）

## 6. 立即可执行的第一个任务

**如果你在 Windows 笔记本上**：用户在跟你讨论方向、记忆库、调研——查 [`05-next-steps.md`](./05-next-steps.md) 看 P1 / P2 待办。

**如果你在实验室 Linux 服务器上（SSH agent）**：
1. 读完本文件 + [`00-context.md`](./00-context.md) + [`05-next-steps.md`](./05-next-steps.md) P0 段
2. 直接开始执行 P0 的 6 步 checklist（clone TTM-RE → 配 env → 下数据 → 下权重 → 跑 inference → 跑 train）
3. 完成每步后 `git commit` + `git push` 更新记忆库进度

## 7. 阅读顺序（5 分钟入门）

1. 本文件（你正在读）
2. [`00-context.md`](./00-context.md) —— 用户背景、目标、约束、B 会论文细节
3. [`04-decisions.md`](./04-decisions.md) —— 决策日志（最重要的是 2026-05-27 两条 + 2026-05-11 那批，按时间倒序读前 7 条即可）
4. [`05-next-steps.md`](./05-next-steps.md) —— 当前在哪、下一步做什么
5. [`01-directions.md`](./01-directions.md) —— 方向锁定理由 + **2026-05-27 两章关联方案 A'/B'/C'/D'**
6. [`02-papers.md`](./02-papers.md) —— TTM-RE 底座详情 + 候选清单 + 2025-2026 顶会论文 + Y 改进点更新
7. [`07-reading-list.md`](./07-reading-list.md) —— **必读论文清单（24 篇）+ PDF 文件名映射 + 跨机复现下载脚本**
8. [`03-trends.md`](./03-trends.md) —— 趋势观察（注意顶部口径警告 + 2026-05-27 arxiv 真实数据）
9. [`notes-ttmre.md`](./notes-ttmre.md) —— TTM-RE 精读笔记（含公式 + 代码对照）

`my_paper/` 目录有用户 B 会论文 PDF + 提取文本，**只用于了解"哪些技术路径不能碰"**，不要试图延续这条线。

`papers/` 目录有 24 篇必读 DocRE 论文 PDF（**被 gitignore 不入库**）。如果当前机器上 `papers/` 为空，按 [`07-reading-list.md`](./07-reading-list.md) 末尾"复现下载"区块的脚本重新下载（PowerShell 或 bash 都给好了）。

## 8. 沟通规则（用户偏好）

- **中文回复**
- 输出说重点，砍掉不改变决策的信息
- 遇到问题追根因，不打补丁
- 动机或目标不清晰时，停下来用 `AskUserQuestion` 讨论
- 高危操作（修改/删除现有数据）必须明确说明计划等待批准
- 长时间命令必须后台跑或明确告诉用户在干什么，**不要默默阻塞对话**

## 9. 记忆库自动更新规则

每轮对话**临结束前**，AI 必须自问以下问题，命中任一就更新对应文件：

1. 用户表达了新的偏好/约束/目标 → 改 `00-context.md`
2. 讨论了新方向 / 推翻了旧判断 → 改 `01-directions.md`
3. 出现新论文 / 论文被下载/读过 / 拿到新引用数 → 改 `02-papers.md`
4. 形成新的横向趋势判断 → 改 `03-trends.md`
5. 做了"选 A 不选 B"、"放弃 X"类决策 → 追加到 `04-decisions.md`
6. 本轮结束有未完成的 next step → 刷新 `05-next-steps.md`

不要等用户提醒。如果本轮纯闲聊或纯解释、没有新结论 → 不更新。

---

**读完本文件 → 读 `00-context.md` → 读 `05-next-steps.md` → 开干。**
