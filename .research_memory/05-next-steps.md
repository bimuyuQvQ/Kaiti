# 当前待办与下次切入点

> 何时更新：每轮对话结束时刷新（保留近 3 次）

---

## 🎯 当前阶段：底座复现（2026-05-11 锁定）

**毕业论文方向**：DocRE × TTM-RE 底座 × LLM 外挂 × RAG 标题包装
**当前在哪**：方向已锁定，环境即将切换（字节 macOS → Windows 笔记本 + 实验室 Linux 服务器）
**下一动作**：在实验室服务器上 clone TTM-RE 跑通复现

---

## 📋 P0：实验室服务器复现 TTM-RE（最优先）

> **执行环境**：实验室 Linux 服务器（8×3090 Ti，SSH 连接）
> **不要在 Windows 笔记本上做这部分**——Windows 笔记本只用来看记忆库 + 跟 AI 讨论

### Checklist

- [ ] **0. 环境准备**
  - SSH 连上实验室服务器
  - `git clone git@github.com:bimuyuQvQ/Kaiti.git ~/Kaiti && cd ~/Kaiti`
  - 阅读 `.research_memory/06-handoff.md`（agent 入口文件，会指向所有上下文）
  - 阅读 `my_paper/_extracted.txt`（用户 B 会论文文本，了解 PCE/AIM 不可碰的边界）

- [ ] **1. clone TTM-RE 代码**
  - `cd ~ && git clone https://github.com/chufangao/TTM-RE.git`
  - 通读 README.md，找出官方推荐的 conda env / requirements

- [ ] **2. 配 conda env**
  - 用作者推荐的 PyTorch / transformers 版本（不要乱升级）
  - 装 wandb（看训练曲线方便）

- [ ] **3. 下载数据集**
  - DocRED：从 [thunlp/DocRED](https://github.com/thunlp/DocRED) 拉
  - Re-DocRED：从 [tonytan48/Re-DocRED](https://github.com/tonytan48/Re-DocRED) 拉
  - ChemDisGene（可选）：作者 README 应该指明
  - 按 TTM-RE README 摆好数据目录结构

- [ ] **4. 下载预训练权重**
  - 从 TTM-RE GitHub release 拉作者发布的 `.pt` 文件
  - 验证文件 hash（如 README 提供的话）

- [ ] **5. 跑 inference 复现 paper Table**
  - 用 release 权重跑 DocRED test split
  - **目标**：F1 / Ign-F1 数值对得上论文 Table 1（误差 ±0.3 内算 OK）
  - **如果对不上**：先查 transformers 版本、tokenizer 版本、数据集版本

- [ ] **6. 跑 train，估算时长**
  - 用单卡 3090 train 一次完整流程（DocRED 训练集）
  - 记录：单 epoch 时长、显存峰值、收敛 epoch 数
  - **用途**：决定后续做 Y 改进时能跑几轮 ablation

---

## 🔥 P0.5：导师 review 触发——选定两章关联方案（2026-05-27 新增，未完成）

**背景**：导师反馈论文要两章 + 两章要关联，且担心 TTM-RE 太老。调研已确认方向不过时，但 Y2/Y3 被 DRELL 抢了。

### 待完成

- [ ] 用户在 A'/B'/C'/D' 四个方案中选 1 个（见 `01-directions.md` 顶部表格）
- [ ] 选定后，深读对应方案的关键论文（如 A' 选定则深读 GREP + AMTL；C' 选定则深读 Re2-DocRED + Anaphor-Aware）
- [ ] 重新评估第一章具体的 Y（在 Y1/Y4/Y5/Y6 中选 1-2 个）

---

## 📚 P1：精读 TTM-RE 论文，定改进点 Y

- [x] 1. 精读 TTM-RE 论文（arXiv: 2406.05906）✅ 2026-05-14
  - 笔记落在 `02-papers.md` 「精读笔记」区块
  - 关键收获：训练 schedule 是两段式（distant pretrain → human finetune）；TTM 的 write 没启用；memory 是全局静态的
- [ ] 2. 精读 LMRC（2408.13889）
  - 关注两阶段范式（关系分类 → 实体抽取）
  - 看 retriever 怎么和 LLM 配合
- [ ] 3. 精读 Correction & Completion（DOI 10.1109/ICAACE65325.2025.11019681）
  - 这是 Y2 LLM verifier 路线的直接 motivation
- [x] 4. 列出 3-5 个具体可实施的 Y 改进点 ✅ 2026-05-14
  - 见 `02-papers.md` 「4 个候选 Y 的具体落点」表格
  - 初步倾向 Y2，但**最终决定要等 P0 复现跑完**

---

## 🔬 P2：决定毕业论文的"小改进 Y"（跑通底座后再决定）

**优先级（2026-05-15 修正，详见 `04-decisions.md` 同日条目 + `01-directions.md`）**：

首选（外挂式，规避 backbone 敏感性风险）：
- **Y2 ⭐**：加 LLM verifier（蹭 RAG 标题，参考 Correction & Completion）
- **Y3 ⭐**：加 LLM reranker（参考 LMRC 第二阶段）

兜底（耦合式，仅在 Y2/Y3 跑不通时启用）：
- **Y1**：替换 retriever（用 B 会 InfoNCE 经验，DocRE 专用检索器）
- **Y4**：对比学习正则（在 TTM token memory 上加 InfoNCE）

**选择策略**：跑通 P0 之后，先做一个**小成本验证**——在 100 个 doc 上跑一遍 Y2 全流程，估算 LLM 推理时长 + 涨分量级。再决定 Y2 还是 Y3。**不要在跑通底座之前预先决定具体 Y**。

---

## 🚫 不要做的事（防止 agent 跑偏）

- ❌ **不要再讨论"要不要换方向"**——方向已锁定
- ❌ **不要尝试延续 CLARE / AIM 思路**——师姐工作 + 鲁棒性差
- ❌ **不要做 EAE / 纯 ICL / RAG 学术**——已排除，理由见 `04-decisions.md`
- ❌ **不要在 Windows 笔记本上跑 GPU 任务**——没卡
- ❌ **不要用 OpenAlex 裸 `search` 查询出趋势数字**——口径污染严重，见 `03-trends.md` 顶部口径警告

---

## 🎯 历史 next steps（保留近 3 次）

### 2026-05-27（同日 N+2：必读论文清单 + PDF 批量下载完成）
- 基于本日 N+1 调研，列出 24 篇必读论文清单（5 档优先级）✅
- 批量下载 PDF 到 `papers/`（22.6 MB，全部成功）✅
  - arxiv 15 篇 + ACL Anthology 9 篇
  - `papers/` 在 `.gitignore` 中，PDF 不入库，跨机器需重新下载
- 写入 `07-reading-list.md`（含分档、文件名映射、PowerShell + bash 复现下载脚本）✅
- 更新 `README.md` / `06-handoff.md`（增加 07-reading-list 索引）✅
- **下一步**（同上）：用户做方向决策 + 找老师 30 分钟 calibrate 接受度

### 2026-05-27（同日 N+1：arxiv 真实数据修正）
- 用户质疑 N 轮里"DocRE 13 vs EAE 7"的口径，要求看 arxiv 调研
- 拉 arxiv title 严格匹配的累计 + 2025/2026 分布 ✅
- 真实数据：DocRE 2025 全年 7（vs EAE 2，比 3.5x）；vs RAG 700 / ICL 840（差 30-40 倍）✅
- 修正了之前 "DocRE 是甜区"的过度乐观 framing ✅
- 提取 2025-2026 DocRE arxiv 10 篇完整清单 ✅
- 重要发现：2025-2026 DocRE arxiv 40% 在做 long-tail/few-shot/data augmentation，**方案 B' 比预想更拥挤** ✅
- 更新 `03-trends.md`（新增趋势 10）/ `04-decisions.md`（新增 N+1 决策记录）/ `01-directions.md`（B' 排序下降）✅
- **下一步**：用户做方向决策（继续 DocRE / 拓宽到 DocRE+X / 重新考虑跨领域）

### 2026-05-27（本轮：导师 review 触发方向重审，DocRE 不过时 + Y2/Y3 被抢）
- 用户与导师讨论，导师两个担忧：
  1. 两章要有关联
  2. TTM-RE (2024) 太老了，方向是不是没人做
- google_scholar 拉 2025-2026 DocRE 顶会论文，确认 13 篇/年（4 主会 + 5 Findings + 4 其他主流）✅
- 关键发现：DRELL (NAACL 2025) 已做 LLM as refiner，**Y2/Y3 必须放弃** ✅
- 关键发现：AMTL (ACL Findings 2025) 是 plug-in loss，**方案 B' 需要找它没覆盖的角度** ✅
- 关键发现：Re2-DocRED (EACL 2026) 提供 JERE 增强数据集，**方案 C 有新窗口** ✅
- 提出 4 个两章关联方案 A'/B'/C'/D'（见 `01-directions.md` 顶部表格）✅
- 推荐：A' > B' > D' > C'（理由：A' 完全避开 2025 已发表工作，关联自然，复用用户经验）✅
- 更新 `02-papers.md` / `04-decisions.md` / `01-directions.md` ✅
- **下一步**：用户在 A'/B'/C'/D' 中选定后，深读 2-3 篇关键论文 + 重新评估第一章具体的 Y

### 2026-05-14（本轮：TTM-RE 精读完成）
- 提取 PDF 文本 + 通读论文 ✅
- 对照代码（ttm.py / model2.py / train2.py）验证关键细节 ✅
- 发现 3 个论文未提的代码细节：残差混合 0.5/0.5、tail 用 head_extractor 疑似 bug、write 操作代码有但未启用 ✅
- 输出 4 个候选 Y 的代码级落点（Y2 优先）✅
- 写入 02-papers.md 精读笔记区块 ✅
- 下一步：仍是 P0 上实验室服务器复现

### 2026-05-11（本轮 N+3：底座锁定 + 跨机交接）
- 修正 OpenAlex 口径错误 ✅
- 重新用 ACL Anthology 拉干净趋势数字 ✅
- 锁定底座 TTM-RE ✅
- 重写记忆库为跨机交接版本 ✅

### 2026-05-11（本轮 N+2：底座调研）
- 用 OpenAlex 拉 DocRE × LLM × 公开代码候选清单 ✅
- WebSearch 验证 8 个候选的 GitHub 代码状态 ✅

### 2026-05-11（本轮 N+1：B 会论文分析）
- 用 PyMuPDF 提取 B 会论文文本 ✅
- 识别 CLARE = PCE（师姐）+ AIM（用户）的分工 ✅
- 排除"延续 B 会"路线 ✅
