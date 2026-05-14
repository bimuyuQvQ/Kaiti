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

候选清单（详见 `01-directions.md`）：
- **Y1**：替换 retriever（用 B 会 InfoNCE 经验，DocRE 专用检索器）
- **Y2**：加 LLM verifier（蹭 RAG 标题，参考 Correction & Completion）
- **Y3**：加 LLM reranker（参考 LMRC 第二阶段）
- **Y4**：对比学习正则（在 TTM token memory 上加 InfoNCE）

**选择策略**：跑通 P0 之后，看哪个改进点对 TTM-RE 现状的"短板"最直接。**不要在跑通底座之前预先决定 Y**。

---

## 🚫 不要做的事（防止 agent 跑偏）

- ❌ **不要再讨论"要不要换方向"**——方向已锁定
- ❌ **不要尝试延续 CLARE / AIM 思路**——师姐工作 + 鲁棒性差
- ❌ **不要做 EAE / 纯 ICL / RAG 学术**——已排除，理由见 `04-decisions.md`
- ❌ **不要在 Windows 笔记本上跑 GPU 任务**——没卡
- ❌ **不要用 OpenAlex 裸 `search` 查询出趋势数字**——口径污染严重，见 `03-trends.md` 顶部口径警告

---

## 🎯 历史 next steps（保留近 3 次）

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
