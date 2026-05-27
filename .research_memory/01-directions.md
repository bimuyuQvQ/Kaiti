# 候选研究方向

> 何时更新：提出新方向、推翻或确认旧方向时

---

## 📐 2026-05-27 新增：两章关联方案（导师 review 触发）

> **背景**：导师要求论文分两章且两章最好有关联。2026-05-27 调研确认 DocRE 在 2025 顶会仍活跃（13 篇/年），但 Y2/Y3（LLM verifier/reranker）被 DRELL (NAACL 2025) 抢了。

### 第一章（确定）

- **基线**：TTM-RE 不动
- **改进点**：在 Y1（retriever 替换）/ Y4（InfoNCE 正则）/ Y5（retrieval long-tail）中选

### 第二章关联方案（待用户选）

| 方案 | Ch1 | Ch2 | 关联强度 | 工作量 | 主要风险 |
|---|---|---|---|---|---|
| **A'** | TTM-RE + Y4（InfoNCE on memory） | TTM-RE + Y1（retriever 替换 static memory） | 强（同底座两改进，都改 memory 模块） | 中 | 低 |
| **B'** | TTM-RE + Y4 在 full-shot 上 | TTM-RE + Y5（retrieval-based long-tail） | 中（同方法两设定） | 中-高 | AMTL 已做 loss 角度长尾，需找它没解决的角度 |
| **C'** | TTM-RE 在 DocRE 上 | 扩展到 Joint Entity-Relation (基于 Re2-DocRED 数据) | 中（任务递进，DocRE → JERE） | **高** | 要增加 NER + coref 任务，工作量 +50% |
| **D'** | TTM-RE（encoder 路线）+ 改进 | 用 TTM-RE 输出做 LLM 协作（必须**避开 DRELL** 的 probability fusion） | 强 | 中 | DRELL 把最直白的 LLM 协作抢了，剩余空间窄 |

### 推荐排序（基于 2026-05-27 调研，含 arxiv 数据修正）

1. **A' > C' > B' > D'**（B' 排序下降）
2. **A' 推荐理由**：
   - 完全避开 2025 已发表工作（DRELL/GREP/AMTL/ET-MIER 都不撞）
   - 两章都在 TTM-RE 同一个 memory 模块上改，关联最自然
   - Y1/Y4 都在 TTM-RE 论文自己的 limitation 里被点名为 future work
   - 复用 B 会 InfoNCE 经验（用户肌肉记忆）
3. **B' 排序下降的理由（2026-05-27 修正）**：
   - arxiv 数据显示 2025-2026 DocRE 40% 在做 long-tail/few-shot/data augmentation
   - AMTL（loss）+ DOREMI（active annotation）+ VaeDiff-DocRE（data aug）+ GLiDRE（few-shot）已经把长尾的几个主要切入点覆盖
   - 我们做 B' 必须找这 4 个工作都没覆盖的子角度（候选：retrieval-based long-tail），空间被压缩了
4. **C' 可选 + 排序上升**：
   - EACL 2026 Re2-DocRED 是顶会主会长文，提供 +27% triplets 的增强数据集
   - 联合抽取 2025-2026 工作充足（Three-stage / MTEI / Anaphor-Aware / Bi-encoder / Karalka）
   - 工作量大但叙事最完整（"我做了 DocRE，还顺手做了 NER+coref"）

---

## 🔒 已锁定方向（2026-05-11）

**毕业论文方向 = DocRE（文档级关系抽取）× TTM-RE 底座 × LLM 外挂 × RAG 标题包装**

### 最终路线图

```
方向：       文档级关系抽取（DocRE）——不是 EAE、不是纯 ICL、不是 RAG 学术
底座：       TTM-RE（ACL 2024 长文，chufangao/TTM-RE，作者发布了预训练权重）
benchmark：  DocRED + Re-DocRED（标准公开数据集）
Backbone：   RoBERTa-large（底座默认）+ LLM 外挂（7B 级，可用 Llama-3 / Mistral）
形态：       encoder 底座保证 SOTA 性能 + LLM 做 verifier/reranker/ICL 外挂
包装：       论文标题往 "Retrieval-Augmented Document-Level Relation Extraction" 靠
叙事：       师姐做句子级 RE → 用户升级到文档级 RE + 加 LLM，合法且干净的差异化
```

### 为什么是这个组合

| 维度 | 判断 | 证据 |
|---|---|---|
| 方向不死 | DocRE 顶会 28 篇/年（2025 ACL+EMNLP+NAACL title 含 "document-level"）| `03-trends.md` 趋势 9 |
| 栈完全对口 | RoBERTa-large + 检索器 + ICL 全在用户肌肉记忆里 | `00-context.md` B 会技术栈 |
| 底座最好复现 | TTM-RE **唯一发布预训练权重**，load 即复现 | candidate list v1 评估 |
| 答辩好讲 | "我用了 LLM"+"我做了 RAG" 两个关键词都能自然命中 | 包装策略 |
| 工作量可控 | 不需要 from scratch 训 LLM，改进点可选范围大 | 8×3090 充足 |
| 不撞师姐 | 师姐做句子级，用户升文档级，数据集/架构/任务都不同 | 差异化叙事 |

### 候选改进点 Y（2026-05-27 重大修正——Y2/Y3 已被抢）

> **重要变更**：2026-05-27 调研发现 DRELL (NAACL 2025 Long) 已经做了"LLM as refiner with task distribution + probability fusion"，且明确比已有 LLM 方法 +25.2% F1。**Y2/Y3 直接做已无新意，必须放弃**。

**当前可做的 Y（重新排序）**：

| Y | 描述 | 状态 | 详细 |
|---|---|---|---|
| ~~Y2~~ | LLM verifier | ❌ 放弃 | DRELL 已做 |
| ~~Y3~~ | LLM reranker | ❌ 放弃 | DRELL probability fusion 覆盖 |
| **Y1** | 替换 retriever：把 TTM-RE 静态 memory 替换为 retrieved-doc memory | ✅ 仍可做 | 没人做过 retrieval-augmented TTM-RE |
| **Y4** | 对比学习正则：在 TTM memory tokens 上加 InfoNCE | ✅ 仍可做 | 复用 B 会经验，没人做过 |
| **Y5（新）** | retrieval-based long-tail：用 retrieval 帮长尾样本 | ✅ 新机会 | AMTL 做 loss 角度，retrieval 角度空白 |
| **Y6（新）** | 扩展到 JERE：从 RE 扩到 NER + coref + RE | ✅ 新机会 | Re2-DocRED (EACL 2026) 提供新数据 |

**风险类比**：TTM-RE 论文 ablation 显示 DeBERTaV3 替 RoBERTa 反掉 4 F1，TTM 模块对 backbone 敏感。Y1/Y4/Y5/Y6 强耦合，继承这个风险（跟 B 会 AIM 失败同类）。但因为 Y2/Y3 这条"外挂"路径被抢了，**只能接受耦合风险**，通过两章互补降低单章失败概率。

**两章选哪个 Y，取决于用户对方案 A'/B'/C'/D' 的选择**——见本文件顶部"两章关联方案"表格。

---

## ❌ 被排除的方向（决策链完整）

### 方向 X1：继续 CLARE / 句子级 RE（路径 A/B/C）
- **排除理由**：CLARE 核心（PCE + 方法设计）是师姐的，不能作为毕业论文核心。AIM 是用户做的但 +0.2 F1、调参调出来的、换模型大概率失效
- **详见** `04-decisions.md`: 2026-05-11 放弃延续 B 会

### 方向 X2：AIM 升级路径（路径 甲）
- **排除理由**：AIM 鲁棒性不足，无法扩展到 RAG/Long-context
- **详见** `04-decisions.md`: 2026-05-11 放弃延续 B 会

### 方向 X3：领域 IE（路径 乙）
- **排除理由**：实验室没有横向项目数据，无法做领域 IE
- **详见** `04-decisions.md`: 2026-05-11 放弃延续 B 会

### 方向 X4：EAE（事件论元抽取）
- **排除理由**：EAE 顶会年产出只有 RE 的 1/5（7 篇 vs 34 篇），可借鉴 baseline/代码少。虽然用户读过 EAE 论文不算零基础，但"工作量可控"角度 RE 更优
- **详见** `04-decisions.md`: 2026-05-11 确认 RE > EAE

### 方向 X5：纯 LLM ICL（脱离 IE 做 demo retrieval 等）
- **排除理由**：ICL 顶会 115 篇/年**比 RE 卷 3 倍**，差异化难；且 B 会 PCE 思路与 EPR/UDR/CEIL 撞车
- **详见** `04-decisions.md`: 2026-05-11 排除纯 ICL

### 方向 X6：RAG 学术路线
- **排除理由**：RAG 顶会 239 篇/年**比 RE 卷 7 倍**，用户字节 RAG 实习虽有工程积淀但不能转化为论文 contribution（只做了文档清洗+chunking，工程标配）
- **保留**：但**包装层仍然蹭 RAG**——论文标题写 "Retrieval-Augmented Document-Level RE"
- **详见** `04-decisions.md`: 2026-05-11 排除 RAG 学术

---

## 🧊 历史候选方向（已冻结，仅供参考）

> 2026-05-11 之前的候选方向列表，现已被上面的锁定方向取代。保留用于追溯决策来源。

### 候选 1：LLM-based IE 的硬问题（推理 / 鲁棒性 / 可控性）
- 子方向：幻觉抑制、关系/事件结构推理、约束生成、自一致性
- 冒头点：LLM 幻觉抑制 × DocRE（2508.14391 代表）
- **当前处理**：被吸收进"锁定方向"的 Y2/Y3（LLM verifier/reranker）

### 候选 2：文档级 / 跨文档 IE
- **当前处理**：就是锁定方向本身（DocRE on DocRED/Re-DocRED）

### 候选 3：复杂事件结构（嵌套 / 多事件 / 论元共享）
- **当前处理**：已排除（归为 X4 EAE 路线）

### 候选 4：Mamba/SSM 在 DocRE（MAUM 路线）
- **当前处理**：被排除（MAUM 没公开权重，复现风险高；TTM-RE 更稳）

### 候选 5：LLM 幻觉抑制 × RE（2508.14391）
- **当前处理**：作为 Y2 的 motivation 引用，不单独做
