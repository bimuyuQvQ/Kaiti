# 候选研究方向

> 何时更新：提出新方向、推翻或确认旧方向时

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

### 候选改进点 Y（2026-05-15 优先级修正）

**首选（外挂式，跟 TTM 主体解耦，规避 backbone 敏感性风险）**：

- **Y2：加 LLM verifier** ⭐——TTM-RE 出候选三元组 → LLM 判真伪，参考 "Correction & Completion (ICAACE 2025)"
- **Y3：加 LLM reranker** ⭐——TTM-RE 出 top-K → LLM rerank，参考 LMRC 第二阶段

**兜底（耦合式，动 memory 内部，仅在 Y2/Y3 跑不通时启用）**：

- **Y1：替换 retriever**——用 B 会 InfoNCE 经验，训 DocRE 专用检索器替换 TTM-RE 默认的检索模块
- **Y4：对比学习正则**——在 TTM 的 token memory 上加 InfoNCE 对齐

**优先级理由**：TTM-RE 论文 ablation 显示 DeBERTaV3 替 RoBERTa 反掉 4 F1，TTM 模块对 backbone 敏感。Y1/Y4 强耦合，继承这个风险（跟 B 会 AIM 失败同类）；Y2/Y3 外挂，独立于底座。详见 `04-decisions.md` 2026-05-15 条目。

**具体选 Y2 还是 Y3，要在 P0 复现 + 验证 LLM 推理成本之后再定**。

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
