# 2025-2026 方法学趋势观察

> 何时更新：引用网络 / 批量调研后产生新的横向判断时

## ⚠️ 口径警告（2026-05-11 追加）

**以下"趋势判断"基于引用网络 + 顶会 proceedings 观察，是定性的。**

曾经在对话里出现过一组基于 **OpenAlex 裸 `search` 查询**的粗数字（IE 253K / RE 69K / EE 93K / ICL+RE 357 等），那组数字**口径不严、不可引用**，原因：

- `search` 字段是全文+标题+摘要的宽泛匹配，跨学科命中严重（物理学的 "excited event extraction"、医学的 "adverse event"+"data extraction" 都会被算进来）
- 没加 concept filter（应 filter=concepts 锁 NLP 领域）
- 没加 venue filter（应限定 ACL/EMNLP/NAACL/COLING/AAAI/NeurIPS）
- 2025 索引滞后 6-12 个月，年度对比本身失真

**正确的趋势验证口径**（未来要用趋势数字时走这个）：

| 来源 | 查询方式 | 可信度 |
|---|---|---|
| DBLP | 直接看 ACL/EMNLP/NAACL/COLING proceedings 的 title | ⭐⭐⭐⭐⭐ |
| ACL Anthology | `aclanthology.org/?q=...&year=2024` | ⭐⭐⭐⭐ |
| Semantic Scholar | API + venue filter | ⭐⭐⭐⭐ |
| OpenAlex + concept_id | 必须加 concept_id=C205649164(NLP) 等 filter | ⭐⭐⭐ |
| OpenAlex 裸 search | ❌ 不可用，跨学科污染严重 | ⭐ |

**下面的 8 条趋势判断来自引用网络观察（可信），不是来自那组垃圾数字。**

---

## 趋势 1：DocRE 两阶段范式已经站住（2024-2026）

- **代表起点**：LMRC (2408.13889, 2024-08) "LLM 关系分类 → SLM 实体抽取"
- **跟进者多样化**：
  - **同向加强**：Two-Stage Loss + Anaphor (IJCNN 2025) 在两阶段框架内加 loss 改进
  - **对偶范式**：RelPrior (2511.08143) 反过来——"关系作为先验"先注入
  - **结构化拆分**：DTPE (ICASSP 2026) 文档树分解 → LLM 精修
  - **后处理化**：Correction & Completion (ICAACE 2025) LLM 当后处理器
- **共识**：关系层和实体层分开处理是合理的设计

## 趋势 2：架构层正在脱离纯 Transformer

- **Mamba/SSM 进 DocRE**：MAUM (IJCNN 2025) U-Mamba + 记忆增强是早期落地
- **意义**：长文档天然适合 SSM（线性复杂度），是有架构红利的方向
- **空白度高**：目前能找到的相关工作只有 1-2 篇

## 趋势 3：LLM 幻觉成为 RE 新的独立子问题

- **代表**：Hallucination-Resistant RE (2508.14391) "依存感知句子简化 + 两层层次化精修"
- **历史背景**：BERT 时代不存在这个问题，**LLM 时代变成核心问题**
- **空白度高**：开题切入点候选

## 趋势 4：低资源 / 长尾重新被重视

- **代表**：Coarse-to-Fine (ICASSP 2026) 专门做低资源 DocRE
- **背景**：DocRED 长尾问题在 LLM 时代依然没解决，反而因为 LLM 偏向高频关系而恶化

## 趋势 5：多智能体协作 + 强化学习用于结构化 IE

- **代表**：GenExtract (2603.02909, 2026-03) 多智能体框架做零样本 DocEAE
- **典型设计**：生成 Agent + 验证 Agent + 协作机制
- **判断**：在 2026 是新热点，但学术风险是"工程感重，方法论新意有限"

## 趋势 6：采样 + 选择 替代贪心解码（DocIE 专项）

- **代表**：ThinkTwice (2601.18395, 2026-01) "Sampling and Selection for DocIE"
- **核心动机**：贪心 LLM 抽取不稳定，采样多次 + 择优更可靠
- **判断**：方法简单但效果显著，可作为基线插件

## 趋势 7：会议分布在"信号处理 / 神经网络应用"会议

- **现象**：DocRE × LLM 主线 2025-2026 的工作主要发在 **ICASSP 2026 / IJCNN 2025 / ICKG 2025**，而不是 ACL/EMNLP 主会
- **隐含**：**这个方向还没卷出明星论文，开题反而有机会**
- **风险**：发顶会难度依然高（ACL/EMNLP 评审更挑剔）

## 趋势 8：数据集层面，法律领域 + 长文档 + 事件共指是新热点

- **证据**：LegalCore (2025-02) 仅 3 个月就被引 2 次，速度高于 CsEAE
- **机会**：领域 IE 论文相对容易做出新意

## 横向小结：开题切入点排序（按"差异化 × 可执行性"）

1. ⭐ LLM 幻觉抑制 × DocRE（趋势 3）
2. ⭐ Mamba/SSM 在 DocRE 的应用（趋势 2）
3. 关系先验 vs 关系分类的统一框架（趋势 1 的元层）
4. 低资源 / 长尾 DocRE 的两阶段优化（趋势 1 + 趋势 4）

---

## 趋势 9：DocRE 在顶会仍然活跃，RAG 在卷成红海（2026-05-11 ACL Anthology 干净口径）

**口径**：直接爬 ACL Anthology 年度 events 页面（acl-2024 / emnlp-2024 / naacl-2024 等），匹配 paper title 关键词。覆盖 main + findings + workshops。

**ACL+EMNLP+NAACL 三家顶会标题命中数加总**：

| 关键词 | 2023 | 2024 | 2025 | 趋势 |
|---|---|---|---|---|
| Relation Extraction | 57 | 40 | 34 | 📉 -40%（下降但有量） |
| Event Extraction | 15 | 15 | 12 | ➡️ 持平略降 |
| Event Argument | 11 | 12 | 7 | 📉 -36% |
| Document-level | 38 | 20 | 28 | ➡️ 波动 |
| In-Context Learning | 70 | 115 | 115 | ➡️ 2024 翻倍后持平 |
| Retrieval-Augmented | 26 | 93 | 239 | 🚀 **9 倍** |
| "RAG"（标题里直接出现） | 0 | 9 | 97 | 🚀 **从零到爆发** |

**对开题的指导**：

- **DocRE 不死**：document-level 命中 28 篇/年，relation extraction 34 篇/年，意味着这是个"冷门但活着"的子领域 → 答辩老师不会喷"过时"，可借鉴 baseline 也够用
- **RE > EAE 5 倍体量**：选 RE 而不是 EAE 的硬证据（EAE 一年只有 7 篇 event argument，可参考工作太少）
- **ICL 持平不跌**：但 115 篇/年比 RE 卷 3 倍，做"纯 ICL demo retrieval"会撞 EPR/UDR/CEIL 一堆成名工作
- **RAG 是红海**：239 篇/年比 RE 卷 7 倍，"做 RAG 学术"性价比差。**但论文标题蹭 "Retrieval-Augmented" 关键词是免费午餐**——主流叙事在那放着，不蹭白不蹭

> 用此表替代之前那组 OpenAlex 裸 search 的脏数字（IE 253K / RE 69K / EE 93K）。


