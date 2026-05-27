# 决策日志

> 何时更新：每次做"取舍"、"换方案"、"放弃某方向"时追加（按时间倒序）

---

### 2026-05-27（同日 N+1）：arxiv 真实数据修正——DocRE 是"冷门活着"而非"甜区"

- **背景**：用户质疑 N 轮里"DocRE 13 篇/年 vs EAE 7 篇/年"的数据来源，且问有没有 arxiv 调研
- **现场调研**：直接拉 arxiv title 严格匹配的累计 + 2025/2026 分布（见 `03-trends.md` 趋势 10）
- **真实数据**：
  - DocRE arxiv title 累计 88 / 2025 全年 7 / 2026 前 5 月 3 / 2024 16
  - EAE arxiv title 累计 46 / 2025 全年 2 / 2026 前 5 月 1 / 2024 11
  - RAG arxiv title 累计 1,522 / 月产 ~60 → 年 ~700
  - ICL arxiv title 累计 2,067 / 月产 ~70 → 年 ~840
- **修正之前的 framing**：
  - 我之前用 "DocRE 13 vs EAE 7" 论证"DocRE 是甜区"——**口径混乱、不能相比**
  - 真实对比：**DocRE 2025 是 EAE 的 3.5x**（7:2，arxiv 严格 title），但都是个位/十位数
  - **DocRE 比 RAG/ICL 冷 30-40 倍**，是冷门子方向，不是热点
- **对老师"过时"担忧的修正回答**：
  - "过时"严格说不对——DocRE 还在顶会主会出（NAACL/COLING/EMNLP/EACL Main）
  - 但准确说法是：**DocRE 一直是冷门子方向，但持续活着，且在 LLM 时代被重新激活**
  - 不能再用"DocRE 比 EAE 多 2 倍所以活"这种话术——量级一致，差异是定性的不是定量的硬证据
- **新发现的 long-tail/few-shot 方向更拥挤**：
  - 2025-2026 arxiv DocRE 的 40% 在做 long-tail/few-shot/data augmentation
  - DOREMI（2026-01）、VaeDiff-DocRE（2025-01）、GLiDRE（2025-08）、AMTL（ACL Findings 2025）合起来已经覆盖 long-tail 的多个角度
  - **方案 B' 的难度比我之前估计高**——必须找 4 个工作都没覆盖的角度（candidate：retrieval-based long-tail）
- **决定**：
  - 数据修正记入记忆库，未来不再用错误的 13/7 估算
  - DocRE 方向保持锁定（理由仍成立：跟用户技术栈匹配、TTM-RE 复现可控、Y1/Y4/Y5/Y6 都有空间）
  - **方案 B' 优先级下降**（从二选项 → 三选项，因为更拥挤）
  - 用户的"小众"疑虑是合理的，但**不影响硕士毕业**——答辩老师不能拿 RAG 标准来要求
- **理由**：研究记忆库的价值在于可信，错的数字留着会污染未来决策。这次修正确保下次需要数据时用的是 arxiv 严格 title 这条干净口径

---

### 2026-05-27：导师 review 触发的方向重审——确认 DocRE 不过时 + Y2/Y3 必须放弃

- **背景**：用户与导师讨论后反馈两个担忧：
  1. 论文要分两章，两章最好有关联
  2. TTM-RE (2024) 太老了，导师担心是不是因为方向没人做了所以才没新论文
- **调研动作**（2026-05-27）：用 google_scholar 拉了 2025-2026 顶会 DocRE 论文，确认主会 + Findings + 其他主流会议 = ~13 篇/年
- **核心发现**：
  - **方向没死**：NAACL 2025 主会（DRELL）/ EMNLP 2025 主会（SciNLP benchmark）/ COLING 2025 主会（CaDRL）/ EACL 2026 主会（Re2-DocRED 联合抽取）都有 DocRE 工作
  - **东北大学 Fu Zhang 团队 2025 发 5 篇 DocRE**——证明有团队 actively investing
  - **🔴 Y2/Y3（LLM verifier/reranker）必须放弃**：DRELL（NAACL 2025 Long）已经做了"LLM as refiner with task distribution + probability fusion"，且明确比已有 LLM 方法 +25.2% F1。我们再做单纯的"LLM verifier"是炒冷饭
  - **🔴 方案 B'（长尾路线）有冲击**：AMTL（ACL Findings 2025）已经做了 plug-in 的长尾 loss 改进，且在 TTM-RE 等模型上一致提升。我们做 B' 必须找 AMTL 没覆盖的子问题（如 retrieval-based long-tail）
  - **🟢 方案 C（联合抽取）有新窗口**：EACL 2026 Re2-DocRED 提供了增强的 JERE 数据集（+27% triplets），且做的就是 NER + coref + RE 联合
- **决定**：
  - 第一章基线**保留 TTM-RE 不动**（理由：ACL 2024 长文 + 唯一公开权重 + 复现可控）
  - 第一章改进**放弃 Y2/Y3**，转向 Y1（retriever 替换）/ Y4（InfoNCE 正则）/ Y5（retrieval-based long-tail）/ Y6（联合抽取扩展）
  - 第二章方向**等用户在 A'/B'/C'/D' 中选定后再具体定**
- **理由**：
  - 不能在毕业论文里讲跟 DRELL 同样的故事（直接被审稿人拍）
  - "小众 ≠ 过时"——DocRE 13 篇/年 vs RAG 240 篇/年 vs ICL 115 篇/年，是冷门**但对硕士毕业是甜区**
  - 用 EACL/COLING/NAACL/EMNLP 主会硬证据反驳导师"过时"担忧
- **遗留待选**：两章关联方案 A'/B'/C'/D' 待用户决定（具体方案见 `01-directions.md` "2026-05-27 两章关联方案"区块）

---

### 2026-05-15：Y 改进点优先级修正——Y2/Y3（LLM 外挂）优先于 Y1/Y4（动 memory 内部）
- **背景**：精读 TTM-RE 后重新审视 4 个候选 Y。新发现 3 个事实：
  1. TTM-RE 真正的胜负手是"两段式 schedule + SSR-PU loss"，memory 模块只是配菜
  2. Human-only setting 下 TTM-RE 没赢 SSR-PU（79.95 vs 80.18）——TTM 优势只在 H+D 成立
  3. **DeBERTaV3 替 RoBERTa 反而掉 4 F1**（80.56 vs 84.01）——TTM 模块对 backbone 敏感
- **关键风险类比**：事实 3 跟用户 B 会 AIM 失败的根因（"换 backbone 大概率失效"）**是同类问题**
- **解耦判断**：
  - Y1（动 memory 内部，替换 nn.Parameter 为 retrieved tokens）/ Y4（在 memory 上加 InfoNCE）**跟 TTM 主体强耦合**，等于在作者代码上做手术——继承 backbone 敏感性风险
  - Y2（LLM verifier）/ Y3（LLM reranker）**完全外挂**，只读 TTM 输出的 top-K 候选——TTM 主体换什么 backbone 都不影响外挂
- **决定**：
  - **首选 Y2 或 Y3**（外挂式）
  - **Y1/Y4 降级为兜底**（如果 Y2/Y3 跑出来效果不行再回过头做）
- **理由**：用户已经在 B 会被"backbone 敏感"坑过一次（AIM +0.2 F1 且换模型失效），不能再栽同一个坑。外挂式改进的"涨分"独立于底座，是更干净的科研姿态
- **遗留待验证**（P0 复现完立即测）：Y2/Y3 的 LLM 推理成本——DocRED test ~1000 doc × 几十~上百候选三元组 = 万到十万级 LLM 调用，7B 在 3090 上可能要几小时到一天。意味着 ablation 数量受限，要先在 sample 子集上调

---

### 2026-05-11：最终锁定底座 = TTM-RE（ACL 2024 长文）
- **背景**：已排除延续 B 会、EAE、纯 ICL、RAG 学术四条路线。需要在 DocRE 方向下从 8 个候选底座中定 1 个
- **候选池**（见 `02-papers.md` "候选底座清单 v1"）：
  - TTM-RE (ACL 2024 长文) / DEEIA (EAE) / CsEAE (EAE) / AutoRE / LMRC / HD-LoA / Context-Guided LP / KeyEE
- **选择**：**TTM-RE**
- **决定性理由**：
  1. **唯一发布预训练权重**（chufangao/TTM-RE release）—— clone 下来 load 即复现，省 1-2 周从零训
  2. ACL 2024 Long Paper（主会，非 Findings），答辩没有"会议档次"问题
  3. RoBERTa-large + InfoNCE 风格对比学习与用户 B 会技术栈 1:1 对口
  4. 数据集 DocRED / Re-DocRED 是 DocRE 标准 benchmark，不需要 LDC 授权
  5. 核心 TTM 模块（Token Turing Machine 记忆模块）有明确改进空间（可加 InfoNCE 正则、可换 retriever、可挂 LLM verifier）
- **候选改进点 Y（跑通复现后再决定）**：Y1 替换 retriever / Y2 加 LLM verifier / Y3 加 LLM reranker / Y4 对比学习正则
- **包装策略**：论文标题往 "Retrieval-Augmented Document-Level Relation Extraction" 靠，蹭 RAG 热点（免费）

---

### 2026-05-11：确认 RE > EAE 作为毕业论文方向
- **背景**：用户纠正"EAE 不是零基础，之前看过论文"，需要在 RE 和 EAE 之间二选一
- **数据**（`03-trends.md` 趋势 9）：
  - RE 顶会命中 34 篇/年（2025）
  - EAE 顶会命中 7 篇/年（2025）——只有 RE 的 1/5
- **选择**：RE
- **理由**：
  1. 体量差 5 倍 → 可借鉴 baseline、开源代码、benchmark 社区都是 RE 更多
  2. 用户 B 会就是 RE，术语/流程/评测都熟，上手成本最低
  3. 存在强候选底座 TTM-RE（权重已发布），EAE 对应候选 DEEIA 没发权重要从零训
  4. "求毕业"目标下，"工作量可控"是最硬约束
- **EAE 不是被"零基础"排除的**，是被"工作量"排除的。用户的 EAE 阅读背景作为**开题报告相关工作章节**的素材保留

---

### 2026-05-11：排除纯 LLM ICL 路线
- **背景**：用户问"要不要完全脱离 IE 做 ICL 本身的研究（demo retrieval 等）"
- **数据**：ICL 顶会 115 篇/年（2025），**比 RE 卷 3 倍**
- **排除理由**：
  1. 成名工作扎堆（EPR / UDR / DPP / CEIL / Skill-KNN / Se2 / ICCL），差异化难
  2. 用户 B 会的 PCE 思路（LLM 前向打分作 retrieval supervision）**和这些工作严重撞车**
  3. 评审/答辩对"创新度"要求高于 IE 子领域
- **保留**：ICL 作为 Y3 LLM reranker 的技术组件出现，但不作为独立方向

---

### 2026-05-11：排除 RAG 学术路线
- **背景**：用户有字节 RAG 实习经验，考虑是否转 RAG 学术方向
- **关键修正**：用户澄清实习只做了文档清洗 + chunking（工程标配），**无法转化为论文 contribution**
- **数据**：RAG 顶会 239 篇/年（2025），**比 RE 卷 7 倍**
- **排除理由**：
  1. RAG 红海卷度太高，差异化点位难找
  2. 工程经验无法直接变论文卖点
  3. 学术 RAG 需要 1-2 周前期文献补课，延后动手时间
- **保留**：**包装层蹭 RAG 关键词**——论文标题写 "Retrieval-Augmented Document-Level RE"，实际工作核心仍是 DocRE

---

### 2026-05-11：修正 OpenAlex 趋势查询的口径错误
- **背景**：之前为回答"IE/RE/EE 是否过时"，用 OpenAlex 裸 `search` 查询拉过一组数字（IE 253K / RE 69K / EE 93K / ICL+RE 357），并据此写了"RE 略微下滑、EE 持续增长"的结论
- **问题**：用户质疑 EE 一年 93K 不合理（凭直觉应该几百篇），追查发现 OpenAlex `search` 是全文+标题+摘要宽泛匹配，跨学科污染严重——物理学 "excited event extraction"、医学 "adverse event"+"data extraction" 都会命中。这组数字**完全不能反映 NLP 领域内真实论文数**
- **修正**：
  - `03-trends.md` 顶部追加"口径警告"段，标注那组数字不可引用
  - 给出未来正确的趋势验证口径（DBLP > ACL Anthology > SS API + venue filter > OpenAlex + concept_id；裸 search 永远不用）
  - 重新用 ACL Anthology 爬年度 proceedings title，作为干净口径（见 `03-trends.md` 趋势 9）
- **教训**：用 OpenAlex/SS 拉数字之前必须先想清楚"这个 query 命中的是不是我想要的范围"。下次出"趋势"类结论必须标注查询口径，让用户能复核
- **理由**：研究记忆库的价值在于可信，错的数字留着会污染未来决策（比如答辩素材）

---

### 2026-05-11：放弃"延续 B 会"路线，毕业论文必须完全换方向
- **背景**：用户反馈
  - B 会 PCE 是师姐主导，用户没动训练，不能作为毕业论文创新点
  - AIM 是用户独立做的，但提升只有 +0.2 F1，且是调参调出来的，**换模型大概率失效**
  - 实验室没有横向项目数据
- **被排除的路径**：
  - ❌ 路径 A：在 CLARE 上做 V2（句子级 RE）——核心是别人的
  - ❌ 路径 B：CLARE 范式迁文档级——PCE 不是用户的，且文档级 PCE 跑不动
  - ❌ 路径 C：CLARE 迁 EE/EAE——同上
  - ❌ 路径 甲：AIM 升级为核心迁 RAG/Long-context——AIM 鲁棒性不够，撑不起
  - ❌ 路径 乙：领域 IE + AIM——没有领域数据
- **保留候选**：
  - 路径 D：文档级 IE 标准 benchmark（DocRED/Re-DocRED）上做小改进
  - 路径 F：RAG 检索器优化（不依赖 AIM，只用 ICL+检索器经验）
  - 新候选：找个**有公开代码 + 公开标准数据集 + 跑得通**的工作做"小修小补"
- **决定**：下一步先用 OpenAlex/SS 拉 2024-2025 DocRE × LLM 中**带公开代码 + 标准 benchmark**的工作，按"复现难度低 + 工作量可控"排序，给用户挑底座
- **理由**：用户的核心目标是毕业，没有领域数据、没有可信方法基底，**最现实的路是找一篇可复现的工作做小改进**，而不是从概念出发设计新方法

---

### 2026-05-11：根本性目标修正——从"发好会议"切换到"能毕业即可"
- **背景**：之前几轮对话默认用户在追求"方法论新颖 + 上顶会"。本轮用户明确：
  - 毕业要求已满足（1 篇 B 会）
  - 不再投会议
  - 去央国企，不读博、不去大厂
- **影响**：
  - 之前"发顶会难度"的担忧**全部失效**
  - 调研策略从"找新方向、追热点"切换到"找成熟方向、抄作业、控制工作量"
  - "会议层级低"反而变成**优点**——说明这个子领域成熟、可参考工作多、答辩老师不挑刺
  - 4 个候选方向里"Mamba/SSM"、"统一框架"这种工作量大的应淘汰，保留"在 LMRC 范式上做小改进 + 跑实验"这种最稳的
- **决定**：
  - 调研重心转向"找一个有公开代码 + 公开数据集 + baseline 跑得通的工作"作为底座
  - 不再花时间分析"是不是顶会方向"
  - 开题选题以"工作量可控 + 毕业论文能讲完整故事"为唯一标准
- **理由**：用户的真实需求与之前调研路径不匹配，必须立即修正，否则后续努力都浪费在错误目标上

---

### 2026-05-11：建立本地研究记忆库
- **背景**：跨会话失忆，每次都要重新交代背景
- **选项**：
  - A. 让 AI 自己开 md 文件记录
  - B. 全部塞 system prompt
  - C. 用外部 RAG 系统
- **决定**：A（本地 md + 自动更新规则）
- **理由**：维护成本低、可读性强、Git 友好、用户随时可手动改

---

### 2026-05-11：选 A 方案——精读 LMRC 引用网络
- **背景**：9 篇必读论文的引用情况已拉完，LMRC 引用最多（SS=11）
- **选项**：A. 精读 LMRC 11 引；B. 下载 GenExtract 精读；C. 申请 SS API Key
- **决定**：A
- **理由**：LMRC 是 DocRE × LLM 主线起点，看它的引用网络能反推整个子领域 2025-2026 的演进路径
- **结果**：见 `02-papers.md` 的"LMRC 11 引"章节

---

### 2026-05-11：选 B 方案——拉 9 篇必读论文的引用情况
- **背景**：手上有 9 篇必读论文，但不知道哪些被同行真接住了
- **选项**：A. 精读、B. 拉引用排名、C. 拉 v.s. 写预实验
- **决定**：B
- **理由**：用引用数客观排序，排除"看上去重要但其实没人看"的论文
- **结果**：LMRC 11 引最高，ULTRA 4 引次之，2026 新论文都是 0（正常）

---

### 2026-05-11：放弃用"作者单位 985/211"过滤论文
- **背景**：用户怀疑某些工作质量
- **选项**：A. 按单位过滤；B. 按四信号过滤（顶会 + 代码 + benchmark + error analysis）
- **决定**：B
- **理由**：单位过滤是错误的代理变量——很多顶会论文来自二本，很多名校论文质量也很差；学术评判应该看研究本身

---

### 2026-05-11：选 A 方案——下载 4 篇做跨论文语义比对
- **背景**：要在 LegalCore / GLiDRE / CsEAE / RelPrior 之间做对比理解
- **选项**：A. 全下载做语义比对；B. 看 abstract 比对；C. WebSearch
- **决定**：A
- **结果**：3 篇成功，LegalCore 因 PDF 依赖缺失失败；语义索引因 pro 依赖缺失失败 → 改为手动 5 维度比对

---

### 2026-05-11：确认 RE/EE 不死，问题驱动方向仍可做
- **背景**：用户因审稿人评论"很意外还有人做 NER"而焦虑
- **结论**：传统固定 benchmark IE 已死，但 LLM 时代问题驱动 IE 仍活跃
- **理由**：6 个活跃方向（LLM-IE 硬问题、文档级/跨文档、复杂事件、领域 IE、多模态 EE、IE × 下游任务）都有持续产出
