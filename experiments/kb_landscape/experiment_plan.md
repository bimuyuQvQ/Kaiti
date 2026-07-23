# 代码实验方案：查询条件化的局部检索景观诊断

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run
- Origin Date: 2026-07-23
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 一、实验目标

验证在固定检索器时，知识库中与当前查询相关的局部分布是否能够预测不同查询操作的检索收益，并检验这种预测能否迁移到训练阶段未见的知识库。

### 研究问题

- **RQ1：操作异质性。** `KEEP`、关键词化、PRF 扩展、后续的 HyDE/分解等操作，是否在不同查询和知识库上产生显著不同的最优选择？
- **RQ2：增量信息。** 局部景观特征能否比固定操作、全局最优操作和 query-only 模型更准确地选择查询操作？
- **RQ3：跨库泛化。** 在留一知识库评测中，局部景观条件是否仍能降低相对 oracle 的遗憾和有害改写率？
- **RQ4：机制。** 哪些局部属性与扩展有益或有害相关，例如分数间隔、局部密度、词项覆盖和排名稳定性？

### 假设

- H1：至少 20% 的查询中，非 `KEEP` 操作相对原查询的 nDCG@10 绝对变化不小于 0.05。
- H2：不同知识库的 oracle 操作分布存在差异，但知识库内部也存在明显异质性，因此固定 corpus-ID 不足。
- H3：留一知识库时，landscape-only 策略相对训练集全局最佳操作，平均 nDCG@10 至少提高 0.01，且配对 bootstrap 95% 置信区间下界大于 0。
- H4：landscape-only 相对 query-only 至少降低 10% 的 oracle regret，或者显著降低有害改写率。

H3、H4 是进入 LoRA-SFT 的主要门槛；H1、H2 只说明现象存在。

## 二、变量

### 自变量

- 知识库：MTRAG 的 ClapNQ、Cloud、FiQA、Govt；后续增加 LoTTE。
- 检索器：第一阶段固定 BM25；通过后增加 Contriever 或其他开源稠密检索器。
- 查询操作：KEEP、KEYWORDS、PRF_EXPAND、PRF_REDUCE；第二阶段增加 LLM 生成操作。
- 策略条件：无条件、query-only、corpus-ID、局部景观、query+局部景观。

### 因变量

- 主指标：nDCG@10。
- 辅助指标：Recall@10、MRR@10、oracle regret、有害改写率、oracle 操作选择准确率。
- 成本指标：每个查询的检索次数和墙钟时间。

### 控制变量

- 固定分词器、BM25 参数、top-k、qrels、候选操作实现和随机种子。
- 同一知识库上的所有方法共享完全相同的索引和候选查询。
- 留一知识库时，特征标准化和模型拟合仅使用训练知识库。

### 主要混杂

- MTRAG 中问题领域与知识库同时变化，不能仅凭跨库差异声称语料因果效应。
- PRF 特征与 BM25 机制强相关，可能不能迁移到稠密检索器。
- 多个操作得分并列会使“最佳操作分类准确率”失真，因此以 selected nDCG 和 regret 为主。

## 三、阶段设计

### 阶段 A：链路与异质性诊断

1. 在每个知识库抽取最多 200 个有 qrels 的测试查询。
2. 固定 BM25，执行四种查询操作并计算逐查询指标。
3. 报告各操作均值、非 KEEP 胜出率、显著增益/退化率、oracle 增益和并列率。
4. 先以 10 个查询冒烟，再扩展到全部可用查询。

通过条件：

- 四库运行完整，无缺失 query/qrel/doc；
- 至少一个非 KEEP 操作在不少于 10% 查询上带来 nDCG@10 ≥ 0.05；
- oracle 相对 KEEP 的平均增益不完全由极少数异常样本贡献。

### 阶段 B：留一知识库预测

每次保留一个知识库作为测试集，其他知识库训练：

1. `KEEP`：永远不改写。
2. `GLOBAL_BEST`：选择训练知识库平均得分最高的固定操作。
3. `QUERY_ONLY`：仅用查询文本预测各操作收益。
4. `LANDSCAPE_ONLY`：仅用初检局部景观预测各操作收益。
5. `QUERY_LANDSCAPE`：联合查询文本和局部景观。
6. `ORACLE`：使用 qrels 选择最高分操作，仅作上界。

使用按查询配对的 bootstrap 置信区间，比较 LANDSCAPE_ONLY 与 GLOBAL_BEST、QUERY_ONLY 的 nDCG 差异。四个留一库结果分别报告，不能只报微平均。

### 阶段 C：LoRA-SFT

仅在阶段 B 达标后执行。训练数据由离线 oracle 构造：

```text
问题 + 检索器标识 + 局部景观摘要 -> 查询操作 + 改写查询
```

- 模型：优先 1.5B–3B 指令模型。
- 训练：LoRA-SFT；LoRA-DPO 仅作可选补充。
- 标签：对小分差样本使用软标签、候选排序或过滤，避免强行学习近似并列的操作。
- 对比：ACL 2026 retriever-aware、query-only SFT、corpus-ID SFT、固定操作、RAMP-like profile。

## 四、数据集安排

| 角色 | 数据集 | 原因 |
|---|---|---|
| 主实验 | MTRAG 四库 | 格式统一、近期、规模可控 |
| 自然控制 | LoTTE 五类 | 同一 StackExchange 来源，弱化平台和文体差异 |
| 外部验证 | BEIR 5–7 子集 | 检验跨任务鲁棒性和有害改写 |
| 压力测试 | BRIGHT 2–3 子集 | 检验推理扩展在代码/数学等环境中的负收益 |
| 负对照 | KILT 多任务 | 固定 Wikipedia，帮助区分任务效应和知识库效应 |

## 五、统计分析

- 主比较以逐查询 nDCG@10 差异为单位，执行 10,000 次配对 bootstrap。
- 同时报告均值差、95% CI、胜/平/负比例，不以单一 p 值作判断。
- 多库比较分别报告各库结果，并使用宏平均；不让大知识库支配结论。
- 对 oracle 操作分布使用 Cramér's V 描述知识库与操作的关联强度。
- 使用 permutation importance 分析景观特征，但只作关联解释，不作因果表述。

## 六、否证与停止条件

- 非 KEEP 的显著胜出率低于 10%，说明动作空间缺乏可利用异质性。
- LANDSCAPE_ONLY 不优于 GLOBAL_BEST，说明景观特征没有决策价值。
- QUERY_LANDSCAPE 不优于 QUERY_ONLY，说明知识库状态未提供增量信息。
- 收益只出现在 PRF 这类与 BM25 同构的操作，换成稠密检索器后消失。
- 多次探测的检索成本明显高于收益，可将方法降级为单次初检特征。

触发停止条件时，不继续训练大模型；章节改为局部检索景观与查询增强收益的实证分析。

## 七、执行配置

- Language/Framework: Python 3.10，NumPy、SciPy、pandas、scikit-learn。
- Working Directory: `/data1/home/lmy/Kaiti`
- 第一条命令：

```bash
PYTHONPATH=experiments/kb_landscape/src \
/data1/home/lmy/miniconda3/envs/retriever/bin/python \
-m unittest discover -s experiments/kb_landscape/tests -v
```

- 冒烟实验超时：30 分钟。
- 正式四库 BM25 诊断超时：每库 2 小时。
- 输出目录：`experiments/kb_landscape/output/`。
