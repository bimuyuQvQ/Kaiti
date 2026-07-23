# 知识库局部检索景观实验

本目录实现“面向局部检索景观的知识库感知查询策略学习”的第一阶段实验。代码独立于原论文的 GRPO 主干，先回答一个更基础的问题：

> 在固定检索器时，无标签的局部检索景观能否预测哪一种查询操作会提高检索质量？

第一阶段不训练语言模型，也不使用强化学习。它以 BEIR 格式语料为输入，为每个查询评估一组确定性查询操作，提取局部景观特征，并执行留一知识库评测。只有诊断结果通过预设门槛后，才进入 LoRA-SFT。

## 输入格式

每个知识库目录包含：

```text
dataset/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    └── test.tsv
```

字段兼容常见 BEIR 命名：

- `corpus.jsonl`：`_id`、`title`、`text`
- `queries.jsonl`：`_id`、`text`
- `qrels/test.tsv`：`query-id`、`corpus-id`、`score`

## 运行

可以先把服务器已有的 ReasonRAG HotpotQA dev 转成真实数据冒烟集：

```bash
PYTHONPATH=experiments/kb_landscape/src \
python -m kb_landscape.prepare_hotpotqa \
  --input baselines/ReasonRAG-main/dataset/hotpotqa/dev.jsonl \
  --output-dir experiments/kb_landscape/output/hotpotqa_smoke_data \
  --max-queries 100
```

随后运行诊断：

```bash
cd /data1/home/lmy/Kaiti
PYTHONPATH=experiments/kb_landscape/src \
python -m kb_landscape.run_diagnostic \
  --dataset /path/to/beir_dataset \
  --corpus-name cloud \
  --output-dir experiments/kb_landscape/output/cloud \
  --max-queries 200
```

对多个知识库完成诊断后：

```bash
PYTHONPATH=experiments/kb_landscape/src \
python -m kb_landscape.analyze_diagnostic \
  --inputs \
    experiments/kb_landscape/output/clapnq/diagnostic.csv \
    experiments/kb_landscape/output/cloud/diagnostic.csv \
    experiments/kb_landscape/output/fiqa/diagnostic.csv \
    experiments/kb_landscape/output/govt/diagnostic.csv \
  --output-dir experiments/kb_landscape/output/analysis
```

## 当前查询操作

- `keep`：保留原查询。
- `keywords`：保留 IDF 最高的原查询词项。
- `prf_expand`：使用初检 top-k 文档中的高判别词扩展查询。
- `prf_reduce`：只保留在初检文档中得到支持的原查询词项。

这些操作是低成本诊断探针，不等同于最终方法。后续可以通过 `--external-candidates` 注入由指令模型生成的 HyDE、问题分解和陈述式查询。

## 输出

- `diagnostic.csv`：逐查询特征、各操作得分、oracle 操作和收益。
- `summary.json`：语料统计、操作分布、平均指标和有害改写率。
- `analysis.csv`：留一知识库评测中各策略的逐库结果。
- `analysis_summary.json`：总体结果和配对 bootstrap 置信区间。

运行产物放在 `output/`，不会提交到 Git。
