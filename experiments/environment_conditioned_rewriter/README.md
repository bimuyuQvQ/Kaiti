# 环境感知查询改写实验

本目录实现“先验证可学习上限，再决定是否训练”的分阶段实验。

## 当前阶段

1. `validate_environment.py`：检查 CUDA、BF16、磁盘、模型缓存和分阶段依赖。
2. `prepare_ragbench.py`：把服务器上直接下载的 RAGBench Parquet 转为统一的语料、查询和 qrels。
3. 后续仅在 best-of-8 门槛通过后加入候选生成、评分和 QLoRA-SFT。

## 快速运行

```bash
python -m env_rewriter.validate_environment \
  --output runs/environment.json

python -m env_rewriter.prepare_ragbench \
  --input-root data/raw/ragbench \
  --output-root data/processed/ragbench \
  --max-train-queries 1000
```

从仓库根目录运行时，需要先设置：

```bash
export PYTHONPATH=experiments/environment_conditioned_rewriter/src
```

数据集的大文件只在服务器直接下载，不通过 Git 或 SCP 传输。
