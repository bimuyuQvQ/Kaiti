# ETC/CURA 研究层

本目录只承载新增研究代码，不修改或替代上一级 ETC baseline。当前 P0 提供：

- 版本化答案抽取规则；
- 稳定的数据划分、ID 和运行清单；
- 问题、ETC-QFS、prefix-gap 三类查询候选的数据契约；
- 保留文档 ID、标题、BM25 分数和排名的检索适配器；
- skip/retrieve 配对轨迹的反事实收益计算与完整性审计。

MVP 配置见 `configs/hotpotqa_cura_mvp.json`。该配置当前是实验契约，不会直接启动模型或完整评测。

CPU 验证命令：

```powershell
python -m unittest discover -s baselines/ETC/research/tests -v
```

