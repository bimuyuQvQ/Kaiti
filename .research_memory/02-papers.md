# 论文池

> 何时更新：新增论文、引用数更新、读完一篇后填观点
>
> 引用数标注：[OA: N] = OpenAlex，[SS: N] = Semantic Scholar

---

## 🥇 底座论文（已锁定，2026-05-11）

### TTM-RE：Memory-Augmented Document-Level Relation Extraction
- **arXiv**：2406.05906（作者：Chufan Gao, Xuan Wang, Jimeng Sun）
- **会议**：ACL 2024 Long Paper（非 Findings，主会）
- **引用**：[SS: 12]
- **代码**：https://github.com/chufangao/TTM-RE ⭐
- **权重**：**已发布预训练权重**（这是选它做底座的关键）
- **数据集**：DocRED / Re-DocRED / ChemDisGene
- **Backbone**：RoBERTa-large
- **核心方法**：Token Turing Machine（TTM）做记忆增强 + PU learning 噪声鲁棒 loss
- **与用户技术栈匹配**：RoBERTa-large + 对比学习 1:1 对口
- **预估显存**：~16G（单卡 3090 足够）
- **与 CLARE 的关系**：CLARE 是句子级 RE + LLM ICL，TTM-RE 是文档级 RE + encoder memory，**任务升级 + 架构换代**，无重叠

**为什么选它做底座**：见 `01-directions.md` "为什么是这个组合" 表。

---

## 🥈 候选底座清单 v1（2026-05-11 评估，备查）

| # | 论文 | 年/会议 | 引用 | 代码 | 数据集 | Backbone | 显存 | 复现风险 | 选择结果 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **TTM-RE** | ACL 2024 长文 | SS:12 | [chufangao/TTM-RE](https://github.com/chufangao/TTM-RE) + 权重 | DocRED/Re-DocRED | RoBERTa-large | ~16G | **极低** | ✅ **已选** |
| 2 | DEEIA | ACL Findings 2024 | — | [LWL-cpu/DEEIA](https://github.com/LWL-cpu/DEEIA) | RAMS/WikiEvents/MLEE/ACE05 | RoBERTa | ~16G | 低 | ❌（EAE 路线被排除） |
| 3 | CsEAE | 2024 / arXiv | OA:1 | [simon-p-j-r/CsEAE](https://github.com/simon-p-j-r/CsEAE) | RAMS/WikiEvents/MLEE | RoBERTa + LLM | ~16G | 中 | ❌（EAE） |
| 4 | AutoRE | ACL 2024 Demo | SS:34 | [THUDM/AutoRE](https://github.com/THUDM/AutoRE) | Re-DocRED | Mistral/Llama-7B + QLoRA | ~24G | 中 | ⚠️（可作 baseline） |
| 5 | LMRC | 2024 / arXiv | SS:11 | [wisper12933/LMRC](https://github.com/wisper12933/LMRC) | DocRED/Re-DocRED | LLM (ICL) | 推理为主 | 中 | ⚠️（作为 Y3 reranker 参考） |
| 6 | HD-LoA | ACL 2024 长文 | SS:11 | [hzzhou01/HD-LoA-Prompting](https://github.com/hzzhou01/HD-LoA-Prompting) | RAMS + DocEE | 纯 prompting | 低 | 低 | ❌（无训练，故事弱） |
| 7 | Context-Guided LP | AAAI 2024 | — | [kracr/document-level-relation-extraction](https://github.com/kracr/document-level-relation-extraction) | DocRED/Re-DocRED | RoBERTa + KG | ~16G | 中 | ⚠️（backup） |
| 8 | KeyEE | BDMA 2024 | SS:11 | [OStars/KeyEE](https://github.com/OStars/KeyEE) | ACE05 / ERE | T5 | ~24G | 高（LDC 数据） | ❌ |

---

## ⭐ 9 篇必读论文（截至 2026-05-11 引用排名）

| # | 简称 | arXiv | 年/会议 | OA | SS | 已下载 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | LMRC | 2408.13889 | 2024 / arXiv | 3 | **11** | ❌ | DocRE 两阶段范式起点，引用网络最活跃 |
| 2 | ULTRA + LEAFER | 2401.13218 | 2024 / ACL Findings | 4 | (429) | ❌ | DocEAE 必引基线，Bloomberg 出品 |
| 3 | LegalCore | 2502.12509 | 2025 / arXiv | 0 | 2 | ❌(失败) | 数据集，3 个月被引 2 次很快 |
| 4 | CsEAE | 2411.05895 | 2024 / arXiv | 1 | (429) | ✅ | DocEAE 小+大模型协作 |
| 5 | KnowRA | 2501.00571 | 2024 / arXiv | 0 | 1 | ❌ | DocRE + 知识增强 |
| 6 | Triggers Needed? | 2411.08708 | 2024 / arXiv | 0 | (429) | ❌ | DocEE 范式反思 |
| 7 | ThinkTwice | 2601.18395 | 2026-01 / arXiv | 0 | (429) | ❌ | 采样+选择，太新无引用很正常 |
| 8 | GenExtract | 2603.02909 | 2026-03 / arXiv | 0 | (429) | ❌ | 多智能体 + 零样本 DocEAE |
| — | ECB++ | — | — | — | — | — | 名字在 arXiv 无对应论文，应是数据集别名 |

## ✅ 用户已下载的论文（在 `/Users/bytedance/projects/kait/.arxiv_papers/`）

| arXiv | 简称 | 是否精读 | 关键观点摘录 |
|---|---|---|---|
| 2411.05895 | CsEAE | ❌ | DocEAE 小+大模型协作 |
| 2508.00757 | GLiDRE | ❌ | GLiNER 风格 bi-encoder 用于 DocRE |
| 2511.08143 | RelPrior | ❌ | "关系作为先验"——LMRC 两阶段范式的对偶 |

## 🔥 LMRC 的 11 篇引用论文（按价值分档，2026-05-11 拉取）

### 第一档：直接接续 DocRE × LLM 主线（7 篇，开题必看）

| 论文 | arXiv/DOI | 时间/会议 | 跟进点 |
|---|---|---|---|
| RelPrior | 2511.08143 | 2025-11 / arXiv | 关系作为先验注入，对偶 LMRC |
| **DTPE** | 10.1109/icassp55912.2026.11460694 | 2026 / **ICASSP 2026** | 文档树解析 + LLM 数据精修 |
| **Coarse-to-Fine** | 10.1109/icassp55912.2026.11461490 | 2026 / **ICASSP 2026** | 低资源 DocRE 的粗到细两阶段 |
| **MAUM** | 10.1109/IJCNN64981.2025.11228956 | 2025 / IJCNN | **U-Mamba 替代 Transformer** + 记忆增强 |
| Two-Stage Loss + Anaphor | 10.1109/IJCNN64981.2025.11228050 | 2025 / IJCNN | 两阶段 loss + 指代消解 |
| Correction & Completion | 10.1109/ICAACE65325.2025.11019681 | 2025 / ICAACE | LLM 当后处理器（**Y2 verifier 的直接 motivation**） |
| **Hallucination-Resistant RE** | 2508.14391 | 2025-08 / arXiv | LLM 幻觉抑制，Y2 motivation |

### 第二档：相关但不直接 follow（1 篇）

| 论文 | 关系 |
|---|---|
| Can an LLM Induce a Graph? (2510.03611, ICKG 2025) | 分析 LLM 长文档下的"记忆漂移"，**可作为"DocRE 长文档问题仍存在"的引证素材** |

### 第三档：领域应用顺手引（3 篇，开题不用看）

- LLMs-Based Documents Classification (UBMK 2025)
- Brazilian Judicial Healthcare NER (ISCC 2025)
- EarthSE (2505.17139)

## 🔧 工具与 API 备注

- **OpenAlex**：不限流，覆盖正式期刊/会议引用，但**滞后 6-12 个月**，arXiv 预印本之间的引用拿不到。**裸 `search` 查询会跨学科污染，见 `03-trends.md` 顶部的口径警告**
- **Semantic Scholar**：覆盖 arXiv 引用，但**公开 API 严格限流（~100 次/5min/IP）**
- **建议**：申请免费的 SS API Key（[这里](https://www.semanticscholar.org/product/api#api-key-form)）后限流升至 ~1000 次/min

