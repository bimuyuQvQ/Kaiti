# 必读论文清单（2026-05-27 整理）

> 本文件入库；PDF 文件全部在 `papers/`（被 gitignore，不入库，需要在每台机器上重新下载）。
> 重新下载脚本见本文件末尾"复现下载"区块。

## 优先级速查

| 档位 | 用途 | 论文数 | 阅读策略 |
|---|---|---|---|
| ⭐⭐⭐ 第一档 | 方案 A' 直接技术依赖 | 4 篇 | **精读 + 代码对照** |
| ⭐⭐ 第二档 | 避免撞车的 2025 顶会论文 | 5 篇 | **方法精读** |
| ⭐ 第三档 | 方案 B' 长尾路线候选 | 4 篇 | 选定 B' 时精读，否则泛读 |
| ⭐ 第四档 | 方案 C' 联合抽取候选 | 2 篇 | 选定 C' 时精读，否则泛读 |
| ○ 第五档 | 开题"相关工作"章节素材 | 8 篇 | 泛读摘要 + 引用即可 |

总计：**24 篇 PDF**（22.6 MB），已全部下载到 `papers/`。

---

## ⭐⭐⭐ 第一档：方案 A' 直接技术依赖（4 篇，必精读）

第一章基线和改进点的技术基础。必须读完才能动手 P0 复现 + Y1/Y4 设计。

| # | 论文 | 文件 | 作用 |
|---|---|---|---|
| 1 | **TTM-RE** (ACL 2024 Long, arXiv:2406.05906) | `TTM-RE_baseline_2024_ACL_Long.pdf` | 第一章基线本身。已精读 → 见 `notes-ttmre.md` |
| 2 | **ATLOP** (AAAI 2021, arXiv:2010.11304) | `ATLOP_adaptive_thresholding.pdf` | TTM-RE 直接前驱，提出 adaptive threshold loss 和 localized context pooling。TTM-RE 的 encoder + classifier 范式全部沿用 ATLOP |
| 3 | **SSR-PU** (EMNLP 2022, arXiv:2206.08709) | `SSR-PU_distant_false_negative.pdf` | TTM-RE 用的 loss（处理 distant supervision 的 false negative）。直接关系到第一章 ablation 设计 |
| 4 | **DREEAM** (EACL 2023, arXiv:2302.08675) | `DREEAM_evidence_attention.pdf` | DocRE 主流 baseline，evidence-guided attention。TTM-RE 在 H+D setting 主要赢的就是 DREEAM |

**阅读顺序**：ATLOP → DREEAM → SSR-PU → TTM-RE（按时间倒序，便于理解每一步加了什么）

---

## ⭐⭐ 第二档：避免撞车的 2025 顶会论文（5 篇，必方法精读）

如果不读这 5 篇，第一章的改进点很可能会撞 Fu Zhang 团队 2025 已经发的工作。

| # | 论文 | 文件 | 风险 |
|---|---|---|---|
| 5 | **DRELL** (NAACL 2025 Main Long) | `DRELL_LLM_refiner_2025_NAACL_Main.pdf` | 🔴 决定 Y2/Y3 死亡的论文。读完确认到底覆盖了什么，确保我们的方案不撞 |
| 6 | **GREP** (ACL 2025 Findings) | `GREP_global_relations_2025_ACL_Findings.pdf` | 🟡 抢了"entity pair reasoning"和"全局关系预测辅助任务"。读完看 Y4 InfoNCE 是否跟它的 entity pair representation 撞 |
| 7 | **AMTL** (ACL 2025 Findings) | `AMTL_multi-threshold_loss_2025_ACL_Findings.pdf` | 🔴 长尾 loss 已被它覆盖。读完看 Y1（retriever 替换）是否能跟 AMTL 叠加 |
| 8 | **ET-MIER** (EMNLP 2025 Findings) | `ET-MIER_entity_type_2025_EMNLP_Findings.pdf` | 🟡 抢了 entity type + evidence retrieval。读完看 Y1 retriever 是否跟它的 evidence retrieval 撞 |
| 9 | **EP-RSR** (NAACL 2025 Findings) | `EP-RSR_entity_pair_LLM_2025_NAACL_Findings.pdf` | 🟡 LLM-based DocRE 新范式（明确说还比不过 SLM SOTA）。读完确认 LLM 路线整体边界 |

**阅读顺序**：DRELL → AMTL（最重要的两个）→ GREP → ET-MIER → EP-RSR

---

## ⭐ 第三档：方案 B' 长尾路线候选（4 篇，B' 选定后精读）

只在用户选方案 B'（full-shot + long-tail）时精读。否则泛读 AMTL（第二档已有）即可。

| # | 论文 | 文件 | 切入点 |
|---|---|---|---|
| 10 | **DOREMI** (2026-01, arXiv:2601.11190) | `DOREMI_long-tail_active_learning_2026.pdf` | Iterative active learning for long-tail，model-agnostic 框架。**B' 必读**——AMTL 是 loss 角度，DOREMI 是 active annotation 角度，留给我们的角度只剩 retrieval-based |
| 11 | **VaeDiff-DocRE** (COLING 2025, arXiv:2412.13503) | `VaeDiff-DocRE_long-tail_data_aug.pdf` | VAE + Diffusion 做 long-tail data augmentation。**B' 必读**——第三个长尾角度已被占 |
| 12 | **GLiDRE** (arXiv:2508.00757) | `GLiDRE_few-shot_DocRE.pdf` | Few-shot DocRE 的 generalist lightweight model，SOTA in few-shot |
| 13 | **CDER** (arXiv:2504.06529) | `CDER_collaborative_evidence_retrieval.pdf` | Collaborative evidence retrieval，跟 Y1 retriever 路线相关 |

---

## ⭐ 第四档：方案 C' 联合抽取候选（2 篇，C' 选定后精读）

只在用户选方案 C'（DocRE → JERE）时精读。

| # | 论文 | 文件 | 切入点 |
|---|---|---|---|
| 14 | **Re2-DocRED** (EACL 2026 Main Long) | `Re2-DocRED_joint_entity_relation_2026_EACL.pdf` | C' 的核心数据集论文。Re-DocRED 增强版（+27% triplets）+ JERE benchmark 设定。**C' 必读** |
| 15 | **CsEAE** (arXiv:2411.05895) | `CsEAE_DocEAE_small_large_collab.pdf` | DocEAE 小+大模型协作（之前下载过）。C' 路线"小模型 + LLM"协作的参考 |

---

## ○ 第五档：开题"相关工作"章节素材（8 篇，泛读摘要即可）

不需要精读，但开题报告的"相关工作"章节需要引用、对比、画 baseline 表。

| # | 论文 | 文件 | 用途 |
|---|---|---|---|
| 16 | **KD-DocRE** (ACL Findings 2022, arXiv:2204.13257) | `KD-DocRE_adaptive_focal_loss.pdf` | 长尾 loss 经典工作（adaptive focal loss），AMTL 的对照 |
| 17 | **LMRC** (arXiv:2408.13889) | `LMRC_two-stage_DocRE.pdf` | DocRE × LLM 两阶段范式起点 |
| 18 | **RelPrior** (arXiv:2511.08143) | `RelPrior_LLM_paradigm.pdf` | LMRC 范式的对偶（"关系作为先验"）|
| 19 | **KnowRA** (arXiv:2501.00571) | `KnowRA_knowledge_retrieval.pdf` | Knowledge retrieval augmented DocRE，跟 Y1 retriever 路线对比 |
| 20 | **Hallucination-Resistant RE** (arXiv:2508.14391) | `Hallucination-Resistant_RE.pdf` | LLM 幻觉抑制，开题报告"动机"章节背景素材 |
| 21 | **Zero-Shot Biomedical DocRE** (arXiv:2505.01077) | `Zero-Shot_Biomedical_DocRE.pdf` | LLM zero-shot DocRE，开题"相关工作"对比 |
| 22 | **SciNLP** (EMNLP 2025 Main) | `SciNLP_benchmark_2025_EMNLP_Main.pdf` | 2025 EMNLP Main 新 benchmark，证明 DocRE 方向不过时的硬证据 |
| 23 | **CaDRL** (COLING 2025 Main) | `CaDRL_rule_learning_2025_COLING_Main.pdf` | 2025 COLING Main rule learning DocRE，方法多样性证据 |
| 24 | **GLiM** (ACL 2025 Findings) | `GLiM_graph_LLM_biomedical_2025_ACL_Findings.pdf` | Graph + LLM 生物医学 DocRE |

---

## 阅读时间预估

| 任务 | 论文数 | 单篇精读 | 总时长 |
|---|---|---|---|
| 第一档（精读+代码对照）| 4 | 2-3 小时 | 8-12 小时 |
| 第二档（方法精读）| 5 | 1-1.5 小时 | 5-8 小时 |
| 第三档（B' 选定后）| 4 | 1 小时 | 4 小时 |
| 第四档（C' 选定后）| 2 | 1.5 小时 | 3 小时 |
| 第五档（泛读摘要）| 9 | 15 分钟 | 2-3 小时 |

**主线（第一+第二档，方案 A' 已确定下要做的）**：~15-20 小时阅读，可在 3-5 天完成。

---

## 复现下载（跨机器）

如果在新机器上（如实验室 Linux 服务器）需要重新下载这 24 篇 PDF，运行：

```powershell
# Windows PowerShell
$ProgressPreference='SilentlyContinue'
$dst = "papers"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# arxiv 论文（15 篇）
$arxiv = @{
  "2406.05906"="TTM-RE_baseline_2024_ACL_Long.pdf";
  "2010.11304"="ATLOP_adaptive_thresholding.pdf";
  "2206.08709"="SSR-PU_distant_false_negative.pdf";
  "2302.08675"="DREEAM_evidence_attention.pdf";
  "2204.13257"="KD-DocRE_adaptive_focal_loss.pdf";
  "2408.13889"="LMRC_two-stage_DocRE.pdf";
  "2501.00571"="KnowRA_knowledge_retrieval.pdf";
  "2412.13503"="VaeDiff-DocRE_long-tail_data_aug.pdf";
  "2504.06529"="CDER_collaborative_evidence_retrieval.pdf";
  "2508.14391"="Hallucination-Resistant_RE.pdf";
  "2505.01077"="Zero-Shot_Biomedical_DocRE.pdf";
  "2511.08143"="RelPrior_LLM_paradigm.pdf";
  "2508.00757"="GLiDRE_few-shot_DocRE.pdf";
  "2411.05895"="CsEAE_DocEAE_small_large_collab.pdf";
  "2601.11190"="DOREMI_long-tail_active_learning_2026.pdf"
}
foreach($k in $arxiv.Keys) {
  Invoke-WebRequest -Uri "https://arxiv.org/pdf/$k" -OutFile (Join-Path $dst $arxiv[$k]) -UserAgent "Mozilla/5.0"
}

# ACL Anthology 论文（9 篇）
$acl = @{
  "2025.naacl-long.319"="DRELL_LLM_refiner_2025_NAACL_Main.pdf";
  "2025.findings-acl.1002"="GREP_global_relations_2025_ACL_Findings.pdf";
  "2025.findings-acl.1081"="AMTL_multi-threshold_loss_2025_ACL_Findings.pdf";
  "2025.findings-emnlp.961"="ET-MIER_entity_type_2025_EMNLP_Findings.pdf";
  "2025.findings-naacl.224"="EP-RSR_entity_pair_LLM_2025_NAACL_Findings.pdf";
  "2026.eacl-long.213"="Re2-DocRED_joint_entity_relation_2026_EACL.pdf";
  "2025.coling-main.551"="CaDRL_rule_learning_2025_COLING_Main.pdf";
  "2025.emnlp-main.732"="SciNLP_benchmark_2025_EMNLP_Main.pdf";
  "2025.findings-acl.727"="GLiM_graph_LLM_biomedical_2025_ACL_Findings.pdf"
}
foreach($k in $acl.Keys) {
  Invoke-WebRequest -Uri "https://aclanthology.org/$k.pdf" -OutFile (Join-Path $dst $acl[$k]) -UserAgent "Mozilla/5.0"
}
```

Linux/macOS 等效（curl）：

```bash
mkdir -p papers && cd papers
# arxiv
for entry in "2406.05906:TTM-RE_baseline_2024_ACL_Long.pdf" \
             "2010.11304:ATLOP_adaptive_thresholding.pdf" \
             "2206.08709:SSR-PU_distant_false_negative.pdf" \
             "2302.08675:DREEAM_evidence_attention.pdf" \
             "2204.13257:KD-DocRE_adaptive_focal_loss.pdf" \
             "2408.13889:LMRC_two-stage_DocRE.pdf" \
             "2501.00571:KnowRA_knowledge_retrieval.pdf" \
             "2412.13503:VaeDiff-DocRE_long-tail_data_aug.pdf" \
             "2504.06529:CDER_collaborative_evidence_retrieval.pdf" \
             "2508.14391:Hallucination-Resistant_RE.pdf" \
             "2505.01077:Zero-Shot_Biomedical_DocRE.pdf" \
             "2511.08143:RelPrior_LLM_paradigm.pdf" \
             "2508.00757:GLiDRE_few-shot_DocRE.pdf" \
             "2411.05895:CsEAE_DocEAE_small_large_collab.pdf" \
             "2601.11190:DOREMI_long-tail_active_learning_2026.pdf"; do
  id="${entry%%:*}"; name="${entry##*:}"
  curl -L -o "$name" "https://arxiv.org/pdf/$id"
done
# acl
for entry in "2025.naacl-long.319:DRELL_LLM_refiner_2025_NAACL_Main.pdf" \
             "2025.findings-acl.1002:GREP_global_relations_2025_ACL_Findings.pdf" \
             "2025.findings-acl.1081:AMTL_multi-threshold_loss_2025_ACL_Findings.pdf" \
             "2025.findings-emnlp.961:ET-MIER_entity_type_2025_EMNLP_Findings.pdf" \
             "2025.findings-naacl.224:EP-RSR_entity_pair_LLM_2025_NAACL_Findings.pdf" \
             "2026.eacl-long.213:Re2-DocRED_joint_entity_relation_2026_EACL.pdf" \
             "2025.coling-main.551:CaDRL_rule_learning_2025_COLING_Main.pdf" \
             "2025.emnlp-main.732:SciNLP_benchmark_2025_EMNLP_Main.pdf" \
             "2025.findings-acl.727:GLiM_graph_LLM_biomedical_2025_ACL_Findings.pdf"; do
  id="${entry%%:*}"; name="${entry##*:}"
  curl -L -o "$name" "https://aclanthology.org/$id.pdf"
done
```
