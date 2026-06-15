> ⚠️ **2026-06-03 更新**：本文件部分表述已被推翻——Re2-DocRED 是**数据集论文**（baseline 实为 AutoRE/TaG/REBEL，不是"它的 JERE 方法"）；"实体 FN"存疑；"InfoNCE 平移"串法不如**关系 FN 主线**硬。**最新结论以 `todo260603.md` + `04-decisions.md`(2026-06-03) + `02-papers.md`(FN 主线区块) 为准。** 本文件保留作历史参考。

我把 C' 完整拆给你，不要求你立即去读论文——我直接告诉你每篇论文什么时候用、用来干什么。

## 一句话总览

**C' = 两章是"同一个故事的两个 setting"**：
- **第一章**：标准 DocRE（entity 已知）—— 在 **Re-DocRED** 上改进 TTM-RE
- **第二章**：JERE（entity 未知，要自己识别）—— 在 **Re2-DocRED** 上做端到端
- **两章的卖点**：你在第一章学到的 entity representation 改进，**在第二章 entity 未知的更难场景下依然有效**——这就是你毕业论文的核心 claim

## 两章具体做什么

### 第一章：DocRE on Re-DocRED（TTM-RE 改进版）

**输入**：文档 + 已经标注好的 entity mentions + coreference clusters
**输出**：每对 entity 之间预测关系

**做的事**：
1. 复现 TTM-RE 在 Re-DocRED 上的结果（F1 ~80）
2. 加一个改进 Y（**建议 Y4：entity pair representation 上加 InfoNCE 对比学习**，因为这个直接用你 B 会的经验，且能为第二章铺垫）
3. ablation + 对照实验

**为什么选 Y4 不选 Y1**：
- Y4（InfoNCE 加在 entity pair representation 上）改的是**表示学习**部分
- 表示学习的改进**可以平移到第二章**（第二章的 entity 也要表示）
- Y1（换 retriever）只对第一章有用，第二章用不上 → 两章脱节

### 第二章：JERE on Re2-DocRED（TTM-RE → TTM-JERE）

**输入**：文档纯文本（**不给 entity 标注**）
**输出**：自动识别 entity mentions → 聚类成 entities → 预测关系（三件事一次做）

**做的事**：
1. 在 TTM-RE 前面接 NER head（mention 识别）+ coref head（mention 聚类）
2. 复用第一章学好的 entity pair representation（**这就是两章的桥**）
3. 在 Re2-DocRED 上 end-to-end 训练
4. 主对照：Re2-DocRED 论文给的 JERE baseline

**架构示意**：

```
                          ┌─ NER head ──→ mentions
文档 ─→ encoder (BERT) ──┤
                          ├─ coref head ─→ entities (clusters)
                          │
                          └─ TTM memory ─→ entity pair repr (来自第一章)
                                                │
                                          ┌─────┴──────┐
                                          │ +InfoNCE   │ (来自第一章 Y4)
                                          └─────┬──────┘
                                                ↓
                                          relation prediction
```

第一章的 Y4 InfoNCE 在第二章里**自动复用**，工作量翻倍但代码工作量不翻倍。

## 两章的"关联"怎么写（这是 C' 卖给老师的核心）

故事线（开题报告写法）：

> "现有 DocRE 工作（TTM-RE 等）都假设 entity 已经被标好，但真实场景下 entity 提取本身就是 noisy 的、是 RE 上游的 error propagation 源头。
>
> 本文（1）先在 entity 已知的 Re-DocRED 上证明改进 entity pair representation 的方法（InfoNCE）能涨点；
> （2）再去掉 entity 已知的假设，在 Re2-DocRED 上做 end-to-end JERE，验证同一种 entity representation 学习目标在 entity 噪声场景下依然有效，且能缓解 error propagation。"

老师听完会觉得：**逻辑闭环、两章不重复、且解决了真实世界问题**——这是 A' 给不出的。

## 执行流程（周级时间线）

按你明年毕业反推，从现在（2026-05）开始有 ~10 个月。我留 3 个月 buffer，按 7 个月排：

| 阶段 | 周数 | 任务 | 产出 |
|---|---|---|---|
| **Phase 0**：复现 | W1-W4 | TTM-RE on Re-DocRED 跑通，F1 对得上论文 | 复现日志 + baseline 数 |
| **Phase 1**：第一章 Y4 | W5-W10 | 加 InfoNCE 到 entity pair repr，ablation 完整 | 第一章方法 + 实验 |
| **Phase 2**：JERE 准备 | W11-W12 | Re2-DocRED 数据 EDA + Re2-DocRED 论文的 baseline 跑通 | 数据理解 + 第二章 baseline |
| **Phase 3**：第二章实现 | W13-W22 | 接 NER head + coref head，端到端训练，调参 | 第二章方法 + 实验 |
| **Phase 4**：开题报告 | W23-W28 | 写第一/第二章 + 相关工作 + 实验对照表 | **开题报告** |
| **Phase 5**：Buffer | W29-W40 | 兜底实验、补 ablation、应对老师反馈 | — |

**关键里程碑**（决策点）：
- W4 末：TTM-RE 复现不上 → 全盘崩，回到记忆库重新选 baseline
- W10 末：Y4 不涨点 → 切到 Y1 / Y5（不算崩，第一章还在）
- W12 末：Re2-DocRED 数据有大坑 → 退回 A'（同 baseline + 两改进），损失 1 个月
- W22 末：第二章 JERE 端到端做不出来 → 退回 pipeline JERE（NER 用现成模型 + TTM-RE），仍能凑章

## 论文你现在该读哪几篇（不要读 24 篇）

按现在你的进度，**先读 3 篇就够，其它都先放着**：

| 优先 | 论文 | 文件名 | 看什么 |
|---|---|---|---|
| 🔥 第一周 | **TTM-RE** | `TTM-RE_baseline_2024_ACL_Long.pdf` | 整篇精读（已经写过笔记 `notes-ttmre.md`，直接看笔记 + 论文 Table 1/2） |
| 🔥 第一周 | **Re2-DocRED** | `Re2-DocRED_joint_entity_relation_2026_EACL.pdf` | **重点读 §2 数据集构造 + §4 JERE baseline + §5 实验**。这是第二章的数据集和起点 |
| 🔥 第二周 | **ATLOP** | `ATLOP_adaptive_thresholding.pdf` | 看 §3 entity pair representation 怎么算的，公式抄下来——你 Y4 InfoNCE 要加在这上面 |

**复现期间（W1-W4）顺手扫**（每篇 20 分钟，看 Abstract + Method 图 + Table 1）：

| 论文 | 看什么 | 为什么 |
|---|---|---|
| **DREEAM** | Method 图 | 主对照 baseline |
| **SSR-PU** | loss 公式 | TTM-RE 用的 loss，复现要懂 |
| **DRELL** | 整体 framework | 确认 Y2/Y3 路线确实被它占了，不能碰 |
| **GREP** | entity pair representation 图 | 确认你 Y4 InfoNCE 不跟它撞 |
| **AMTL** | loss 设计 | 确认你不做 long-tail loss，避开它 |

剩下 16 篇全部留给写开题报告"相关工作"章节时再看，按需引用即可。

## 第一步具体怎么动手（这周）

1. **今天/明天**：把 `Re2-DocRED_joint_entity_relation_2026_EACL.pdf` 看一遍 §2 + §4，确认这个数据集是真的能用（**这是 C' 方案能不能立的关键 gate**——如果 Re2-DocRED 数据集质量不行或者不公开代码，C' 立刻死掉，要退 A'）
2. **本周内**：约导师 30 分钟，方案换成 C' 跟他过一遍。**注意话术**：不要说"师兄说做不下去所以我换"，要说"调研发现 Re2-DocRED EACL 2026 刚出，可以做 DocRE → JERE 的任务进阶，两章衔接更自然，比单 baseline 两改进更稳"
3. **下周**：实验室服务器 P0 复现 TTM-RE 启动

## 风险对账

| 风险 | 概率 | 应对 |
|---|---|---|
| Re2-DocRED 数据集质量差 / 代码不开源 | 🟡 中（EACL 2026 刚发，社区还没充分使用过）| W11 之前做 EDA，不行立刻回退 A' |
| NER + coref 头训不出来 | 🟡 中（DocRE 圈很少做 end-to-end JERE）| 退回 pipeline JERE：NER 用现成模型（如 spaCy / Flair）+ TTM-RE 做 RE |
| Y4 InfoNCE 在 TTM-RE 上不 work | 🟡 中（DocRE 上对比学习已被 SAIS 做过）| 切 Y5（adaptive threshold 改进）或 Y1（retriever 替换）|
| 导师不接受 C'，坚持 A' | 🟢 低 | 你回去说"那继续 A'"，不损失什么——P0 复现的成果通用 |

要不要我现在就把 C' 切换 + 这套时间线 + 优先阅读顺序更新到记忆库？方案排序变了（A'→C'）+ Y 选择变了（推荐 Y4 over Y1）+ 时间线 + reading priority 都值得落盘，下次跨机或者下一轮对话可以直接接上。