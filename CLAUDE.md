# 项目背景

## 目标

从零实现一个垂直领域中文语言模型的完整训练流程，目标领域为**日本战国时期**（约1180–1615年）。SFT 产物是一个能以**结构化 Markdown 格式**回答战国历史问题的中文问答模型。

---

## 训练流程

三阶段串行流程，基座唯一，后续阶段在其上独立展开：

```
基础预训练（通用中文基座）
│
└──▶ 持续预训练 CPT（日本战国语料）
        │
        └──▶ SFT（结构化 Markdown 问答）
```

**当前进度：**
- 基础预训练：**已完成**（50.4B tokens，eval loss ≈ 2.4778）
- CPT 语料构建：**进行中**（日文维基 HTML 清洗、PDF 书籍提取）
- CPT 训练：未开始
- SFT：未开始

---

## 模型配置

Decoder-Only Transformer，总参数量约 **1.12B**。

| 参数 | 值 |
|------|----|
| `vocab_size` | 40960 |
| `hidden_size` | 1536 |
| `num_layers` | 36 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 16（MHA，无 GQA）|
| `intermediate_size` | 4096 |
| `max_position_embeddings` | 4096 |
| `rope_base` | 10000 |
| `rms_eps` | 1e-6 |
| `dropout_prob` | 0.0 |

**架构特征：** Pre-Norm、SwiGLU MLP、RoPE、RMSNorm，通过 Liger Kernel 融合 lm_head + cross-entropy（`LigerFusedLinearCrossEntropyLoss`），lm_head 权重不与 embedding 共享。

---

## 硬件与环境

- **GPU：** 4 × H800 80GB（共 320GB 显存）
- **分布式：** DDP
- **精度：** BF16
- **CUDA：** ≤ 12.7（使用 cu126 PyTorch wheels）
- **Liger Kernel：** Triton JIT，目标 sm_90

---

## CPT 语料构成（进行中）

| 来源 | 状态 | 说明 |
|------|------|------|
| 日文维基百科 | 清洗中 | Parsoid HTML → BeautifulSoup 清洗 → LLM 翻译为中文 |
| 战国非虚构书籍 PDF | 提取中 | 多模态 LLM 提取结构化 JSON → Markdown |
| 日本外史 | 规划中 | 古汉语 → 现代白话中文转换 |

**语料质量原则：**
- 仅使用有学术资质作者的非虚构类书籍（Tier 1/2 优先）
- 排除历史小说、演义、架空历史
- 语料必须来源可追溯，禁止 LLM 自行补充内容（已因此经历一次全量重采集）

---

## SFT 规划（待启动）

- 数据组成：约 80–90% 战国领域 Q&A + 10–20% 通用指令数据
- Q&A 对由 LLM 基于 CPT 语料生成，LLM 负责格式与问题设计，不负责知识供给
- 按问题子类型映射不同 Markdown 模板（如因果单事件 → 转折结构；因果过程 → 分阶段结构）

---

## 关键设计原则

1. **确定性优先**：管道中尽量使用规则/代码，LLM 调用限于规则真正不够的环节，并严格限定输入输出范围。
2. **语料完整性**：CPT 数据必须来源可追溯，LLM 不得在清洗/翻译时补充原文不存在的内容。
3. **eval loss 可比性依赖受控条件**：评估数据集变更会破坏跨 run 的 loss 可比性，跨 run 对比应使用评分式生成质量评估。
4. **书籍筛选以作者为第一过滤器**：作者资质须通过检索核实，不可从书名或出版社推断。