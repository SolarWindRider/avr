# Qwen3-VL-2B 上 HEVA_light 实验开发文档

## 1. 实验目标（Goal）

在 **不进行 no-image rerun** 的前提下，验证并量化：

> **模型在生成过程中，高熵 token 是否显著依赖视觉 token（image tokens）进行决策。**

具体目标：

1. 在 Qwen3-VL-2B-Instruct 上完成一次 **带图像的 rollout**
2. 记录生成过程中：

   * token-level logits → 计算 entropy
   * decoder self-attention → 计算对视觉 token 的注意力占比
3. 使用 **Top 20% 高熵 token** 构造 **HEVA_light 指标**
4. 输出：

   * HEVA_light 数值
   * token-level 可解释统计（哪些 token “看图”）

---

## 2. 模型与前提假设

### 2.1 模型选择

* 模型：`Qwen/Qwen3-VL-2B-Instruct`
* 框架：HuggingFace Transformers
* 推理模式：`generate()`（非训练）

### 2.2 关键结构假设（非常重要）

Qwen3-VL 和 Qwen2.5-VL 在 **跨模态机制上是同一范式**：

1. **无 cross-attention**
2. 图像 → vision encoder → patch tokens
3. patch tokens → merger → embedding
4. embedding 作为 **序列前缀 token**
5. decoder **self-attention 同时 attend 文本 + 图像 token**

👉 结论：
**检测“是否使用图像信息”，只需看 decoder self-attention**

---

## 3. 核心概念定义（Coding 必须明确）

### 3.1 Token 类型划分

在 decoder 的 attention key 维度中：

| token 类型      | 说明                               |
| ------------- | -------------------------------- |
| visual tokens | 图像 patch 经 merger 后的 token（序列前缀） |
| text tokens   | prompt + 已生成 token               |

你需要明确一个东西：

> **visual token 的 index 范围：[0, V−1]**

V 可以通过 processor 或 attention mask 推断。

---

### 3.2 高熵 token（High-Entropy Token）

对每个生成步骤 (t)：

* logits：(z_t \in \mathbb{R}^{|\mathcal{V}|})
* 概率：(p_t = \mathrm{softmax}(z_t))
* 熵：
  [
  H_t = -\sum_v p_{t,v}\log p_{t,v}
  ]

排序后取：

[
S = \text{Top-20%}(H_1, \dots, H_T)
]

⚠️ 注意：

* **只在生成 token 上算**
* 不包括 prompt token

---

### 3.3 HEVA_light（核心指标）

对每个 (t \in S)：

[
R_{\mathrm{vis}}(t) =
\frac{
\sum_{h=1}^{H}\sum_{k=0}^{V-1} A_{h,t,k}
}{
\sum_{h=1}^{H}\sum_{k=0}^{K-1} A_{h,t,k}
}
]

最终：

[
\mathrm{HEVA_light} =
\frac{1}{|S|}
\sum_{t\in S}
H_t \cdot R_{\mathrm{vis}}(t)
]

---

## 4. 系统整体架构（Implementation View）

```
┌──────────────┐
│ Image + Text │
└──────┬───────┘
       ↓
┌───────────────────────┐
│ Qwen3-VL Processor    │
│  - image tokens       │
│  - text tokens        │
└──────┬────────────────┘
       ↓
┌─────────────────────────────┐
│ Qwen3-VL Model (generate)   │
│                             │
│  ┌───────────────────────┐ │
│  │ Decoder Layer L       │ │◀── hook attention
│  │  - self-attn          │ │
│  └───────────────────────┘ │
│                             │◀── hook logits
└──────┬──────────────────────┘
       ↓
┌──────────────────────────┐
│ HEVA_light Computation   │
└──────────────────────────┘
```

---

## 5. Hook 设计规范（给 coding agent 的硬约束）

### 5.1 必须 Hook 的对象

#### （1）Decoder Self-Attention

目标模块（示例）：

```python
model.model.language_model.layers[i].self_attn
```

需要捕获：

* attention weights
* shape: `(batch, heads, query_len, key_len)`

⚠️ 注意：

* `output` 里通常 **不直接返回 attention**
* 需要：

  * 使用 `output_attentions=True`
  * 或 monkey-patch forward

---

#### （2）Logits（生成 token）

捕获方式：

* 使用 `generate(..., return_dict_in_generate=True, output_scores=True)`
* `scores[t]` 即第 t 步 logits

---

### 5.2 Hook 输出规范（必须遵守）

| 数据                 | 形状             | 说明     |    |     |
| ------------------ | -------------- | ------ | -- | --- |
| attention          | `[L, H, T, K]` | L=选取层数 |    |     |
| logits             | `[T,           | V      | ]` | 生成步 |
| visual_token_count | int            | V      |    |     |

---

## 6. 实验流程（Step-by-step）

### Step 1：准备输入

* 单张图像 + 一个视觉推理 prompt
* prompt 中 **不要泄露答案**

---

### Step 2：一次 generate rollout

* max_new_tokens：建议 64~128
* temperature > 0（保证熵有区分度）

---

### Step 3：收集数据

* logits → 计算 entropy
* attention → 切出 visual token attention

---

### Step 4：计算 HEVA_light

* 排序 entropy
* 选 top 20%
* 计算 attention ratio
* entropy 加权平均

---

### Step 5：输出与 sanity check

至少输出：

```text
Total tokens: 78
High-entropy tokens: 16
Mean entropy: 3.21
Mean visual attention ratio (high-entropy): 0.34
HEVA_light: 1.09
```

---

## 7. Sanity Check（非常重要）

你必须验证以下现象：

1. **描述性问题（Describe the image）**

   * HEVA_light 应明显 > 0
2. **纯文本问题（无需图像）**

   * HEVA_light ≈ 0
3. **错误图像 / 随机噪声图像**

   * HEVA_light 显著下降

否则说明：

* visual token index 错
* attention hook 错
* entropy 算错

---

## 8. 预期实验结论（Research Claim）

你希望最终能支持以下论断：

> *High-entropy tokens concentrate visual attention, indicating that visual evidence is primarily used when the model resolves uncertain decisions.*

这句话**非常适合直接写进 paper**。

---

## 9. 后续可扩展方向（给未来的你）

* 用 HEVA_light 做：

  * GRPO / RTPO reward
  * rollout filtering
  * curriculum（先高 HEVA 样本）
* 对比：

  * 不同层 attention
  * 不同模型（Qwen2.5-VL vs Qwen3-VL）

---

## 10. 一句话总结（给 vibe coding 工具）

> **目标：**
> 在 Qwen3-VL-2B-Instruct 的一次生成过程中，
> 使用 decoder self-attention + token entropy，
> 仅在 top 20% 高熵 token 上计算其对视觉 token 的注意力占比，
> 得到 HEVA_light 指标，无需 no-image rerun。

