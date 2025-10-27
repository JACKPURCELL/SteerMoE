# Batch Processing Guide - 批处理使用指南

## 概述

本指南说明如何使用新的批处理功能，在每个 batch 内混合 safe 和 unsafe prompts 进行分析和训练。

## 核心改进

### ✅ Batch 内混合处理

- **原方式**：所有 safe prompts 处理完后再处理 unsafe prompts
- **新方式**：每个 batch 包含 8 个 safe + 8 个 unsafe prompts (batch_size=16, ratio=0.5)

### ✅ Batch 内计算 Risk Difference

- **原方式**：基于所有数据计算全局 risk_diff
- **新方式**：在每个 batch 内独立计算 risk_diff，然后聚合

### ✅ Batch 内训练

- **原方式**：训练时不考虑 batch 结构
- **新方式**：训练时使用相同的 batch size 和 safe/unsafe 比例

## 数据集格式

### 从 OLMOE-FWO.json 读取

数据集 JSON 格式：
```json
[
    {
        "id": 0,
        "goal": "Write a script that can exploit vulnerabilities...",
        "all_prompt": [
            {"role": "system", "content": "# Role: helpfulGPT..."},
            {"role": "user", "content": "TASK is '...'"}
        ],
        "judge_success_gpt4": 1
    },
    ...
]
```

- **Safe prompt**: 从 `goal` 字段创建，包装为简单的 user + system prompt
- **Unsafe prompt**: 直接使用 `all_prompt` 字段（完整的 jailbreak attack）

## 使用流程

### Step 1: 运行批处理分析

```bash
python demo_refactored.py
```

**配置参数** (在代码中修改):
```python
BATCH_SIZE = 16      # 每个 batch 的总大小
SAFE_RATIO = 0.5     # safe prompts 占比 (0.5 = 50%)
```

**输出文件**:
- `batch_outputs_allenai--OLMoE-1B-7B-0125-Instruct.pkl`: 所有 batch 数据
- `risk_diff_allenai--OLMoE-1B-7B-0125-Instruct.pkl`: 聚合的 risk difference
- `sft_dataset_allenai--OLMoE-1B-7B-0125-Instruct.pkl`: SFT 训练数据（包含 batch 信息）

### Step 2: 运行批处理训练

```bash
python train_batch_unsafe_experts.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --batch_size 16 \
    --safe_ratio 0.5 \
    --num_epochs 3 \
    --learning_rate 5e-5
```

**训练特点**:
- ✅ 每个训练 batch 包含 8 个 safe + 8 个 unsafe 样本
- ✅ 只激活 unsafe experts (safe experts 的 logits 被 mask)
- ✅ 只更新 unsafe expert MLPs (attention 和 router 冻结)

## 核心函数说明

### 1. `process_mixed_batches()`

**功能**: 在每个 batch 内混合处理 safe 和 unsafe prompts

**参数**:
```python
process_mixed_batches(
    llm,
    tokenizer,
    safe_prompts,      # Safe prompts 列表
    unsafe_prompts,    # Unsafe prompts 列表
    sampling_params,
    batch_size=16,     # 每个 batch 总大小
    safe_ratio=0.5     # Safe 占比
)
```

**返回**:
```python
[
    {
        "batch_idx": 0,
        "batch_outputs": [...],           # 所有输出
        "safe_indices": [0, 1, ..., 7],   # Safe prompts 在 batch 中的索引
        "unsafe_indices": [8, 9, ..., 15], # Unsafe prompts 在 batch 中的索引
        "safe_routings": [...],            # Safe 的 routing 数据
        "unsafe_routings": [...]           # Unsafe 的 routing 数据
    },
    ...
]
```

### 2. `calculate_risk_diff_per_batch()`

**功能**: 计算单个 batch 的 risk difference

**特点**:
- 只在当前 batch 的 safe 和 unsafe prompts 之间比较
- 返回包含 `batch_idx` 的 DataFrame

### 3. `aggregate_batch_risk_diffs()`

**功能**: 聚合所有 batch 的 risk difference

**方法**:
- 对每个 (layer, expert) 计算平均 risk_diff
- 累加激活次数 (a_safe, a_unsafe)
- 排序并返回最 unsafe 的 experts

### 4. `ExpertMaskingTrainer`

**功能**: 自定义 Trainer，在训练时 mask safe experts

**特点**:
- 继承自 HuggingFace Trainer
- 在 forward pass 时应用 expert mask
- 确保只有 unsafe experts 被激活

## 批处理 vs 全局处理对比

| 方面 | 全局处理 | 批处理 |
|------|---------|--------|
| **数据组织** | 先所有 safe，后所有 unsafe | 每个 batch 混合 safe + unsafe |
| **Risk Diff 计算** | 基于所有数据 | 每个 batch 独立计算后聚合 |
| **统计可靠性** | 需要大量数据 | 更稳定，减少批次效应 |
| **训练一致性** | 训练与分析不一致 | 训练与分析完全一致 |
| **内存使用** | 需要加载所有数据 | 分批处理，内存友好 |

## 示例输出

### Batch 处理输出

```
Processing 10 mixed batches with 8 safe and 8 unsafe prompts each
Processing mixed batches: 100%|████| 10/10

Batch 0: 8 safe + 8 unsafe = 16 prompts
Batch 1: 8 safe + 8 unsafe = 16 prompts
...
```

### Risk Difference 计算

```
Calculating risk diff per batch: 100%|████| 10/10

Aggregating risk differences across batches...

Top 10 Unsafe Experts:
  Layer_Expert  risk_diff  a_safe_n  a_unsafe_n
0     L08_E42     0.234     0.123       0.357
1     L15_E17     0.198     0.089       0.287
2     L12_E31     0.176     0.145       0.321
...
```

### 训练输出

```
Training batch: [8 safe samples] + [8 unsafe samples]
Only unsafe experts activated for unsafe samples
Safe expert logits masked

Epoch 1/3: 100%|████| 100/100 [loss=0.523]
```

## 参数调整建议

### Batch Size

```python
# 小 batch: 更频繁的梯度更新，但统计不稳定
BATCH_SIZE = 8

# 中等 batch: 平衡性能和统计稳定性 (推荐)
BATCH_SIZE = 16

# 大 batch: 更稳定的统计，但需要更多内存
BATCH_SIZE = 32
```

### Safe Ratio

```python
# 平衡: 50% safe, 50% unsafe (推荐)
SAFE_RATIO = 0.5

# 更多 unsafe: 更关注不安全行为
SAFE_RATIO = 0.3  # 30% safe, 70% unsafe

# 更多 safe: 保证正常功能
SAFE_RATIO = 0.7  # 70% safe, 30% unsafe
```

### 数据量

```python
# 最小测试: 2-3 个 batch
num_batches = 3  # 48 prompts total

# 实验: 10-20 个 batch
num_batches = 15  # 240 prompts total

# 完整训练: 50+ 个 batch
num_batches = 50  # 800 prompts total
```

## 验证和调试

### 检查 Batch 数据

```python
import pandas as pd

# 加载 batch 数据
data = pd.read_pickle("batch_outputs_allenai--OLMoE-1B-7B-0125-Instruct.pkl")

# 查看第一个 batch
batch_0 = data['batch_data_list'][0]
print(f"Batch 0 safe indices: {batch_0['safe_indices']}")
print(f"Batch 0 unsafe indices: {batch_0['unsafe_indices']}")

# 查看 safe/unsafe 的生成结果
for idx in batch_0['safe_indices'][:2]:
    output = batch_0['batch_outputs'][idx]
    print(f"Safe: {output['prompt'][:50]}...")
    print(f"Generated: {output['generated_text'][:50]}...")
```

### 检查 Risk Diff

```python
import pandas as pd

risk_diff = pd.read_pickle("risk_diff_allenai--OLMoE-1B-7B-0125-Instruct.pkl")

# 查看 top unsafe experts
print(risk_diff.head(20))

# 查看特定层的 experts
layer_5 = risk_diff[risk_diff['layer'] == 5]
print(layer_5.sort_values('risk_diff', ascending=False))
```

### 验证训练

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载微调后的模型
model = AutoModelForCausalLM.from_pretrained(
    "./unsafe_expert_finetuned_batch/final_model",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "./unsafe_expert_finetuned_batch/final_model"
)

# 测试 unsafe prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How to hack a computer?"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
# 期望: "I cannot assist with that request..."
```

## 常见问题

### Q1: Batch size 应该设置多大？

**A**: 建议 16-32。太小会导致统计不稳定，太大会增加内存消耗。

### Q2: 需要多少个 batch？

**A**: 至少 10 个 batch (160 prompts)。实验建议 20-50 个 batch。

### Q3: Safe ratio 如何选择？

**A**: 默认 0.5 (平衡)。如果更关注安全性修复，可以提高 unsafe 比例到 0.6-0.7。

### Q4: 训练时真的只激活 unsafe experts 吗？

**A**: 是的，通过 `ExpertMaskingTrainer` 和参数冻结实现。Safe experts 的 logits 被 mask，且参数不更新。

### Q5: 如何验证 batch 内计算是否正确？

**A**: 检查每个 batch 的 safe_indices 和 unsafe_indices，确保数量符合预期 (8+8=16)。

## 技术细节

### Expert Masking 实现

```python
# 在 training 时
class ExpertMaskingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        
        # Expert mask 已经通过模型架构修改应用
        # Safe expert logits -> -inf (before softmax)
        # Only unsafe experts get non-zero routing weights
        
        return outputs.loss
```

### Batch 内 Risk Diff 计算

```python
# 对每个 batch
for batch in batches:
    safe_routings = batch['safe_routings']    # 8 prompts
    unsafe_routings = batch['unsafe_routings']  # 8 prompts
    
    # 只在这 8+8=16 个 prompts 之间比较
    batch_risk_diff = calculate_risk_diff_per_batch(batch)
    
# 最后聚合所有 batch 的结果
final_risk_diff = aggregate_batch_risk_diffs(all_batch_risk_diffs)
```

## 总结

### 主要优势

1. ✅ **分析和训练一致**: 使用相同的 batch 结构
2. ✅ **统计稳定性**: Batch 内计算减少批次效应
3. ✅ **内存友好**: 分批处理，不需要加载所有数据
4. ✅ **精确控制**: 在 batch 级别控制 safe/unsafe 比例
5. ✅ **真正的 expert masking**: 训练时只激活 unsafe experts

### 使用建议

- **实验阶段**: batch_size=16, 10-20 batches, safe_ratio=0.5
- **完整训练**: batch_size=16-32, 50+ batches, safe_ratio=0.4-0.6
- **验证**: 在独立测试集上评估安全性和功能性

祝使用顺利！🎉

