# Per-Batch Unsafe Experts - 每个 Batch 有自己的 Unsafe Experts

## 核心改进

**现在每个 batch 都有自己独立识别的 unsafe experts！**

### 之前的实现

```python
# 所有 batch 共享同一个 unsafe experts 列表
all_batches = [batch_0, batch_1, batch_2, ...]
↓ 计算每个 batch 的 risk_diff
↓ 聚合所有 batch
↓ 识别一个全局的 unsafe_experts 列表
global_unsafe_experts = [(L0, E5), (L2, E8), ...]  # 所有 batch 共用
```

### 现在的实现

```python
# 每个 batch 有自己的 unsafe experts
batch_0 → risk_diff_0 → unsafe_experts_0 = [(L0, E5), (L1, E12), ...]
batch_1 → risk_diff_1 → unsafe_experts_1 = [(L0, E8), (L2, E3), ...]
batch_2 → risk_diff_2 → unsafe_experts_2 = [(L1, E15), (L3, E7), ...]
...

# 每个 batch 有自己的 expert mask
batch_0 → expert_mask_0
batch_1 → expert_mask_1
batch_2 → expert_mask_2
```

## 为什么需要 Per-Batch Experts？

### 1. 不同的 unsafe prompts 激活不同的 experts

```
Batch 0 (hacking prompts):
  → 可能更多激活 L5_E12, L8_E7 (技术相关 experts)

Batch 1 (violence prompts):
  → 可能更多激活 L3_E9, L10_E15 (暴力相关 experts)

Batch 2 (illegal content):
  → 可能更多激活 L2_E4, L7_E11 (法律相关 experts)
```

### 2. 更精细的控制

- 可以针对性地修复每个 batch 对应的 experts
- 避免过度修复不相关的 experts
- 保持模型在其他领域的能力

### 3. 更好的可解释性

- 可以分析哪些 experts 对特定类型的 unsafe 内容负责
- 可以追踪每个 batch 的训练效果
- 便于调试和优化

## 实现细节

### 1. 分析阶段（demo_refactored.py）

```python
# 为每个 batch 独立计算和识别
batch_unsafe_experts = []
batch_expert_masks = []

for batch_data in batch_data_list:
    # 1. 计算这个 batch 的 risk_diff
    batch_risk_diff = calculate_risk_diff_per_batch(batch_data, num_experts_per_tok)
    
    # 2. 识别这个 batch 的 unsafe experts
    batch_unsafe = identify_unsafe_experts(batch_risk_diff, threshold=0.05, top_k=50)
    batch_unsafe_experts.append(batch_unsafe)
    
    # 3. 创建这个 batch 的 expert mask
    batch_mask = create_expert_mask(num_layers, n_experts, batch_unsafe)
    batch_expert_masks.append(batch_mask)

# 保存
sft_data = {
    "batch_unsafe_experts": batch_unsafe_experts,  # List[List[(layer, expert)]]
    "batch_expert_masks": batch_expert_masks,      # List[np.ndarray]
}
```

### 2. 训练阶段（train_batch_unsafe_experts.py）

#### 参数冻结：取并集

```python
def freeze_model_except_unsafe_experts(model, batch_unsafe_experts):
    # 取所有 batch 的 unsafe experts 的并集
    all_unsafe_experts = set()
    for batch_experts in batch_unsafe_experts:
        all_unsafe_experts.update(batch_experts)
    
    # 解冻所有出现过的 unsafe experts
    # 这样每个 batch 需要的 experts 都是可训练的
    for layer, expert in all_unsafe_experts:
        unfreeze_expert(model, layer, expert)
```

**原因**：
- Batch 0 可能需要 Expert A 和 B
- Batch 1 可能需要 Expert B 和 C
- 需要解冻 A, B, C（并集），这样两个 batch 都能训练

#### 动态 Expert Masking：根据 batch_idx

```python
class ExpertMaskingTrainer(Trainer):
    def __init__(self, batch_expert_masks, ...):
        # 保存所有 batch 的 mask
        self.batch_expert_masks = batch_expert_masks  # List of masks
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取当前 batch 的 batch_idx
        batch_idx = inputs['batch_idx'][0].item()
        
        # 使用对应的 expert mask
        current_mask = self.batch_expert_masks[batch_idx]
        
        # 应用 mask：只激活这个 batch 的 unsafe experts
        # Safe experts 的 logits 被 mask
        ...
```

**关键**：训练时根据 `batch_idx` 动态选择 mask！

## 数据流

```
┌─────────────────────────────────────────────────────┐
│ Batch 0: [8 safe + 8 unsafe (hacking)]            │
│   ↓ Calculate risk_diff for this batch            │
│   ↓ Identify unsafe_experts_0: [(L5,E12), ...]    │
│   ↓ Create expert_mask_0                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Batch 1: [8 safe + 8 unsafe (violence)]           │
│   ↓ Calculate risk_diff for this batch            │
│   ↓ Identify unsafe_experts_1: [(L3,E9), ...]     │
│   ↓ Create expert_mask_1                           │
└─────────────────────────────────────────────────────┘

         ↓
    Save all batch-specific data
         ↓
         
┌─────────────────────────────────────────────────────┐
│ Training:                                           │
│                                                     │
│ 1. Unfreeze union of all unsafe experts           │
│    (so all batches can train their experts)        │
│                                                     │
│ 2. For each training batch:                        │
│    - Read batch_idx from data                      │
│    - Use corresponding expert_mask                 │
│    - Only activate that batch's unsafe experts     │
│    - Update only those experts' parameters         │
└─────────────────────────────────────────────────────┘
```

## 示例输出

### 分析阶段

```
Step 3: Calculate Risk Difference Per Batch
==========================================
Calculating risk diff per batch: 100%|████| 10/10

Step 4: Unsafe Experts Summary
==========================================
Number of batches: 10

Per-batch unsafe experts count:
  Batch 0: 47 unsafe experts
  Batch 1: 52 unsafe experts
  Batch 2: 43 unsafe experts
  Batch 3: 50 unsafe experts
  Batch 4: 45 unsafe experts

Global top 10 unsafe experts (aggregated):
  Layer_Expert  risk_diff  a_safe_n  a_unsafe_n
0     L08_E42     0.234     0.123       0.357
1     L15_E17     0.198     0.089       0.287
2     L12_E31     0.176     0.145       0.321
...

Batch 0 top 5 unsafe experts:
   layer  expert  risk_diff
0      8      42      0.245
1     15      17      0.212
2     12      31      0.189
3      5      23      0.178
4     18       9      0.165
```

### 训练阶段

```
Loading prepared batch data...
==========================================
Loaded SFT data: 160 examples
Number of batches: 10

Per-batch unsafe experts:
  Batch 0: 47 unsafe experts
  Batch 1: 52 unsafe experts
  Batch 2: 43 unsafe experts

Freezing model parameters...
==========================================
Unfreezing union of unsafe experts from all batches:
  Total unique unsafe experts across all batches: 85

Parameter freezing complete:
  Unfrozen 85 unsafe expert MLPs (union across all batches)
  Trainable: 12,345,678 / 1,234,567,890 (1.00%)

Starting training...
==========================================
Batch size: 16
Safe ratio: 0.5 (8 safe + 8 unsafe per batch)
Number of batches: 10
Each batch has its own set of unsafe experts!

Per-batch unsafe expert counts:
  Min: 43, Max: 52, Mean: 47.8

Total unique unsafe experts (union): 85
These experts' MLPs are unfrozen for training
But only batch-specific experts are activated per batch!
==========================================

Training Batch 0: Uses expert_mask_0 (47 experts)
Training Batch 1: Uses expert_mask_1 (52 experts)
Training Batch 2: Uses expert_mask_2 (43 experts)
...
```

## 优势

### 1. 精确性

```python
# 之前：所有 batch 用同一个 mask
Batch 0 (hacking): uses global_mask (100 experts)
Batch 1 (violence): uses global_mask (100 experts)  # 可能包含不相关的 experts

# 现在：每个 batch 用自己的 mask
Batch 0 (hacking): uses mask_0 (47 hacking-related experts)  ✓
Batch 1 (violence): uses mask_1 (52 violence-related experts)  ✓
```

### 2. 效率

- 每个 batch 只训练相关的 experts
- 避免浪费计算在不相关的 experts 上
- 训练更聚焦

### 3. 灵活性

```python
# 可以单独分析每个 batch
batch_0_performance = evaluate(model, batch_0_data)
batch_1_performance = evaluate(model, batch_1_data)

# 可以针对性调整
if batch_0_performance < threshold:
    # 增加 batch 0 的训练权重
    # 或者重新识别 batch 0 的 unsafe experts
```

### 4. 可解释性

```python
# 可以回答问题：
"哪些 experts 对 hacking 类内容负责？"
→ 查看 batch_0 (hacking) 的 unsafe_experts_0

"哪些 experts 对 violence 类内容负责？"
→ 查看 batch_1 (violence) 的 unsafe_experts_1

"有哪些 experts 对多种 unsafe 内容负责？"
→ 找出现在多个 batch 的 unsafe experts
```

## 数据结构

### 保存的数据

```python
sft_data = {
    # Per-batch 信息
    "batch_unsafe_experts": [
        [(0, 5), (1, 12), ...],  # Batch 0 的 unsafe experts
        [(0, 8), (2, 3), ...],   # Batch 1 的 unsafe experts
        ...
    ],
    
    "batch_expert_masks": [
        np.array([[0, 0, 1, ...], ...]),  # Batch 0 的 mask
        np.array([[0, 1, 0, ...], ...]),  # Batch 1 的 mask
        ...
    ],
    
    # 全局信息（参考）
    "aggregated_risk_diff": pd.DataFrame(...),  # 聚合的 risk_diff
    "global_unsafe_experts": [...],              # 全局识别的 unsafe experts
}
```

### 检查数据

```python
import pandas as pd

# 加载数据
data = pd.read_pickle("sft_dataset_*.pkl")

# 查看每个 batch 的 unsafe experts
for i, experts in enumerate(data['batch_unsafe_experts'][:3]):
    print(f"Batch {i}: {len(experts)} unsafe experts")
    print(f"  Sample: {list(experts)[:5]}")

# 查看重叠
batch_0_set = set(data['batch_unsafe_experts'][0])
batch_1_set = set(data['batch_unsafe_experts'][1])
overlap = batch_0_set & batch_1_set
print(f"Overlap between batch 0 and 1: {len(overlap)} experts")
```

## 训练监控

可以添加监控来验证每个 batch 使用了正确的 mask：

```python
class BatchMonitoringTrainer(ExpertMaskingTrainer):
    def training_step(self, model, inputs):
        batch_idx = inputs['batch_idx'][0].item()
        
        # 验证使用了正确的 mask
        expected_mask = self.batch_expert_masks[batch_idx]
        assert torch.equal(self.current_expert_mask, expected_mask.to(self.args.device))
        
        print(f"Training batch_idx={batch_idx}, "
              f"using {expected_mask.sum().item():.0f} unsafe experts")
        
        return super().training_step(model, inputs)
```

## 总结

### 关键点

1. ✅ **每个 batch 独立识别 unsafe experts**
2. ✅ **保存每个 batch 的 expert mask**
3. ✅ **训练时解冻所有 batch 的 unsafe experts 的并集**
4. ✅ **根据 batch_idx 动态使用对应的 mask**
5. ✅ **更精确、更高效、更可解释**

### 使用建议

- **分析多个 batch 的差异**：了解不同类型 unsafe 内容的 expert pattern
- **监控训练过程**：确保每个 batch 使用了正确的 mask
- **评估分 batch 效果**：单独评估每类 unsafe 内容的修复效果

这个设计充分利用了 batch 内分析的优势，实现了真正的精细化控制！🎯

