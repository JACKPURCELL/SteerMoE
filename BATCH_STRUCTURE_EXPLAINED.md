# Batch Structure - 批次结构详解

## 核心改进：按 Batch 保存数据集

### 问题背景

**之前的问题**：虽然我们在分析时使用了 batch 内混合（8 safe + 8 unsafe），但训练时数据被展平并重新打乱，导致训练和分析不一致。

```python
# 分析时：Batch 内混合
Batch 0: [safe_0, safe_1, ..., safe_7, unsafe_0, ..., unsafe_7]  # 8+8=16
Batch 1: [safe_8, safe_9, ..., safe_15, unsafe_8, ..., unsafe_15]

# 之前训练时：被打乱
Training Batch: [safe_0, unsafe_12, safe_25, ...]  # 随机组合，比例不固定
```

### 解决方案：保持 Batch 结构

现在数据集按照分析时的 batch 结构保存，训练时使用 `BatchPreservingSampler` 维持这个结构。

```python
# 分析时：Batch 内混合
Batch 0: [safe_0, ..., safe_7, unsafe_0, ..., unsafe_7]

# 训练时：保持相同结构
Training Batch 0: [safe_0, ..., safe_7, unsafe_0, ..., unsafe_7]  # 完全相同！
```

## 实现细节

### 1. 数据保存：按 Batch 组织

**函数**: `create_batch_structured_dataset()`

```python
def create_batch_structured_dataset(sft_data, tokenizer, batch_size=16, safe_ratio=0.5):
    """
    返回 List[List[Dict]]
    外层 List: 每个元素是一个 batch
    内层 List: batch 内的样本，前 8 个 safe，后 8 个 unsafe
    """
    all_batches = []
    
    for batch_data in sft_data['batch_data_list']:
        batch_examples = []
        
        # 先添加 safe samples
        for idx in batch_data['safe_indices']:
            batch_examples.append({
                "messages": [...],
                "label": "safe",
                "batch_idx": batch_data['batch_idx'],  # 记录 batch ID
                "in_batch_idx": len(batch_examples)     # 记录在 batch 内的位置
            })
        
        # 再添加 unsafe samples
        for idx in batch_data['unsafe_indices']:
            batch_examples.append({
                "messages": [...],
                "label": "unsafe",
                "batch_idx": batch_data['batch_idx'],
                "in_batch_idx": len(batch_examples)
            })
        
        all_batches.append(batch_examples)  # 保存整个 batch
    
    return all_batches
```

**关键点**：
- 每个样本都有 `batch_idx` 字段，标识它属于哪个 batch
- 每个样本都有 `in_batch_idx` 字段，标识它在 batch 内的位置
- Batch 内的顺序固定：先 safe，后 unsafe

### 2. 数据加载：转换为 HuggingFace Dataset

**函数**: `create_batch_aware_dataset()`

```python
def create_batch_aware_dataset(batches, tokenizer, max_length=512):
    """
    将 batch-structured 数据转换为 HuggingFace Dataset
    但保留 batch_idx 信息
    """
    # 展平但保留 batch 信息
    all_examples = []
    for batch in batches:
        all_examples.extend(batch)
    
    # 创建 Dataset，包含 batch_idx
    dataset_dict = {
        "messages": [ex["messages"] for ex in all_examples],
        "batch_idx": [ex["batch_idx"] for ex in all_examples],  # 保留！
        "label": [ex["label"] for ex in all_examples],
    }
    
    return Dataset.from_dict(dataset_dict)
```

**关键点**：
- 数据虽然展平成一维列表，但每个样本保留了 `batch_idx`
- Tokenize 后，`batch_idx` 字段仍然保留在 dataset 中

### 3. 训练采样：BatchPreservingSampler

**类**: `BatchPreservingSampler`

```python
class BatchPreservingSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle_batches=True):
        # 按 batch_idx 分组
        self.batch_groups = {}
        for idx, item in enumerate(dataset):
            batch_idx = item['batch_idx']
            if batch_idx not in self.batch_groups:
                self.batch_groups[batch_idx] = []
            self.batch_groups[batch_idx].append(idx)
        
        # batch_groups[0] = [0, 1, 2, ..., 15]  # Batch 0 的样本索引
        # batch_groups[1] = [16, 17, 18, ..., 31]  # Batch 1 的样本索引
    
    def __iter__(self):
        # 可以打乱 batch 的顺序
        batch_ids = list(self.batch_groups.keys())
        if self.shuffle_batches:
            random.shuffle(batch_ids)
        
        # 但 batch 内的样本顺序保持不变
        for batch_id in batch_ids:
            batch_indices = self.batch_groups[batch_id]
            yield from batch_indices  # 按原顺序输出
```

**关键点**：
- 可以打乱 batch 的顺序（epoch 之间变化）
- 但 batch 内的样本顺序固定（8 safe + 8 unsafe）
- 确保每个 training batch 都有正确的 safe/unsafe 比例

### 4. 集成到 Trainer

```python
# 创建 Trainer
trainer = ExpertMaskingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 包含 batch_idx 的 dataset
    tokenizer=tokenizer,
)

# 替换 DataLoader，使用自定义 Sampler
def get_batch_preserving_dataloader():
    dataloader = original_get_train_dataloader()
    return DataLoader(
        trainer.train_dataset,
        batch_size=batch_size,  # 16
        sampler=BatchPreservingSampler(  # 使用自定义 sampler！
            trainer.train_dataset, 
            batch_size,
            shuffle_batches=True
        ),
        collate_fn=dataloader.collate_fn,
    )

trainer.get_train_dataloader = get_batch_preserving_dataloader
```

## 完整流程示例

### 分析阶段（demo_refactored.py）

```python
# Step 1: 处理数据，生成混合 batch
batch_data_list = process_mixed_batches(
    llm, tokenizer, 
    safe_prompts, unsafe_prompts,
    batch_size=16, safe_ratio=0.5
)

# batch_data_list[0] = {
#     "batch_idx": 0,
#     "safe_indices": [0, 1, 2, ..., 7],
#     "unsafe_indices": [8, 9, 10, ..., 15],
#     "safe_routings": [...],
#     "unsafe_routings": [...]
# }

# Step 2: 在每个 batch 内计算 risk_diff
for batch_data in batch_data_list:
    batch_risk_diff = calculate_risk_diff_per_batch(batch_data, num_experts_per_tok)

# Step 3: 聚合所有 batch 的结果
risk_diff_df = aggregate_batch_risk_diffs(batch_risk_diffs)

# Step 4: 保存（包含完整的 batch 结构）
sft_data = {
    "batch_data_list": batch_data_list,  # 保留完整 batch 结构！
    "unsafe_experts": unsafe_experts,
    "expert_mask": expert_mask,
}
pd.to_pickle(sft_data, "sft_dataset_*.pkl")
```

### 训练阶段（train_batch_unsafe_experts.py）

```python
# Step 1: 加载数据（包含 batch 结构）
sft_data = pd.read_pickle("sft_dataset_*.pkl")
batch_data_list = sft_data['batch_data_list']

# Step 2: 创建 batch-structured dataset
batches = create_batch_structured_dataset(sft_data, tokenizer)
# batches[0] = [
#     {"messages": ..., "label": "safe", "batch_idx": 0, "in_batch_idx": 0},
#     ...
#     {"messages": ..., "label": "safe", "batch_idx": 0, "in_batch_idx": 7},
#     {"messages": ..., "label": "unsafe", "batch_idx": 0, "in_batch_idx": 8},
#     ...
#     {"messages": ..., "label": "unsafe", "batch_idx": 0, "in_batch_idx": 15},
# ]

# Step 3: 转换为 HuggingFace Dataset（保留 batch_idx）
train_dataset = create_batch_aware_dataset(batches, tokenizer)

# Step 4: 使用 BatchPreservingSampler 训练
trainer = ExpertMaskingTrainer(...)
trainer.get_train_dataloader = get_batch_preserving_dataloader  # 关键！
trainer.train()
```

## 验证 Batch 结构

### 检查数据集

```python
import pandas as pd

# 加载数据
sft_data = pd.read_pickle("sft_dataset_allenai--OLMoE-1B-7B-0125-Instruct.pkl")

# 查看 batch 数量
print(f"Number of batches: {len(sft_data['batch_data_list'])}")

# 查看第一个 batch
batch_0 = sft_data['batch_data_list'][0]
print(f"Batch 0 safe indices: {batch_0['safe_indices']}")
print(f"Batch 0 unsafe indices: {batch_0['unsafe_indices']}")
print(f"Total samples in batch 0: {len(batch_0['batch_outputs'])}")
```

### 检查训练时的 batch

在训练脚本中添加：

```python
# 在训练前验证
sample_batch_indices = [i for i, item in enumerate(train_dataset) if item['batch_idx'] == 0]
print(f"Samples in batch 0: {len(sample_batch_indices)}")
print(f"Expected: {batch_size}")

# 验证 safe/unsafe 分布
labels = [train_dataset[i]['label_str'] for i in sample_batch_indices]
print(f"Safe samples: {labels.count('safe')}")
print(f"Unsafe samples: {labels.count('unsafe')}")
```

### 在训练过程中监控

```python
class BatchMonitoringTrainer(ExpertMaskingTrainer):
    def training_step(self, model, inputs):
        # 检查当前 batch 的 batch_idx 分布
        if 'batch_idx' in inputs:
            batch_ids = inputs['batch_idx'].cpu().numpy()
            unique_batches = np.unique(batch_ids)
            print(f"Current training batch contains samples from batches: {unique_batches}")
            # 应该只有一个 unique batch_idx
            assert len(unique_batches) == 1, "Batch structure broken!"
        
        return super().training_step(model, inputs)
```

## 优势总结

### ✅ 完全一致性

```
分析：Batch 0 = [8 safe + 8 unsafe]
训练：Batch 0 = [8 safe + 8 unsafe]  ← 完全相同！
```

### ✅ 可重现性

```
# 相同的 batch 结构
# 相同的 safe/unsafe 比例
# 相同的 expert activation pattern
→ 训练结果更可重现
```

### ✅ 可控性

```python
# 精确控制每个 batch 的组成
Batch 0: 8 safe (easy) + 8 unsafe (hard)
Batch 1: 8 safe (medium) + 8 unsafe (extreme)
...
```

### ✅ 可追溯性

```python
# 训练时可以追溯到分析时的数据
training_sample.batch_idx = 5
→ 对应 analysis batch 5
→ 可以查看该 batch 的 risk_diff
→ 可以分析为什么这个 batch 效果好/差
```

## 与之前实现的对比

| 方面 | 之前（展平） | 现在（保持结构） |
|------|------------|----------------|
| **数据组织** | 展平成一维列表 | 按 batch 组织 |
| **训练时 batch** | 随机重组 | 保持原始结构 |
| **Safe/Unsafe 比例** | 每个 batch 不固定 | 每个 batch 固定 50/50 |
| **分析与训练一致性** | ❌ 不一致 | ✅ 完全一致 |
| **可追溯性** | ❌ 难以追溯 | ✅ 完全可追溯 |
| **实验重现性** | ⚠️ 较差 | ✅ 优秀 |

## 技术要点

### 1. 保留 batch_idx 是关键

```python
# 在每个阶段都保留 batch_idx
create_batch_structured_dataset  → 添加 batch_idx
create_batch_aware_dataset       → 保留 batch_idx (在 tokenize 中)
BatchPreservingSampler           → 使用 batch_idx 分组
```

### 2. 不在 batch 内打乱

```python
# ✅ 正确：可以打乱 batch 顺序
batch_order = [5, 2, 8, 1, ...]  # random order

# ✅ 正确：batch 内顺序不变
Batch 5: [safe_0, ..., safe_7, unsafe_0, ..., unsafe_7]  # 固定顺序

# ❌ 错误：打乱 batch 内顺序
Batch 5: [unsafe_3, safe_1, unsafe_0, ...]  # 这样会破坏结构
```

### 3. DataLoader 配置

```python
TrainingArguments(
    per_device_train_batch_size=16,  # 必须与分析时相同
    dataloader_drop_last=False,      # 不丢弃不完整 batch
    group_by_length=False,           # 不按长度分组
    # shuffle 通过 sampler 控制
)
```

## 总结

通过按 batch 保存数据集和使用 `BatchPreservingSampler`，我们实现了：

1. **分析与训练完全一致**：相同的 batch 结构，相同的 safe/unsafe 比例
2. **更可靠的实验结果**：减少了随机性，提高了可重现性
3. **更好的可追溯性**：可以精确追踪每个训练样本到分析阶段
4. **更容易调试**：可以针对特定 batch 进行分析和优化

这是实现真正的 batch 内训练的关键！🎯

