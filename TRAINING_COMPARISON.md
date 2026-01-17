# Training Comparison Experiments

本文档说明如何运行选择性专家训练和全参数基线对比实验。

## 实验设置

### 1. 选择性专家训练（Selective Expert Training）

只训练识别出的 unsafe experts，其他参数冻结：

```bash
python train_batch_unsafe_experts.py \
    --mode selective \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./unsafe_expert_finetuned_batch \
    --num_epochs 3 \
    --batch_size 16 \
    --safe_ratio 0.5 \
    --learning_rate 5e-5
```

**特点：**
- ✅ 只训练 unsafe expert MLPs（union of all batch-specific unsafe experts）
- ✅ 使用 ExpertMaskingTrainer 进行 batch-specific expert masking
- ✅ 训练参数占比很小（~1-5%）
- ✅ 保存到 `./unsafe_expert_finetuned_batch/`

### 2. 全参数基线训练（Full Model Baseline）

训练所有模型参数作为对比：

```bash
python train_batch_unsafe_experts.py \
    --mode baseline \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./unsafe_expert_finetuned_batch \
    --num_epochs 3 \
    --batch_size 16 \
    --safe_ratio 0.5 \
    --learning_rate 5e-5
```

**特点：**
- ✅ 训练所有模型参数（100%）
- ✅ 使用标准 Trainer（不进行 expert masking）
- ✅ 使用相同的数据集和训练设置
- ✅ 自动保存到 `./full_model_baseline/`（或 `{output_dir}_baseline`）

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 训练模式：`selective` 或 `baseline` | `selective` |
| `--model_name` | HuggingFace 模型名称 | `allenai/OLMoE-1B-7B-0125-Instruct` |
| `--output_dir` | 输出目录 | `./unsafe_expert_finetuned_batch` |
| `--num_epochs` | 训练 epochs | `3` |
| `--batch_size` | 训练 batch size | `16` |
| `--safe_ratio` | 每个 batch 中 safe prompts 的比例 | `0.5` |
| `--learning_rate` | 学习率 | `5e-5` |

## 输出文件

### Selective Expert Training 输出

```
./unsafe_expert_finetuned_batch/
├── final_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── training_metadata.json
└── batch_expert_masks.npy
```

**training_metadata.json** 包含：
- `training_mode`: "selective_experts"
- `total_unique_unsafe_experts`: 训练的专家数量
- `per_batch_unsafe_expert_counts`: 每个 batch 的专家数量
- `batch_unsafe_experts_summary`: 前3个 batch 的专家示例

### Full Model Baseline 输出

```
./full_model_baseline/
├── final_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
└── training_metadata.json
```

**training_metadata.json** 包含：
- `training_mode`: "full_model_baseline"
- `trainable_params`: 可训练参数数量
- `trainable_ratio`: 1.0 (100%)
- `note`: 说明这是基线实验

## 对比实验流程

### Step 1: 准备数据（运行一次即可）

```bash
python demo_refactored.py
```

这会生成：
- `batch_outputs_{model}.pkl`
- `risk_diff_{model}.pkl`
- `sft_dataset_{model}.pkl`

### Step 2: 运行选择性专家训练

```bash
python train_batch_unsafe_experts.py --mode selective
```

### Step 3: 运行全参数基线训练

```bash
python train_batch_unsafe_experts.py --mode baseline
```

### Step 4: 评估和对比

使用评估脚本对比两个模型的性能：

```bash
# 评估选择性专家模型
python 3_evaluation/evaluation_refactored.py \
    --model_path ./unsafe_expert_finetuned_batch/final_model \
    --output_name selective_expert_results

# 评估基线模型
python 3_evaluation/evaluation_refactored.py \
    --model_path ./full_model_baseline/final_model \
    --output_name baseline_results
```

## 预期差异

| 方面 | Selective Expert | Full Model Baseline |
|------|-----------------|---------------------|
| 训练参数量 | ~1-5% | 100% |
| 训练速度 | 快 | 慢 |
| 内存占用 | 低 | 高 |
| 专家行为 | 针对性修改 unsafe experts | 全局修改所有参数 |
| 副作用风险 | 低（其他能力保持） | 高（可能影响其他能力） |

## 注意事项

1. **数据一致性**：两个实验使用完全相同的数据集和 batch 结构
2. **超参数一致性**：learning rate、batch size 等保持一致
3. **评估公平性**：使用相同的评估数据集和指标
4. **模型名称**：baseline 会自动修改输出目录名称以避免覆盖

## 疑难解答

### Q: 如何确认使用了正确的训练模式？

A: 查看训练开始时的日志输出：
- Selective: 会显示 "SELECTIVE EXPERT training" 和 unfrozen expert 数量
- Baseline: 会显示 "BASELINE EXPERIMENT: Training ALL model parameters"

### Q: 两个模型可以同时训练吗？

A: 可以，它们使用不同的输出目录。建议使用不同的 GPU：

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python train_batch_unsafe_experts.py --mode selective

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python train_batch_unsafe_experts.py --mode baseline
```

### Q: 如何验证 baseline 确实训练了所有参数？

A: 查看 `training_metadata.json` 中的 `trainable_ratio`，应该为 1.0。

