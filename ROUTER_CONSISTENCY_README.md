# Router Consistency Loss - 使用说明

## 概述

Router Consistency Loss 是一个可选的训练优化，用于强制 batch 内的 safe 和 unsafe 样本使用一致的 router 分布。这是一个消融实验功能，可以单独启用或与 unsafe experts finetuning 结合使用。

## 🚀 快速开始

### 核心思想
- **问题**: 原方法只 finetune unsafe experts，可能影响模型整体行为
- **解决**: 添加 router consistency loss，让 safe/unsafe 样本激活相同的专家
- **优势**: 理论上可保持 safe 任务性能，同时提升 unsafe 拒绝能力

### 三个关键参数
1. `--no_finetune_unsafe_experts`: 禁用 expert finetuning（只训练 router）
2. `--use_router_consistency_loss`: 启用 router consistency loss
3. `--router_consistency_weight`: 控制 loss 权重（默认 0.1）

### 最简单的用法

```bash
# 原有方法（只 finetune experts）
python train_batch_unsafe_experts.py --mode selective

# 新方法1: 只优化 router
python train_batch_unsafe_experts.py --mode selective \
    --no_finetune_unsafe_experts --use_router_consistency_loss

# 新方法2: 两者结合（推荐）
python train_batch_unsafe_experts.py --mode selective \
    --use_router_consistency_loss
```

## 设计原理

### 损失计算位置
在 **router logits** 上计算 KL 散度（推荐方案）：
- ✅ 完全可微，梯度可以流向 router
- ✅ 理论合理：KL 散度天然度量概率分布差异
- ✅ 全局一致性：约束所有专家的选择概率

### KL 散度形式
对每一层 l，计算：
```python
safe_avg_probs = softmax(safe_router_logits[l]).mean(dim=0)    # [n_experts]
unsafe_avg_probs = softmax(unsafe_router_logits[l]).mean(dim=0) # [n_experts]

# 对称 KL 散度（默认）
kl_loss = (KL(safe || unsafe) + KL(unsafe || safe)) / 2
```

使用平均分布的原因：
- Batch 内 safe 和 unsafe 数量可能不同，避免一对一配对复杂性
- 平均分布更稳定，代表该 batch 中样本的"典型" router 行为

### KL 散度类型
1. **Forward KL**: `KL(safe || unsafe)` - 让 unsafe 接近 safe
2. **Reverse KL**: `KL(unsafe || safe)` - 让 safe 接近 unsafe
3. **Symmetric**: `(KL(safe || unsafe) + KL(unsafe || safe)) / 2` - 双向约束（**推荐**）

## 使用方法

### 基本用法

**1. 仅使用 Router Consistency Loss（消融实验 - Router Only）**
```bash
python train_batch_unsafe_experts.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --mode selective \
    --no_finetune_unsafe_experts \
    --use_router_consistency_loss \
    --router_consistency_weight 0.1 \
    --router_consistency_type symmetric \
    --router_consistency_layers all \
    --output_dir ./output_router_only
```
**说明**: 只训练 router 参数，experts 保持冻结

**2. 仅 Finetune Unsafe Experts（消融实验 - Expert Only）**
```bash
python train_batch_unsafe_experts.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --mode selective \
    --output_dir ./output_expert_only
```
**说明**: 这是原有的方法，只 finetune unsafe experts，不使用 router consistency loss

**3. Router Consistency + Unsafe Experts Finetuning（组合优化）**
```bash
python train_batch_unsafe_experts.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --mode selective \
    --use_router_consistency_loss \
    --router_consistency_weight 0.1 \
    --output_dir ./output_combined
```
**说明**: 同时训练 unsafe experts 和 router，两个优化目标结合

**4. Baseline（冻结模型）**
```bash
python train_batch_unsafe_experts.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --mode selective \
    --no_finetune_unsafe_experts \
    --output_dir ./output_frozen_baseline
```
**说明**: 所有参数冻结，作为对照（几乎没有实际意义）

### 超参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--finetune_unsafe_experts` | flag | True | 是否 finetune unsafe expert MLPs（默认启用） |
| `--no_finetune_unsafe_experts` | flag | - | 禁用 unsafe expert finetuning（用于 router-only 消融） |
| `--use_router_consistency_loss` | flag | False | 是否启用 router 一致性损失 |
| `--router_consistency_weight` | float | 0.1 | Router 一致性损失的权重 |
| `--router_consistency_type` | str | symmetric | KL 散度类型：forward/reverse/symmetric |
| `--router_consistency_layers` | str | all | 计算哪些层：all 或 unsafe_only |

**重要说明**:
- 默认情况下 `finetune_unsafe_experts=True`，保持原有行为
- 使用 `--no_finetune_unsafe_experts` 来禁用 expert finetuning
- 如果只用 router consistency loss（`--no_finetune_unsafe_experts --use_router_consistency_loss`），只有 router 参数会被训练

### 超参数调优建议

1. **`router_consistency_weight`**
   - 建议范围：0.01 ~ 1.0
   - 起始值：0.1
   - 如果 router loss 太大导致训练不稳定，降低此值
   - 如果想要更强的 router 一致性约束，增大此值

2. **`router_consistency_type`**
   - 推荐：`symmetric`（双向约束，最稳定）
   - 如果想明确让 unsafe 学习 safe 的 router 行为：`forward`
   - 如果想让 safe 也适应 unsafe：`reverse`（不太推荐）

3. **`router_consistency_layers`**
   - 推荐：`all`（全面约束）
   - 如果计算资源有限或想减少约束：`unsafe_only`（目前未实现，需要传入 unsafe experts 信息）

## 实验设计建议

### 消融实验矩阵

| 实验 | Finetune Experts | Router Consistency | 可训练参数 | 目的 |
|------|-----------------|-------------------|-----------|------|
| Baseline | ❌ | ❌ | 无 | 冻结模型对照（可选） |
| Expert Only | ✅ | ❌ | Unsafe Experts | 仅 finetune experts 的效果 |
| Router Only | ❌ | ✅ | Routers | 仅 router 一致性的效果 |
| Combined | ✅ | ✅ | Experts + Routers | 两者结合的协同效应 |

### 完整运行示例

```bash
MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"

# Ablation 1: 仅 Finetune Unsafe Experts（原有方法，推荐作为 baseline）
python train_batch_unsafe_experts.py \
    --model_name $MODEL_NAME \
    --mode selective \
    --output_dir ./exp_expert_only \
    --num_epochs 3 \
    --batch_size 16

# Ablation 2: 仅 Router Consistency Loss
python train_batch_unsafe_experts.py \
    --model_name $MODEL_NAME \
    --mode selective \
    --no_finetune_unsafe_experts \
    --use_router_consistency_loss \
    --router_consistency_weight 0.1 \
    --output_dir ./exp_router_only \
    --num_epochs 3 \
    --batch_size 16

# Ablation 3: 两者结合（推荐）
python train_batch_unsafe_experts.py \
    --model_name $MODEL_NAME \
    --mode selective \
    --use_router_consistency_loss \
    --router_consistency_weight 0.1 \
    --output_dir ./exp_combined \
    --num_epochs 3 \
    --batch_size 16

# 不同权重的敏感性分析
for weight in 0.01 0.05 0.1 0.5 1.0; do
    python train_batch_unsafe_experts.py \
        --model_name $MODEL_NAME \
        --mode selective \
        --use_router_consistency_loss \
        --router_consistency_weight $weight \
        --output_dir ./exp_weight_$weight \
        --num_epochs 3 \
        --batch_size 16
done

# 不同 KL 类型的对比
for kl_type in forward reverse symmetric; do
    python train_batch_unsafe_experts.py \
        --model_name $MODEL_NAME \
        --mode selective \
        --no_finetune_unsafe_experts \
        --use_router_consistency_loss \
        --router_consistency_type $kl_type \
        --output_dir ./exp_kl_$kl_type \
        --num_epochs 3 \
        --batch_size 16
done
```

## 实现细节

### 架构支持
- ✅ OLMoE: `model.layers[i].mlp.gate`
- ✅ Mixtral: `model.layers[i].mlp.block_sparse_moe.gate`
- 其他架构需要修改 `_register_router_hooks` 方法

### 技术特点
1. **Forward Hooks**: 自动捕获每层的 router logits
2. **梯度流**: 不使用 `detach()`，保持梯度可回传
3. **Batch-aware**: 通过 `label_ids` 字段区分 safe/unsafe 样本
4. **自动平均**: 跨序列位置和层数平均，稳定训练

### 输出和日志
- 每 100 步打印一次详细损失：
  ```
  Step 100: CE Loss: 2.3456, Router Loss: 0.0234, Total Loss: 2.3690
  ```
- 训练元数据保存在 `training_metadata.json`，包含所有 router consistency 配置

## 预期效果

### 优势
✅ Router 学习到 safe 和 unsafe 样本应激活相似专家  
✅ 有助于 unsafe experts 专门处理 unsafe 内容，而不改变 router 行为  
✅ 理论上可保持 safe 任务性能，同时提升 unsafe 任务拒绝能力

### 潜在风险
⚠️ 如果 safe 和 unsafe 本质上需要不同专家，约束可能降低表达能力  
⚠️ 权重设置不当可能导致训练不稳定（KL 散度可能很大）  
⚠️ 只优化 router 一致性而不优化 experts，效果可能有限

## 调试和问题排查

### 检查是否正常工作
1. 运行测试脚本：
   ```bash
   python3 test_router_consistency.py
   ```
   应该看到所有测试通过。

2. 检查训练日志：
   - 启动时应看到 "Router Consistency Loss Configuration"
   - 应看到 "Registered N router hooks"
   - 每 100 步应有详细损失打印

3. 检查梯度：
   - Router loss 应该 > 0
   - 如果始终为 0，可能 hooks 没有正确捕获 logits

### 常见问题

**Q: Router Loss 始终为 0？**  
A: 检查 `label_ids` 是否正确传递，确保 batch 中同时有 safe 和 unsafe 样本。

**Q: 训练不稳定/loss 爆炸？**  
A: 降低 `--router_consistency_weight`，从 0.01 开始尝试。

**Q: 支持其他 MoE 架构？**  
A: 修改 `ExpertMaskingTrainer._register_router_hooks` 方法，添加对应架构的 router 位置。

**Q: 如何只计算 unsafe expert 所在层的 loss？**  
A: 当前 `--router_consistency_layers unsafe_only` 未实现，需要额外传入 unsafe experts 信息并在 `_compute_router_consistency_loss` 中过滤层。

## 引用和参考

这个实现基于以下理论：
- KL 散度作为分布间距离度量
- MoE 模型的 router 一致性约束
- Safe/Unsafe 样本的专家选择模式

## 更新日志

- **2025-10-31**: 初始实现
  - 支持三种 KL 散度类型
  - 自动 forward hooks 捕获 router logits
  - 完整的命令行接口和消融实验支持

