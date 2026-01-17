# Gradient-Based Unsafe Expert Identification

## 概述

这是一个新的基于梯度幅度的不安全专家识别方法,相比原有的routing weights方法更加直接和高效。

## 方法对比

### 原有方法 (`train_alternating_optimization.py`)
1. 同时运行安全和不安全的prompt
2. 通过比较routing weights找出不安全的专家
3. SFT这些不安全的专家
4. 调节路由以匹配安全的routing pattern

### 新方法 (`train_gradient_based_expert_identification.py`)
1. 直接用不安全的prompt + 安全的回复进行SFT
2. 在backward时计算每个专家的gradient magnitude
3. 对gradient进行ranking,找出top-k个gradient变化最大的专家
4. 这些就是最不安全的专家(因为它们需要改变最多才能产生安全回复)
5. 去掉其他专家的gradient,只更新这些不安全专家
6. 记录识别出的不安全专家
7. (可选)然后调节路由以匹配安全的routing pattern

## 优势

1. **更直接的识别**: Gradient magnitude直接衡量哪些专家需要改变最多
2. **更高效**: 不需要同时运行安全和不安全的prompt来进行对比
3. **理论基础更强**: Gradient表示参数需要改变的方向和程度,更能反映专家在不安全行为中的作用

## 使用方法

### 基本用法

```bash
python train_gradient_based_expert_identification.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset_path /path/to/your/dataset.json \
    --output_dir ./gradient_based_results \
    --num_rounds 3 \
    --top_k_experts 100 \
    --batch_size 8 \
    --expert_lr 5e-5 \
    --router_lr 1e-4
```

### 只训练专家(不训练router)

```bash
python train_gradient_based_expert_identification.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset_path /path/to/your/dataset.json \
    --output_dir ./gradient_based_expert_only \
    --num_rounds 3 \
    --top_k_experts 100 \
    --expert_lr 5e-5 \
    --skip_router_training
```

### Qwen模型示例

```bash
python train_gradient_based_expert_identification.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --dataset_path /home/stufs1/jiachliang/FlipAttack/reproduce_result/FlipAttack-FWO-CoT-LangGPT-Few-shot-Qwen3-30B-A3B-advbench-0_519-final.json \
    --output_dir ./qwen_gradient_based \
    --num_rounds 3 \
    --top_k_experts 100 \
    --batch_size 4 \
    --expert_lr 5e-5 \
    --max_samples 128
```

## 参数说明

- `--model_name`: HuggingFace模型名称
- `--dataset_path`: 数据集JSON文件路径
- `--output_dir`: 输出目录(保存checkpoints)
- `--num_rounds`: 训练轮数
- `--top_k_experts`: 每轮识别并更新的top-k不安全专家数量
- `--batch_size`: 训练batch size
- `--expert_lr`: 专家训练的学习率
- `--router_lr`: Router训练的学习率
- `--max_samples`: 使用的最大样本数
- `--skip_router_training`: 跳过router训练(只训练专家)
- `--kl_type`: KL散度类型(forward/reverse/symmetric)

## 输出文件

训练完成后会生成以下文件:

1. `round_1/`, `round_2/`, ...: 每轮训练后的checkpoint
2. `final_model/`: 最终训练好的模型
3. `training_metadata.json`: 训练元数据和日志
4. `expert_gradients.pkl`: 每轮识别出的专家及其gradient信息

## 核心算法

### 1. Gradient计算

```python
# 对于每个专家,计算其所有参数的平均gradient magnitude
for expert in experts:
    total_grad = sum(|param.grad|) for all params
    avg_grad = total_grad / num_params
```

### 2. Top-K选择

```python
# 按gradient magnitude排序,选择top-k
sorted_experts = sort(experts, by=avg_grad, descending=True)
unsafe_experts = sorted_experts[:k]
```

### 3. Gradient Masking

```python
# 只保留unsafe experts的gradient,其他的清零
for expert in all_experts:
    if expert not in unsafe_experts:
        expert.grad.zero_()
```

### 4. 更新参数

```python
# 只有unsafe experts的参数会被更新
optimizer.step()
```

## 实验建议

### Top-K值的选择

- **小模型** (如OLMoE-1B-7B): `top_k_experts=50-100`
- **大模型** (如Qwen3-30B): `top_k_experts=100-200`

### 学习率设置

- **Expert LR**: `5e-5` (较小,避免过度改变专家)
- **Router LR**: `1e-4` (较大,鼓励router适应)

### 训练轮数

- 建议从 `num_rounds=3` 开始
- 如果效果不够好,可以增加到5-10轮

## 与原方法的对比实验

可以运行对比实验:

```bash
# 新方法 (gradient-based)
python train_gradient_based_expert_identification.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./gradient_based_results

# 原方法 (routing-based)
python train_alternating_optimization.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./routing_based_results
```

然后比较两种方法在相同评估集上的表现。

## 注意事项

1. **内存使用**: Gradient计算需要保存所有参数的梯度,确保有足够GPU内存
2. **Batch Size**: 如果显存不足,可以减小batch_size
3. **数据集格式**: 确保数据集包含`all_prompt`和`goal`字段
4. **模型架构**: 目前支持OLMoE和Mixtral架构,其他MoE架构可能需要调整代码

## 疑难解答

### Q: 显存不足怎么办?
A: 减小`batch_size`或使用更小的`max_samples`

### Q: 如何查看识别出的不安全专家?
A: 查看`training_metadata.json`中的`identified_experts_per_round`字段

### Q: 可以只运行专家训练吗?
A: 可以,使用`--skip_router_training`参数

### Q: Gradient accumulation是什么?
A: 在所有batch上累积gradient后再选择top-k,这样更稳定(默认启用)
