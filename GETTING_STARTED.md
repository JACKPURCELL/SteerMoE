# Getting Started with Gradient-Based Expert Identification

## 快速开始 (3步)

### 1. 测试环境
```bash
python test_gradient_method.py
```
这会检查:
- 必要的Python包是否已安装
- 数据集是否可以访问
- GPU是否可用
- 代码语法是否正确

### 2. 运行快速示例
```bash
./quick_start_gradient_method.sh
```
这会自动:
- 训练一个OLMoE模型(3轮)
- 识别top-100不安全专家
- 保存所有checkpoints
- 生成可视化分析

### 3. 查看结果
```bash
ls -la quick_start_results/
cat quick_start_results/training_metadata.json
```

## 完整工作流程

### 步骤1: 准备数据
确保你的数据集是JSON格式,包含以下字段:
```json
[
  {
    "goal": "原始不安全问题",
    "all_prompt": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "jailbreak prompt"}
    ],
    "judge_success_gpt4": 1
  }
]
```

### 步骤2: 训练模型

#### 选项A: 使用新方法(Gradient-Based)
```bash
python train_gradient_based_expert_identification.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset_path /path/to/dataset.json \
    --output_dir ./my_trained_model \
    --num_rounds 3 \
    --top_k_experts 100 \
    --batch_size 8 \
    --expert_lr 5e-5 \
    --router_lr 1e-4
```

#### 选项B: 使用原方法(Routing-Based)
```bash
python train_alternating_optimization.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./original_method \
    --num_rounds 3 \
    --epochs_per_round 1
```

注意: 原方法需要先运行 `demo_refactored.py` 生成routing数据。

### 步骤3: 分析结果
```bash
python analyze_identified_experts.py \
    --gradient_dir ./my_trained_model \
    --output_dir ./analysis_results
```

### 步骤4: 评估模型
```bash
cd 3_evaluation
python evaluation_example.py \
    --model_path ../my_trained_model/final_model \
    --output_file results.json
```

## 高级用法

### 只训练专家(不训练router)
```bash
python train_gradient_based_expert_identification.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --dataset_path /path/to/dataset.json \
    --output_dir ./expert_only \
    --skip_router_training
```

### 调整识别的专家数量
```bash
# 识别top-50专家(少量)
python train_gradient_based_expert_identification.py \
    --top_k_experts 50 \
    ...

# 识别top-200专家(大量)
python train_gradient_based_expert_identification.py \
    --top_k_experts 200 \
    ...
```

### 使用不同的MoE模型

#### Qwen3-MoE
```bash
python train_gradient_based_expert_identification.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --batch_size 4 \
    --top_k_experts 150
```

#### Mixtral
```bash
python train_gradient_based_expert_identification.py \
    --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch_size 4 \
    --top_k_experts 32
```

### 多GPU训练
训练脚本会自动使用所有可用的GPU (通过 `device_map="auto"`)

如果想指定GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_gradient_based_expert_identification.py ...
```

## 对比实验

运行完整的对比实验(4种配置):
```bash
./run_comparison_experiments.sh
```

这会生成:
1. `comparison_gradient_based/` - 新方法(完整)
2. `comparison_routing_based/` - 原方法(完整)
3. `comparison_gradient_expert_only/` - 新方法(仅专家)
4. `comparison_routing_expert_only/` - 原方法(仅专家)

## 输出文件说明

训练完成后的目录结构:
```
output_dir/
├── round_1/                    # Round 1模型checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── round_2/                    # Round 2模型checkpoint
├── round_3/                    # Round 3模型checkpoint
├── final_model/                # 最终训练好的模型
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── training_metadata.json      # 训练日志和专家信息
└── expert_gradients.pkl        # 详细的gradient数据
```

### training_metadata.json 内容
```json
{
  "model_name": "allenai/OLMoE-1B-7B-0125-Instruct",
  "training_strategy": "gradient_based_identification",
  "num_rounds": 3,
  "top_k_experts": 100,
  "identified_experts_per_round": {
    "1": {
      "experts": [[0, 15], [1, 23], ...],
      "gradients": {
        "L0_E15": 0.001234,
        "L1_E23": 0.001156,
        ...
      }
    },
    "2": {...},
    "3": {...}
  },
  "training_logs": [
    {
      "round": 1,
      "step": "expert",
      "loss": 2.345,
      "num_identified_experts": 100
    },
    ...
  ]
}
```

## 常见问题

### Q1: 显存不足 (CUDA out of memory)
**解决方法:**
- 减小 `--batch_size` (从8→4→2)
- 减小 `--max_samples` (从128→64→32)
- 使用更小的模型进行测试

### Q2: 数据集格式错误
**检查:**
- 确保数据是JSON格式
- 包含 `goal` 和 `all_prompt` 字段
- `all_prompt` 是消息列表格式

### Q3: 训练太慢
**优化方法:**
- 减少 `--num_rounds`
- 减少 `--max_samples`
- 增大 `--batch_size` (如果显存够)
- 使用更多GPU

### Q4: 如何查看识别出的专家?
```python
import json

with open('output_dir/training_metadata.json', 'r') as f:
    metadata = json.load(f)

# 查看Round 3识别的专家
round_3_experts = metadata['identified_experts_per_round']['3']['experts']
print(f"Round 3识别了 {len(round_3_experts)} 个专家:")
for layer, expert in round_3_experts[:10]:
    print(f"  Layer {layer}, Expert {expert}")
```

### Q5: 如何评估训练好的模型?
```bash
# 方法1: 使用evaluation脚本
cd 3_evaluation
python evaluation_example.py --model_path ../output_dir/final_model

# 方法2: 手动测试
python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> model = AutoModelForCausalLM.from_pretrained("output_dir/final_model")
>>> tokenizer = AutoTokenizer.from_pretrained("output_dir/final_model")
>>> # 测试模型...
```

## 性能建议

### Top-K值选择
- **小模型** (OLMoE-1B-7B): 50-100
- **中等模型** (Mixtral-8x7B): 50-100
- **大模型** (Qwen3-30B): 100-200

### 学习率设置
- **Expert LR**: 5e-5 (标准值)
  - 太大: 模型可能forget原有能力
  - 太小: 训练效果不明显
- **Router LR**: 1e-4 (标准值)
  - 通常比expert LR大一些

### 训练轮数
- 开始: 3轮
- 如果效果不够: 5-10轮
- 注意: 太多轮可能导致过拟合

## 文件说明

### 核心文件
- `train_gradient_based_expert_identification.py` - 主训练脚本(新方法)
- `train_alternating_optimization.py` - 原方法训练脚本
- `demo_refactored.py` - 原方法的routing数据生成

### 辅助文件
- `test_gradient_method.py` - 测试脚本
- `analyze_identified_experts.py` - 分析工具
- `quick_start_gradient_method.sh` - 快速开始脚本
- `run_comparison_experiments.sh` - 对比实验脚本

### 文档
- `GETTING_STARTED.md` - 本文件
- `GRADIENT_BASED_METHOD_README.md` - 方法详细说明
- `IMPLEMENTATION_SUMMARY.md` - 实现总结

## 下一步

1. **运行测试**: `python test_gradient_method.py`
2. **快速开始**: `./quick_start_gradient_method.sh`
3. **查看结果**: 检查 `quick_start_results/training_metadata.json`
4. **评估模型**: 在你的测试集上评估
5. **对比实验**: 运行 `./run_comparison_experiments.sh` 对比两种方法

## 获取帮助

查看命令行帮助:
```bash
python train_gradient_based_expert_identification.py --help
```

查看所有参数:
```bash
python train_gradient_based_expert_identification.py --help | grep "^  --"
```




