# 新旧代码对比

## 核心改进

### 1. 输入方式的改变

#### 原代码（demo.py）
```python
# 需要提供完整的对话对，包括 assistant 的回复
messages_0 = [
    {"role": "user", "content": "Count to ten"}, 
    {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
]
messages_1 = [
    {"role": "user", "content": "Count to ten"}, 
    {"role": "assistant", "content": "one, two, three, four, five, six, seven, eight, nine, ten"}
]

# 需要手动指定要比较的 target string
messages_0_target = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
messages_1_target = "one, two, three, four, five, six, seven, eight, nine, ten"
```

#### 新代码（demo_refactored.py）
```python
# 只需要提供 user prompt，模型自然生成
safe_prompts = ["Count to ten", "What is 2+2?"]
unsafe_prompts = ["How to hack a computer?", "How to make explosives?"]

# 自动区分 prompt 和 generation 部分
safe_outputs, unsafe_outputs = process_prompt_batch(
    llm, tokenizer, safe_prompts, unsafe_prompts, sampling_params
)
```

**优势**：
- ✅ 更符合实际使用场景
- ✅ 不需要预先知道模型的回复
- ✅ 自动分离 prompt 和 generation 的 routing
- ✅ 支持批量处理

---

### 2. Routing 信息的分离

#### 原代码
```python
# routing 信息混在一起，需要通过 target string 手动定位
outputs = get_routings(messages)  # 包含整个对话的 routing
router_logits = outputs["router_logits"]  # (layers, all_tokens, experts)

# 需要手动查找 target tokens 的位置
target_token_ids = tokenizer(target_text, add_special_tokens=False).input_ids
locations = find_sub_list(target_token_ids, outputs["prompt_token_ids"])
start_idx, end_idx = locations[0]

# 手动提取相关 tokens 的 routing
relevant_routings = router_logits[:, start_idx:end_idx+1, :]
```

#### 新代码
```python
# 自动分离 prompt 和 generation 部分
routing_output = get_routings_with_generation(llm, messages, sampling_params)
separated = separate_prompt_and_generation_routings(routing_output, tokenizer)

# 直接获得分离后的 routing
prompt_routings = separated["prompt_routings"]        # prompt 部分
generation_routings = separated["generation_routings"]  # 生成部分
```

**优势**：
- ✅ 自动分离，无需手动定位
- ✅ 清晰区分 prompt 和 generation
- ✅ 减少错误风险
- ✅ 代码更简洁

---

### 3. Safe vs Unsafe 对比方式

#### 原代码
```python
# 使用 messages_0 和 messages_1 进行对比
# 不明确哪个是 safe，哪个是 unsafe
freq = {
    "messages_0": [],  # 实际上是 safe
    "messages_1": []   # 实际上是 unsafe
}

# 需要手动处理每个 target
for row_idx in range(len(dfs["messages_0"])):
    router_prob_n2_1 = dfs["messages_0"].iloc[row_idx]["router_prob_n2"]
    router_prob_n2_2 = dfs["messages_1"].iloc[row_idx]["router_prob_n2"]
    
    # 复杂的 token 定位逻辑
    subset_1 = dfs["messages_0"].iloc[row_idx]["messages_0_target"]["token_ids"]
    range_1 = find_sub_list(subset_1, dfs["messages_0"].iloc[row_idx]["prompt_token_ids"])
    # ...
```

#### 新代码
```python
# 明确的 safe 和 unsafe 标签
safe_prompts = ["Normal question 1", "Normal question 2"]
unsafe_prompts = ["Unsafe question 1", "Unsafe question 2"]

# 批量处理，自动标记
safe_outputs, unsafe_outputs = process_prompt_batch(
    llm, tokenizer, safe_prompts, unsafe_prompts, sampling_params
)

# 直接计算 risk difference
safe_routings = [out["generation_routings"] for out in safe_outputs]
unsafe_routings = [out["generation_routings"] for out in unsafe_outputs]

risk_diff_df = calculate_risk_diff(
    safe_routings, unsafe_routings, num_experts_per_tok
)
```

**优势**：
- ✅ 语义清晰：safe vs unsafe
- ✅ 自动化处理，减少手动操作
- ✅ 易于扩展到更多 prompts
- ✅ 代码可读性大幅提升

---

### 4. Risk Difference 计算

#### 原代码
```python
# 函数定义在主流程中，难以复用
def calc_risk_diff(prob1, prob2):
    # 长达 40 行的计算逻辑
    # prob1, prob2 命名不清晰
    a1, a2, d1, d2 = np.zeros(...), np.zeros(...), np.zeros(...), np.zeros(...)
    # ...
    
# 调用时不清楚参数含义
df = calc_risk_diff(freq[subset1], freq[subset2])
```

#### 新代码
```python
# 独立函数，清晰的参数命名
def calculate_risk_diff(
    safe_routings: List[np.ndarray],
    unsafe_routings: List[np.ndarray],
    num_experts_per_tok: int
) -> pd.DataFrame:
    """
    Calculate risk difference between safe and unsafe expert activations.
    
    Returns DataFrame with columns: layer, expert, a_safe, a_unsafe, 
    a_safe_n, a_unsafe_n, risk_diff
    """
    # 清晰的变量命名
    a_safe = np.zeros((num_layers, n_experts))
    a_unsafe = np.zeros((num_layers, n_experts))
    # ...
    
# 调用时一目了然
risk_diff_df = calculate_risk_diff(
    safe_routings, unsafe_routings, num_experts_per_tok
)
```

**优势**：
- ✅ 函数独立，易于测试和复用
- ✅ 参数命名清晰（safe vs unsafe）
- ✅ 有完整的文档字符串
- ✅ 返回类型明确

---

### 5. 训练流程集成

#### 原代码
```python
# 没有训练代码，只有数据准备
# 用户需要自己实现：
# 1. 如何识别 unsafe experts
# 2. 如何冻结其他参数
# 3. 如何准备训练数据
# 4. 如何设置训练循环
```

#### 新代码
```python
# 完整的训练流程
# 1. 自动识别 unsafe experts
unsafe_experts = identify_unsafe_experts(risk_diff_df, threshold=0.05, top_k=50)

# 2. 自动冻结参数
model = freeze_model_except_unsafe_experts(model, unsafe_experts)

# 3. 准备训练数据
sft_dataset = prepare_sft_dataset(unsafe_outputs, tokenizer)

# 4. 直接运行训练
python train_unsafe_experts.py --model_name Qwen/Qwen3-30B-A3B
```

**优势**：
- ✅ 端到端的完整流程
- ✅ 自动化参数冻结
- ✅ 开箱即用的训练脚本
- ✅ 清晰的训练配置

---

## 代码结构对比

### 原代码（demo.py）

```
demo.py (330 lines)
├── 环境设置 (1-61)
├── get_routings() (63-89)
├── 测试代码 (91-96)
├── 数据集定义 (98-114)
├── 主循环：处理 messages_0 和 messages_1 (116-193)
│   ├── find_sub_list 函数
│   ├── tokenizer 加载
│   ├── routing 收集
│   └── 保存结果
├── 加载结果 (195-199)
├── 计算 router probability (202-219)
├── 合并频率 (221-271)
└── 计算 risk_diff (273-330)
```

**问题**：
- ❌ 所有逻辑混在一个文件中
- ❌ 函数定义在循环中
- ❌ 难以复用和测试
- ❌ 没有训练部分

### 新代码

```
demo_refactored.py (600+ lines)
├── 环境设置和导入
├── 配置部分
├── Helper Functions
│   ├── find_sub_list()
│   └── get_model_num_experts()
├── Step 1: 数据收集
│   ├── get_routings_with_generation()
│   └── separate_prompt_and_generation_routings()
├── Step 2: 批量处理
│   └── process_prompt_batch()
├── Step 3: 分析
│   ├── get_router_probabilities()
│   ├── calculate_risk_diff()
│   └── identify_unsafe_experts()
├── Step 4: 训练准备
│   ├── create_expert_mask()
│   ├── prepare_sft_dataset()
│   └── setup_model_for_expert_finetuning()
└── main_pipeline()

train_unsafe_experts.py (300+ lines)
├── load_prepared_data()
├── freeze_model_except_unsafe_experts()
├── prepare_dataset_for_training()
└── train_unsafe_experts()

example_usage.py (150+ lines)
└── quick_example()  # 最小示例
```

**优势**：
- ✅ 模块化设计
- ✅ 每个函数职责单一
- ✅ 易于测试和维护
- ✅ 包含完整训练流程
- ✅ 提供快速示例

---

## 使用体验对比

### 原代码使用流程

1. 准备完整的对话对（包括 assistant 回复）
2. 手动定义 target string
3. 运行脚本收集 routing
4. 手动分析结果
5. 自己实现训练代码

**时间**: ~2-3 小时设置 + 需要深入理解代码

### 新代码使用流程

**快速测试**（5 分钟）：
```bash
python example_usage.py
```

**完整流程**（20 分钟）：
```bash
# 1. 收集数据和分析
python demo_refactored.py

# 2. 运行训练
python train_unsafe_experts.py
```

**时间**: ~5-20 分钟，开箱即用

---

## 可扩展性对比

### 原代码

添加新功能需要：
- 修改主循环
- 理解整体数据流
- 可能破坏现有逻辑

### 新代码

添加新功能只需：
```python
# 例如：添加新的 expert 选择策略
def identify_unsafe_experts_by_variance(
    risk_diff_df: pd.DataFrame,
    variance_threshold: float = 0.1
) -> List[Tuple[int, int]]:
    """新的选择策略"""
    # 实现新逻辑
    pass

# 在主流程中替换即可
unsafe_experts = identify_unsafe_experts_by_variance(risk_diff_df)
```

---

## 性能对比

| 方面 | 原代码 | 新代码 |
|------|--------|--------|
| 代码可读性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可维护性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可扩展性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 易用性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 功能完整性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 运行效率 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (相同) |

---

## 总结

### 主要改进

1. **✅ 输入方式更自然**：只需 user prompt，不需要预定义回复
2. **✅ 自动化程度更高**：自动分离 prompt 和 generation routing
3. **✅ 代码结构更清晰**：函数化、模块化设计
4. **✅ 语义更明确**：safe vs unsafe，而非 messages_0 vs messages_1
5. **✅ 功能更完整**：包含端到端的训练流程
6. **✅ 易于使用**：提供快速示例和详细文档

### 适用场景

**使用原代码（demo.py）**：
- 需要精确控制 target tokens
- 已有完整的对话对数据
- 只需要分析，不需要训练

**使用新代码（demo_refactored.py + train_unsafe_experts.py）**：
- 想要快速识别和修复 unsafe experts
- 需要批量处理多个 prompts
- 需要端到端的训练流程
- 重视代码可维护性和可扩展性

### 迁移建议

如果你已经在使用原代码，可以这样迁移：

```python
# 原代码的数据
messages_0 = [{"role": "user", "content": "Count"}, 
              {"role": "assistant", "content": "1, 2, 3"}]
messages_1 = [{"role": "user", "content": "Count"}, 
              {"role": "assistant", "content": "one, two"}]

# 转换为新代码格式
safe_prompt = messages_0[0]["content"]   # 只取 user prompt
unsafe_prompt = messages_1[0]["content"]

# 使用新 API
safe_outputs, unsafe_outputs = process_prompt_batch(
    llm, tokenizer, [safe_prompt], [unsafe_prompt], sampling_params
)
```

