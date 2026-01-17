# Load dataset from JSON file
dataset_path = "/home/stufs1/jiachliang/SteerMoE/OLMOE-FWO.json"
with open(dataset_path, 'r') as f:
    dataset = json.load(f)
ori_prompts = []
ori_messages = []
unsafe_prompts = []
save_items = []
for item in dataset:
    if item["judge_success_gpt4"] == 0:
        continue
    # Safe prompt: use goal with simple user prompt and system prompt
    goal = item["goal"]
    safe_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": goal}
    ]
    ori_prompts.append(goal)
    ori_messages.append(safe_messages)
    # Unsafe prompt: use all_prompt (complete jailbreak attack)
    unsafe_messages = item["all_prompt"]
    unsafe_prompts.append(unsafe_messages)   
    save_items.append({"ori_prompt": goal, "unsafe_messages": unsafe_messages, "ori_messages": safe_messages})
    
    # messages = [{"role":"system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}]
from tqdm import tqdm
# judged_results = []
results = []
def chat_model(messages):
    outputs = llm.chat(
        messages, 
        sampling_params, 
        use_tqdm=False, 
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}
    )
    response = outputs[0].outputs[0].text
    return response
response = outputs[0].outputs[0].text
for item in tqdm(save_items[:50]):
    unsafe_messages = item["unsafe_messages"]
    response = chat_model(unsafe_messages)
    print(response)
    item["response"] = response
    results.append(item)


async def process_batch(batch):
    """Process a batch of items concurrently"""
    tasks = [eval(item["ori_prompt"], item["response"]) for item in batch]
    eval_jsons = await asyncio.gather(*tasks)
    return eval_jsons

async def process_all_results(results, batch_size=3):
    """Process all results in batches"""
    eval_results = []
    score = 0
    safe = 0
    
    # Split results into batches
    for i in tqdm(range(0, len(results), batch_size)):
        batch = results[i:i + batch_size]
        eval_jsons = await process_batch(batch)
        
        # Update items with eval results
        for item, eval_json in zip(batch, eval_jsons):
            item["eval_json"] = eval_json
            eval_results.append(item)
            score += eval_json["score"]
            if eval_json["score"] == 0:
                safe += 1
    
    score = score / len(results)
    return eval_results, score, safe

# Run the async function
eval_results, score, safe = await process_all_results(results, batch_size=3)
print(score)
print(safe)
# 100%|██████████| 50/50 [06:13<00:00,  7.47s/it]
# 0.12
# 48
