import os
import datasets
from vllm import SamplingParams
from openai import OpenAI
import time

from judge import evaluate_with_judge
from tqdm import tqdm
dataset = datasets.load_dataset("marslabucla/XGuard-Train")
model_id = "Qwen/Qwen3-30B-A3B"
# model_id = "allenai/OLMoE-1B-7B-0125-Instruct"
def vLLMchatCompletion(messages, base_url="http://0.0.0.0:8000/v1"):
    retry_time = 0
    # vLLM uses OpenAI-compatible API, no API key needed for local deployment
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key for local deployment
        base_url=base_url
    )
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.6
        )
    except Exception as e:
        print(e)
        for retry_time in range(retry_time):
            retry_time = retry_time + 1
            print(f"{model_id} Retry {retry_time}")
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    # temperature=self.temperature
                )
                break
            except:
                continue
            
    model_output = response.choices[0].message.content.strip()

    return model_output
safe_messages = []
unsafe_messages = []
for item in tqdm(dataset["train"]):
    conversations = item["conversations"] 
    messages = []
    
    for conversation in conversations:
        if conversation["from"] == "human":
            messages.append({"role": "user", "content": conversation["value"]})
        else:
            messages.append({"role": "assistant", "content": conversation["value"]})
    response = vLLMchatCompletion(messages[:-1])
    judge_result = evaluate_with_judge(messages[:-1], response)
    if judge_result["score"] != 0:
        safe_messages.append(messages)
        unsafe_message = messages[:-1]
        unsafe_message.append({"role": "assistant", "content": response})
        unsafe_messages.append(unsafe_message)