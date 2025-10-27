# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HOME"] = "/mnt/localssd/.hfcache/"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from importlib import reload
from dotenv import load_dotenv
import huggingface_hub as hf_hub
from vllm import LLM, SamplingParams

from src.utils import register_vllm_models, steer_moe

try:
    load_dotenv()
    hf_hub.login(os.environ["HF_TOKEN"])
except Exception as e:
    print("HF_TOKEN not found in environment variables. Continuing without login.")
    pass

num_experts = pd.read_json("activations/num_experts.jsonl", lines=True)


# Supported Models: 
# "Qwen/Qwen3-30B-A3B", "openai/gpt-oss-120b", 
# "microsoft/Phi-3.5-MoE-instruct", "openai/gpt-oss-20b", 
# "mistralai/Mixtral-8x7B-Instruct-v0.1", "allenai/OLMoE-1B-7B-0125-Instruct"
MODEL = "Qwen/Qwen3-30B-A3B"
TASK = "faithfulness"  # "faithfulness" or "safety"
REVERSE_EFFECT = 0  # 0 to increase faithfulness/safety, 1 to decrease safety
config = {
    "model": MODEL,
    "task": TASK,
    "reverse_effect": REVERSE_EFFECT,
    "max_tokens": 512,

    "activations_path": f"activations/activations_[{MODEL.replace('/', '--')}]_[{TASK}].pkl",
    "num_pos_experts": num_experts[(num_experts["model"] == MODEL) & (num_experts["Task"] == TASK) & (num_experts["Reverse"] == REVERSE_EFFECT) & (num_experts["Activation"] == "Activated")]["Num Experts"].values[0],  # From Table A.1 in the paper
    "num_neg_experts": num_experts[(num_experts["model"] == MODEL) & (num_experts["Task"] == TASK) & (num_experts["Reverse"] == REVERSE_EFFECT) & (num_experts["Activation"] == "Deactivated")]["Num Experts"].values[0],  # From Table A.1 in the paper
}
config


register_vllm_models()

llm = LLM(
    model=MODEL, 
    max_seq_len_to_capture=4096, max_model_len=4096, 
    tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95, max_num_seqs=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    trust_remote_code=True,
)

batch_messages = [
    [
        {
            "role": "user", 
            "content": "Document: iPod was developed by Google\n Question: Who is the developer of iPod? \n Final Answer Only:"
        }
    ],
    [
        {
            "role": "user", 
            "content": "Document: The chief executive officer of Google is Lakshmi Mittal\n Question: Who is the chief executive officer of Google? \n Final Answer Only:"
        }
    ],
    [
        {
            "role": "user", 
            "content": "Document: Anderson Cooper is employed by National Review\n Question: Who is the employer of Anderson Cooper? \n Final Answer Only:"
        }
    ],
]

### Before Steering
paired_ttest_df = steer_moe(
    llm, config["activations_path"],
    num_pos_experts=0, num_neg_experts=0,
    steering_magnitude=1000, reverse_effect=config["reverse_effect"], strategy="risk_diff"
)
sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1, min_p=0, max_tokens=config["max_tokens"], seed=0)
outputs = llm.chat(batch_messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"},)
generations = [output.outputs[0].text for output in outputs]
generations

### After Steering
paired_ttest_df = steer_moe(
    llm, config["activations_path"],
    num_pos_experts=config["num_pos_experts"], num_neg_experts=config["num_neg_experts"],
    steering_magnitude=1000, reverse_effect=config["reverse_effect"], strategy="risk_diff"
)
sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1, min_p=0, max_tokens=config["max_tokens"], seed=0)
outputs = llm.chat(batch_messages, sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"},)
generations = [output.outputs[0].text for output in outputs]
generations