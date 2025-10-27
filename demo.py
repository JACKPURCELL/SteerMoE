# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/mnt/localssd/.hfcache/"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

os.environ["TEMP_NPY_BASE_PATH"] = "./temp_routings/"

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import register_vllm_save_models, register_vllm_models, steer_moe

try:
    load_dotenv()
    hf_hub.login(os.environ["HF_TOKEN"])
except Exception as e:
    print("HF_TOKEN not found in environment variables. Continuing without login.")
    pass

if not os.path.exists(os.environ["TEMP_NPY_BASE_PATH"]):
    os.makedirs(os.environ["TEMP_NPY_BASE_PATH"])
    
    
    # Supported Models: 
# "Qwen/Qwen3-30B-A3B", "openai/gpt-oss-120b", 
# "microsoft/Phi-3.5-MoE-instruct", "openai/gpt-oss-20b", 
# "mistralai/Mixtral-8x7B-Instruct-v0.1", "allenai/OLMoE-1B-7B-0125-Instruct"
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

register_vllm_save_models()
sampling_params = SamplingParams(temperature=0, top_p=0.8, top_k=1, min_p=0, max_tokens=1, seed=0)
llm = LLM(
    model=MODEL_NAME, 
    max_seq_len_to_capture=4000, max_model_len=4000, 
    tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95, max_num_seqs=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    trust_remote_code=True
)

def get_routings(messages):
    """
    Get the routing logits for the given messages.
    """
    for layer in range(500):
        TEMP_NPY_PATH = f"{os.environ['TEMP_NPY_BASE_PATH']}/router_logits_L{layer}.npy"
        if os.path.exists(TEMP_NPY_PATH):
            os.remove(TEMP_NPY_PATH)
        
    outputs = llm.chat(messages, sampling_params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"})
    
    all_router_logits = []
    for layer in range(500):
        try:
            TEMP_NPY_PATH = f"{os.environ['TEMP_NPY_BASE_PATH']}/router_logits_L{layer}.npy"
            router_logits = np.load(TEMP_NPY_PATH).astype(np.float16)
            all_router_logits.append(router_logits)
        except FileNotFoundError:
            continue

    all_router_logits = np.stack(all_router_logits, axis=0)  # (num_layers, num_tokens, n_experts)
    output = {
        "router_logits": all_router_logits.astype(np.float16),  # (num_layers, num_tokens, n_experts)
        "messages": messages,
        "prompt_token_ids": outputs[0].prompt_token_ids,
    }
    return output

messages = [
    [{"role": "user", "content": "Hello"},]
]
r = get_routings(messages)
print(r.keys(), len(r["prompt_token_ids"]))
r["router_logits"].shape, r["router_logits"][:2, :2, :2]


### - messages_0: The messages for the first behavior response (ex safe)
### - messages_1: The messages for the second behavior response (ex unsafe)
### - messages_0_target: The target string for the first behavior response (Which tokens to compare routings)
### - messages_1_target: The target string for the second behavior response (Which tokens to compare routings)
DATASET_NAME = "custom_dataset"

df_ds = pd.DataFrame([
    {
        "messages_0": [{"role": "user", "content": "Count to ten"}, {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}],
        "messages_1": [{"role": "user", "content": "Count to ten"}, {"role": "assistant", "content": "one, two, three, four, five, six, seven, eight, nine, ten"}],
        "messages_0_target": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        "messages_1_target": "one, two, three, four, five, six, seven, eight, nine, ten",
    },
])

df_ds

for PAIR_CHOICE in ["messages_0", "messages_1"]:
    def find_sub_list(sl,l):
        results = []
        sll = len(sl)
        for ind in (i for i,e in enumerate(l) if e == sl[0]):
            if l[ind:ind+sll] == sl:
                results.append((ind, ind + sll - 1))
        return results

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    outputs_list = []
    for i in tqdm(range(len(df_ds))):
        # Get routings
        messages = df_ds.iloc[i][PAIR_CHOICE]
        outputs = get_routings(messages)  # 'router_logits', 'messages', 'prompt_token_ids'
        # Check shapes
        num_layers, num_tokens, n_experts = outputs["router_logits"].shape
        assert num_tokens == len(outputs["prompt_token_ids"])
        print(outputs["router_logits"].shape)
        # Store prompt tokens
        outputs["prompt_tokens"] = tokenizer.convert_ids_to_tokens(outputs["prompt_token_ids"], skip_special_tokens=False)
        outputs["prompt_tokens_special_mask"] = tokenizer.get_special_tokens_mask(
            outputs["prompt_token_ids"], already_has_special_tokens=True,
        )
        # Store the messages in a tokenized format
        outputs["messages_tokenized"] = [{
            "role": message["role"], 
            "content_token_ids": tokenizer(message["content"], add_special_tokens=False).input_ids,
            "content_tokens": tokenizer.convert_ids_to_tokens(tokenizer(message["content"], add_special_tokens=False).input_ids)
        } for message in messages]
        # Store the target texts and their tokenized forms (for detection comparison on these tokens)
        for col in [f"{PAIR_CHOICE}_target"]:
            target_text = df_ds.iloc[i][col]
            target_token_ids = tokenizer(df_ds.iloc[i][col], add_special_tokens=False).input_ids
            target_tokens = tokenizer.convert_ids_to_tokens(target_token_ids, skip_special_tokens=False)
            locations = find_sub_list(target_token_ids, outputs["prompt_token_ids"])
            assert len(locations) >= 1, f"Expected exactly one location: {locations}, for target text: \n{target_tokens}, in prompt tokens: \n{outputs['prompt_tokens']}"
            if len(locations) > 1:
                print(f"Expected exactly one location: {locations}, for target text: \n{target_tokens}, in prompt tokens: \n{outputs['prompt_tokens']}")
                print("Using last one")
                locations[0] = locations[-1]  # Use the last location if there are multiple
            outputs[col] = {
                "text": target_text,
                "tokens": target_tokens,
                "token_ids": target_token_ids,
                "start_idx": locations[0][0] if locations else None,
                "end_idx": locations[0][1] if locations else None,
            }
        # Append the outputs for this example to the list
        outputs_list.append(outputs)

    def get_model_num_experts(self):
        model = self.model_runner.model
        if hasattr(model, "model_config") and hasattr(model.model_config, "num_experts_per_tok"):
            return model.model_config.num_experts_per_tok  # gpt-oss
        elif hasattr(model.config, "num_experts_per_tok"):
            num_experts_per_tok = f"{model.config.num_experts_per_tok}"
        else:
            num_experts_per_tok = f"{model.config.text_config.num_experts_per_tok}"  # llama4
        return num_experts_per_tok
    num_experts_per_tok = llm.collective_rpc(get_model_num_experts)[0]
    print(num_experts_per_tok)
    print(outputs_list[0].keys())
    df = pd.DataFrame(outputs_list)
    df.attrs = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "doc_choice": PAIR_CHOICE,
        "num_experts": n_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "col_names": {},
    }
    path = f"output_[{MODEL_NAME.replace('/', '--')}]_[{DATASET_NAME}]_[{PAIR_CHOICE}]_[{len(df)}].pkl"
    df.to_pickle(path)
    print("### SAVED ROUTINGS AT:", path)
    print(len(df))
    print(df.attrs)
    df.head(2)
    
    
    dfs = {
    "messages_0": pd.read_pickle(f"output_[{MODEL_NAME.replace('/', '--')}]_[{DATASET_NAME}]_[messages_0]_[{len(df)}].pkl"),
    "messages_1": pd.read_pickle(f"output_[{MODEL_NAME.replace('/', '--')}]_[{DATASET_NAME}]_[messages_1]_[{len(df)}].pkl"),
}
    
    
    TOKEN_REDUCE_FN = "rd"

def find_sub_list(sl,l):
    results = []
    sll = len(sl)
    for ind in (i for i,e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind + sll - 1))
    return results  # [start_idx, end_idx]

def get_router_prob_n2(row):
    """Get the probability of each expert selected by the router for a given token."""
    router_logits = torch.tensor(row["router_logits"])  # (layer, token, expert)
    router_prob = torch.nn.functional.softmax(router_logits, dim=-1)  # (layer, token, expert)
    return router_prob.cpu().numpy()  # (layer, token, expert)

for key in tqdm(dfs.keys()):
    dfs[key]["router_prob_n2"] = dfs[key].apply(get_router_prob_n2, axis=1)  # (layer, token, expert)

# Concat router freq for all examples
key_df_1 = "messages_0"
key_df_2 = "messages_1"
print(f"key_df_1: {key_df_1}, key_df_2: {key_df_2}")
freq = {key_df_1: [], key_df_2: []}
tokens = {key_df_1: [], key_df_2: []}

debug_example_starts = []
num_used_examples = 0
for row_idx in tqdm(range(0, len(dfs[key_df_1]))):
    router_prob_n2_1 = dfs[key_df_1].iloc[row_idx]["router_prob_n2"]  # (layer, token, expert)
    router_prob_n2_2 = dfs[key_df_2].iloc[row_idx]["router_prob_n2"]  # (layer, token, expert)
    num_tokens_1, num_tokens_2 = router_prob_n2_1.shape[1], router_prob_n2_2.shape[1]

    subset_1 = dfs[key_df_1].iloc[row_idx]["messages_0_target"]["token_ids"]
    subset_2 = dfs[key_df_2].iloc[row_idx]["messages_1_target"]["token_ids"]
    range_1 = find_sub_list(subset_1, dfs[key_df_1].iloc[row_idx]["prompt_token_ids"])
    range_2 = find_sub_list(subset_2, dfs[key_df_2].iloc[row_idx]["prompt_token_ids"])
    assert len(range_1) >= 1 and len(range_2) >= 1, f"Expected more than one range for each dataset, got {len(range_1)} and {len(range_2)}"
    range_1 = range_1[-1]
    range_2 = range_2[-1]
    num_used_examples += 1
    debug_example_starts.append(len(freq[key_df_1]))

    for token_1_idx in range(range_1[0], range_1[1] + 1):
        freq[key_df_1].append(router_prob_n2_1[:, token_1_idx, :])
        tokens[key_df_1].append(dfs[key_df_1].iloc[row_idx]['prompt_tokens'][token_1_idx])

    for token_2_idx in range(range_2[0], range_2[1] + 1):
        freq[key_df_2].append(router_prob_n2_2[:, token_2_idx, :])
        tokens[key_df_2].append(dfs[key_df_2].iloc[row_idx]['prompt_tokens'][token_2_idx])
    
    if len(freq[key_df_1]) > 2000000:
        print("Reached 2M token comparisons, stopping...")
        break

print(len(freq[key_df_1]))
freq[key_df_1] = np.stack(freq[key_df_1])
freq[key_df_2] = np.stack(freq[key_df_2])
print(freq[key_df_1].shape, freq[key_df_2].shape)
print(f"Used examples: {num_used_examples}")

if "eq" in TOKEN_REDUCE_FN:
    # Equalize the number of tokens in both datasets
    min_tokens = min(len(freq[key_df_1]), len(freq[key_df_2]))
    freq[key_df_1] = freq[key_df_1][:min_tokens]
    freq[key_df_2] = freq[key_df_2][:min_tokens]
    tokens[key_df_1] = tokens[key_df_1][:min_tokens]
    tokens[key_df_2] = tokens[key_df_2][:min_tokens]
    print(freq[key_df_1].shape, freq[key_df_2].shape)
# dfs['safe'].head(2)   

from scipy.stats import ttest_rel

NUM_EXPERTS_PER_TOK = int(dfs[key_df_1].attrs["num_experts_per_tok"])
print(f"Number of experts per token: {NUM_EXPERTS_PER_TOK}")

def calc_risk_diff(prob1, prob2):
    ### prob1,2 = (batch, layer, expert)
    ### Count how many times each expert is activated
    a1, a2, d1, d2 = np.zeros((prob1.shape[1], prob1.shape[2])), np.zeros((prob2.shape[1], prob2.shape[2])), np.zeros((prob1.shape[1], prob1.shape[2])), np.zeros((prob2.shape[1], prob2.shape[2]))
    pre_processed_act1 = np.argsort(prob1, axis=-1)  # Get top experts
    pre_processed_act2 = np.argsort(prob2, axis=-1)  # Get top experts
    
    for token_idx in tqdm(range(prob1.shape[0])):
        for layer in range(prob1.shape[1]):
            activated_experts_1 = pre_processed_act1[token_idx, layer, -NUM_EXPERTS_PER_TOK:]  # Get top 8 experts
            a1[layer, activated_experts_1] += 1
            deactivated_experts_1 = pre_processed_act1[token_idx, layer, :-NUM_EXPERTS_PER_TOK]  # Experts not activated in prob1
            d1[layer, deactivated_experts_1] += 1
            assert len(activated_experts_1) + len(deactivated_experts_1) == prob1.shape[2]  # num experts
    
    for token_idx in tqdm(range(prob2.shape[0])):
        for layer in range(prob2.shape[1]):
            activated_experts_2 = pre_processed_act2[token_idx, layer, -NUM_EXPERTS_PER_TOK:]
            a2[layer, activated_experts_2] += 1
            deactivated_experts_2 = pre_processed_act2[token_idx, layer, :-NUM_EXPERTS_PER_TOK]  # Experts not activated in prob2
            d2[layer, deactivated_experts_2] += 1
            assert len(activated_experts_2) + len(deactivated_experts_2) == prob2.shape[2]  # num experts

    layer_expert_paired_ttest = []
    for layer in tqdm(range(prob1.shape[1])):
        for expert in range(prob1.shape[2]):
            test_results = {
                "layer": layer,
                "expert": expert,
                "a1": a1[layer, expert],
                "a2": a2[layer, expert],
                "d1": d1[layer, expert],
                "d2": d2[layer, expert],
                "a1_n": (a1[layer, expert] / (a1[layer, expert] + d1[layer, expert])),
                "a2_n": (a2[layer, expert] / (a2[layer, expert] + d2[layer, expert])),
                "risk_diff": (a1[layer, expert] / (a1[layer, expert] + d1[layer, expert])) - (a2[layer, expert] / (a2[layer, expert] + d2[layer, expert]))
            }
            layer_expert_paired_ttest.append(test_results)
    return pd.DataFrame(layer_expert_paired_ttest)


subset1, subset2 = "messages_0", "messages_1"  # "random_doc"
df = calc_risk_diff(freq[subset1], freq[subset2])
df["Layer_Expert"] = df.apply(lambda x: f"L{int(x['layer']):02d}\nE{int(x['expert']):02d}", axis=1)

df["risk_diff_abs"] = df["risk_diff"].abs()
df = df.sort_values(by="risk_diff_abs", ascending=False).reset_index(drop=True)

path = f"activations_[{dfs[subset1].attrs['model_name'].replace('/', '--')}]_[{dfs[subset1].attrs['dataset_name']}]_[{TOKEN_REDUCE_FN}]_[{len(dfs[key_df_1])}]_[{len(freq[subset1])}].pkl"
df.to_pickle(path)
print(f"Saved to {path}")

df