#!/usr/bin/env python3
"""Debug why router logits only have 1 token."""

import numpy as np
import pickle

# Load the saved dataset
with open("sft_dataset_Qwen--Qwen3-30B-A3B.pkl", 'rb') as f:
    data = pickle.load(f)

print("Dataset structure:")
print(f"Keys: {data.keys()}\n")

# Check batch data
batch_data_list = data['batch_data_list']
print(f"Number of batches: {len(batch_data_list)}")

if len(batch_data_list) > 0:
    batch_0 = batch_data_list[0]
    print(f"\nBatch 0 structure:")
    print(f"  Keys: {batch_0.keys()}")
    print(f"  Number of outputs: {len(batch_0['batch_outputs'])}")
    print(f"  Safe indices: {batch_0['safe_indices']}")
    print(f"  Unsafe indices: {batch_0['unsafe_indices']}")
    
    # Check routing shapes
    if len(batch_0['batch_outputs']) > 0:
        output_0 = batch_0['batch_outputs'][0]
        print(f"\nOutput 0 structure:")
        print(f"  Keys: {output_0.keys()}")
        
        if 'generation_routings' in output_0:
            gen_routings = output_0['generation_routings']
            print(f"  Generation routings shape: {gen_routings.shape}")
            print(f"  Expected format: (num_layers, num_generated_tokens, n_experts)")
            
        if 'prompt_routings' in output_0:
            prompt_routings = output_0['prompt_routings']
            print(f"  Prompt routings shape: {prompt_routings.shape}")
            
        if 'all_routings' in output_0:
            all_routings = output_0['all_routings']
            print(f"  All routings shape: {all_routings.shape}")
            
        if 'num_generated_tokens' in output_0:
            print(f"  Number of generated tokens: {output_0['num_generated_tokens']}")
        if 'num_prompt_tokens' in output_0:
            print(f"  Number of prompt tokens: {output_0['num_prompt_tokens']}")
            
        if 'generated_text' in output_0:
            print(f"  Generated text: {output_0['generated_text'][:100]}...")
