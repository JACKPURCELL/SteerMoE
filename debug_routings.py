#!/usr/bin/env python
"""
Debug script to check routing data in batch_outputs.
"""

import pandas as pd
import numpy as np

print("Loading batch_outputs...")
data = pd.read_pickle("batch_outputs_allenai--OLMoE-1B-7B-0125-Instruct.pkl")

batch_data_list = data['batch_data_list']

print(f"\nNumber of batches: {len(batch_data_list)}")

# Check first batch
batch_0 = batch_data_list[0]

print("\n" + "=" * 80)
print("Batch 0 Information:")
print("=" * 80)
print(f"Batch index: {batch_0['batch_idx']}")
print(f"Safe indices: {batch_0['safe_indices']}")
print(f"Unsafe indices: {batch_0['unsafe_indices']}")
print(f"Number of safe routings: {len(batch_0['safe_routings'])}")
print(f"Number of unsafe routings: {len(batch_0['unsafe_routings'])}")

# Check safe_routings
print("\n" + "=" * 80)
print("Safe Routings:")
print("=" * 80)
for i, routing in enumerate(batch_0['safe_routings']):
    print(f"Safe {i}: shape = {routing.shape}")
    if routing.shape[1] == 0:
        print(f"  ⚠️  WARNING: No tokens in generation_routings!")

# Check unsafe_routings
print("\n" + "=" * 80)
print("Unsafe Routings:")
print("=" * 80)
for i, routing in enumerate(batch_0['unsafe_routings']):
    print(f"Unsafe {i}: shape = {routing.shape}")
    if routing.shape[1] == 0:
        print(f"  ⚠️  WARNING: No tokens in generation_routings!")

# Check one sample output
print("\n" + "=" * 80)
print("Sample Output (Safe 0):")
print("=" * 80)
safe_idx = batch_0['safe_indices'][0]
sample_output = batch_0['batch_outputs'][safe_idx]
print(f"Prompt: {sample_output.get('prompt', 'N/A')[:100]}...")
print(f"Generated text: {sample_output.get('generated_text', 'N/A')[:100]}...")
print(f"Num prompt tokens: {sample_output.get('num_prompt_tokens', 'N/A')}")
print(f"Num generated tokens: {sample_output.get('num_generated_tokens', 'N/A')}")

if 'generation_routings' in sample_output:
    print(f"Generation routings shape: {sample_output['generation_routings'].shape}")
else:
    print("⚠️  generation_routings not in sample_output!")

if 'prompt_routings' in sample_output:
    print(f"Prompt routings shape: {sample_output['prompt_routings'].shape}")

if 'all_routings' in sample_output:
    print(f"All routings shape: {sample_output['all_routings'].shape}")
else:
    print("⚠️  all_routings not in sample_output!")

# Check if we should use prompt_routings instead
print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)

if len(batch_0['safe_routings']) > 0:
    total_safe_tokens = sum(r.shape[1] for r in batch_0['safe_routings'])
    total_unsafe_tokens = sum(r.shape[1] for r in batch_0['unsafe_routings'])
    
    print(f"Total safe tokens: {total_safe_tokens}")
    print(f"Total unsafe tokens: {total_unsafe_tokens}")
    
    if total_safe_tokens == 0 and total_unsafe_tokens == 0:
        print("\n⚠️  PROBLEM FOUND: Both safe and unsafe routings have 0 tokens!")
        print("\nThis means generation_routings is empty.")
        print("We should use prompt_routings instead!")
        
        # Check prompt_routings
        print("\nChecking prompt_routings from batch_outputs...")
        if 'prompt_routings' in batch_0['batch_outputs'][0]:
            for i, idx in enumerate(batch_0['safe_indices'][:2]):
                output = batch_0['batch_outputs'][idx]
                if 'prompt_routings' in output:
                    print(f"Safe {i} prompt_routings: {output['prompt_routings'].shape}")

