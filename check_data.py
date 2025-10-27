#!/usr/bin/env python
"""
Quick script to check the generated data files.
"""

import pandas as pd
import numpy as np

# Load the saved data
print("Loading sft_dataset...")
sft_data = pd.read_pickle("sft_dataset_allenai--OLMoE-1B-7B-0125-Instruct.pkl")

print("\n" + "=" * 80)
print("Data Structure:")
print("=" * 80)
print(f"Keys: {list(sft_data.keys())}")

print("\n" + "=" * 80)
print("Batch Information:")
print("=" * 80)
print(f"Number of batches: {len(sft_data['batch_data_list'])}")

print("\n" + "=" * 80)
print("Unsafe Experts per Batch:")
print("=" * 80)
batch_unsafe_experts = sft_data['batch_unsafe_experts']
for i in range(min(10, len(batch_unsafe_experts))):
    num_experts = len(batch_unsafe_experts[i])
    print(f"  Batch {i}: {num_experts} unsafe experts")
    if num_experts > 0 and i < 3:
        print(f"    Sample: {list(batch_unsafe_experts[i])[:5]}")

print("\n" + "=" * 80)
print("Statistics:")
print("=" * 80)
expert_counts = [len(experts) for experts in batch_unsafe_experts]
print(f"Total batches: {len(expert_counts)}")
print(f"Unsafe expert counts: Min={min(expert_counts)}, Max={max(expert_counts)}, Mean={np.mean(expert_counts):.2f}")

if max(expert_counts) == 0:
    print("\n⚠️  WARNING: No unsafe experts identified!")
    print("This could be because:")
    print("  1. Threshold is too high")
    print("  2. Not enough data per batch")
    print("  3. Risk_diff values are all very small")
    
    # Check risk_diff data
    print("\nChecking risk_diff data...")
    risk_diff_data = pd.read_pickle("risk_diff_allenai--OLMoE-1B-7B-0125-Instruct.pkl")
    
    if 'batch_risk_diffs' in risk_diff_data:
        print(f"\nNumber of batch risk_diffs: {len(risk_diff_data['batch_risk_diffs'])}")
        
        # Check first batch
        if len(risk_diff_data['batch_risk_diffs']) > 0:
            batch_0_df = risk_diff_data['batch_risk_diffs'][0]
            print(f"\nBatch 0 risk_diff DataFrame:")
            print(f"  Shape: {batch_0_df.shape}")
            print(f"  Columns: {list(batch_0_df.columns)}")
            print(f"  Risk_diff range: [{batch_0_df['risk_diff'].min():.6f}, {batch_0_df['risk_diff'].max():.6f}]")
            print(f"  Risk_diff > 0: {(batch_0_df['risk_diff'] > 0).sum()} experts")
            print(f"  Risk_diff > 0.01: {(batch_0_df['risk_diff'] > 0.01).sum()} experts")
            print(f"  Risk_diff > 0.05: {(batch_0_df['risk_diff'] > 0.05).sum()} experts")
            
            print(f"\nTop 10 experts by risk_diff (Batch 0):")
            top_10 = batch_0_df.nlargest(10, 'risk_diff')[['layer', 'expert', 'risk_diff', 'a_safe', 'a_unsafe']]
            print(top_10)
    
    print("\n💡 Suggestion: Re-run demo_refactored.py with threshold=0.0 to get top-k experts")
else:
    print(f"\n✓ Successfully identified unsafe experts across batches")
    
    # Show union
    all_unsafe = set()
    for batch_experts in batch_unsafe_experts:
        all_unsafe.update(batch_experts)
    print(f"\nTotal unique unsafe experts (union): {len(all_unsafe)}")

print("\n" + "=" * 80)
print("Model Information:")
print("=" * 80)
if 'num_layers' in sft_data:
    print(f"Number of layers: {sft_data['num_layers']}")
if 'n_experts' in sft_data:
    print(f"Number of experts per layer: {sft_data['n_experts']}")
if 'num_experts_per_tok' in sft_data:
    print(f"Number of experts per token: {sft_data['num_experts_per_tok']}")

