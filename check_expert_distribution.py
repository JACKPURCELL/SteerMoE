#!/usr/bin/env python
"""
Check the distribution of unsafe experts across layers.
"""

import pandas as pd
import numpy as np
from collections import Counter

# Load data
sft_data = pd.read_pickle("sft_dataset_allenai--OLMoE-1B-7B-0125-Instruct.pkl")

batch_unsafe_experts = sft_data['batch_unsafe_experts']
num_layers = sft_data['num_layers']
n_experts = sft_data['n_experts']

print("=" * 80)
print("Expert Distribution Analysis")
print("=" * 80)
print(f"Model: {num_layers} layers, {n_experts} experts per layer")
print(f"Number of batches: {len(batch_unsafe_experts)}")

# Analyze first batch
batch_0_experts = batch_unsafe_experts[0]
print(f"\nBatch 0: {len(batch_0_experts)} unsafe experts")

# Count by layer
layer_counts = Counter()
for layer, expert in batch_0_experts:
    layer_counts[layer] += 1

print("\nExperts per layer (Batch 0):")
for layer in range(num_layers):
    count = layer_counts.get(layer, 0)
    status = "✓" if count > 0 else "⚠️  EMPTY"
    print(f"  Layer {layer:2d}: {count:2d} unsafe experts {status}")

# Check if any layer is empty
empty_layers = [layer for layer in range(num_layers) if layer_counts.get(layer, 0) == 0]
if empty_layers:
    print(f"\n⚠️  WARNING: {len(empty_layers)} layers have NO unsafe experts:")
    print(f"  Layers: {empty_layers}")
    print("\n💡 This means:")
    print("  1. If we mask ALL safe experts, these layers can't route any tokens")
    print("  2. Current implementation: We only freeze/unfreeze parameters, don't mask routing")
    print("  3. So the model can still use all experts, but only unsafe ones get updated")
else:
    print(f"\n✓ All {num_layers} layers have at least one unsafe expert")

# Show sample experts per layer
print("\nSample unsafe experts from each layer (Batch 0):")
for layer in range(min(5, num_layers)):
    experts_in_layer = [e for l, e in batch_0_experts if l == layer]
    if experts_in_layer:
        print(f"  Layer {layer}: experts {experts_in_layer[:5]}")

