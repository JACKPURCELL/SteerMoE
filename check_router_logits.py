#!/usr/bin/env python3
"""Check router logits files to see their shapes and content."""

import numpy as np
import glob
import os

TEMP_NPY_BASE_PATH = "./temp_routings/"

# Find all router logits files
files = sorted(glob.glob(f"{TEMP_NPY_BASE_PATH}/router_logits_L*.npy"))

print(f"Found {len(files)} router logits files\n")

# Check first few files
for i, file_path in enumerate(files[:5]):
    logits = np.load(file_path)
    layer_num = os.path.basename(file_path).replace("router_logits_L", "").replace(".npy", "")
    print(f"Layer {layer_num}:")
    print(f"  Shape: {logits.shape}")
    print(f"  Dtype: {logits.dtype}")
    print(f"  Min/Max: {logits.min():.4f} / {logits.max():.4f}")
    print(f"  Mean: {logits.mean():.4f}")
    print(f"  All zeros: {np.all(logits == 0)}")
    print()

# Check if all values are zero
all_zeros = []
for file_path in files:
    logits = np.load(file_path)
    if np.all(logits == 0):
        layer_num = os.path.basename(file_path).replace("router_logits_L", "").replace(".npy", "")
        all_zeros.append(layer_num)

if all_zeros:
    print(f"WARNING: {len(all_zeros)} layers have all-zero router logits!")
    print(f"Layers: {', '.join(all_zeros)}")
else:
    print("All layers have non-zero router logits.")

