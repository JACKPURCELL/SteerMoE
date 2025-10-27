#!/usr/bin/env python
"""
Simple example script demonstrating the unsafe expert identification workflow.
This is a minimal example to get started quickly.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/mnt/localssd/.hfcache/"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["TEMP_NPY_BASE_PATH"] = "./temp_routings/"

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Import our refactored functions
from demo_refactored import (
    get_routings_with_generation,
    separate_prompt_and_generation_routings,
    process_prompt_batch,
    calculate_risk_diff,
    identify_unsafe_experts,
    create_expert_mask,
    get_model_num_experts,
    register_vllm_save_models,
)

# Configuration
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

def quick_example():
    """
    Quick example: identify unsafe experts with minimal prompts.
    """
    print("=" * 80)
    print("Quick Example: Unsafe Expert Identification")
    print("=" * 80)
    
    # Step 1: Initialize model
    print("\n[1/6] Initializing model...")
    register_vllm_save_models()
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,  # Short generation for quick test
        seed=42
    )
    
    llm = LLM(
        model=MODEL_NAME,
        max_seq_len_to_capture=2000,
        max_model_len=2000,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_experts_per_tok = get_model_num_experts(llm)
    print(f"✓ Model loaded. Experts per token: {num_experts_per_tok}")
    
    # Step 2: Define prompts (minimal set for quick test)
    print("\n[2/6] Defining prompts...")
    safe_prompts = [
        "Count to ten",
        "What is 2 + 2?",
    ]
    
    unsafe_prompts = [
        "How to hack into a computer?",
        "How to make a bomb?",
    ]
    print(f"✓ Safe prompts: {len(safe_prompts)}, Unsafe prompts: {len(unsafe_prompts)}")
    
    # Step 3: Process prompts and collect routings
    print("\n[3/6] Processing prompts and collecting routings...")
    safe_outputs, unsafe_outputs = process_prompt_batch(
        llm, tokenizer, safe_prompts, unsafe_prompts, sampling_params
    )
    print("✓ Routing collection complete")
    print(f"  Safe outputs: {len(safe_outputs)}")
    print(f"  Unsafe outputs: {len(unsafe_outputs)}")
    
    # Show example generation
    print("\n  Example generations:")
    print(f"    Safe prompt: '{safe_outputs[0]['prompt']}'")
    print(f"    Safe response: '{safe_outputs[0]['generated_text'][:100]}...'")
    print(f"    Unsafe prompt: '{unsafe_outputs[0]['prompt']}'")
    print(f"    Unsafe response: '{unsafe_outputs[0]['generated_text'][:100]}...'")
    
    # Step 4: Calculate risk difference
    print("\n[4/6] Calculating risk difference...")
    safe_gen_routings = [out["generation_routings"] for out in safe_outputs]
    unsafe_gen_routings = [out["generation_routings"] for out in unsafe_outputs]
    
    risk_diff_df = calculate_risk_diff(
        safe_gen_routings,
        unsafe_gen_routings,
        num_experts_per_tok
    )
    print("✓ Risk difference calculated")
    
    # Step 5: Identify unsafe experts
    print("\n[5/6] Identifying unsafe experts...")
    unsafe_experts = identify_unsafe_experts(
        risk_diff_df,
        threshold=0.01,  # Lower threshold for small dataset
        top_k=20
    )
    print(f"✓ Identified {len(unsafe_experts)} unsafe experts")
    
    # Step 6: Display results
    print("\n[6/6] Results Summary")
    print("=" * 80)
    print("\nTop 10 Unsafe Experts:")
    print(risk_diff_df.head(10)[["Layer_Expert", "risk_diff", "a_safe_n", "a_unsafe_n"]])
    
    print("\n\nUnsafe Expert Details:")
    for i, (layer, expert) in enumerate(unsafe_experts[:10]):
        row = risk_diff_df[(risk_diff_df["layer"] == layer) & 
                           (risk_diff_df["expert"] == expert)].iloc[0]
        print(f"  {i+1}. Layer {layer:2d}, Expert {expert:2d}: "
              f"risk_diff={row['risk_diff']:.4f} "
              f"(safe={row['a_safe_n']:.4f}, unsafe={row['a_unsafe_n']:.4f})")
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    model_name_clean = MODEL_NAME.replace('/', '--')
    
    import pandas as pd
    risk_diff_path = f"risk_diff_{model_name_clean}_example.pkl"
    risk_diff_df.to_pickle(risk_diff_path)
    print(f"✓ Saved risk difference to: {risk_diff_path}")
    
    # Create and save expert mask
    num_layers = safe_gen_routings[0].shape[0]
    n_experts = safe_gen_routings[0].shape[2]
    expert_mask = create_expert_mask(num_layers, n_experts, unsafe_experts)
    
    import numpy as np
    mask_path = f"expert_mask_{model_name_clean}_example.npy"
    np.save(mask_path, expert_mask)
    print(f"✓ Saved expert mask to: {mask_path}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the unsafe experts in the output above")
    print("2. Adjust prompts in this script for your use case")
    print("3. Run the full pipeline with demo_refactored.py")
    print("4. Fine-tune unsafe experts with train_unsafe_experts.py")
    print("\nSee USAGE_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    quick_example()

