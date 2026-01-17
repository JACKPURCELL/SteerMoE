#!/usr/bin/env python3
"""Check Qwen3-30B-A3B model configuration to see which layers are MoE layers."""

from transformers import AutoConfig

MODEL_NAME = "Qwen/Qwen3-30B-A3B"

print(f"Loading config for {MODEL_NAME}...")
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("\n" + "="*80)
print("Model Configuration")
print("="*80)
print(f"Model type: {config.model_type}")
print(f"Number of hidden layers: {config.num_hidden_layers}")
print(f"Number of experts: {config.num_experts}")
print(f"Number of experts per token: {config.num_experts_per_tok}")
print(f"decoder_sparse_step: {config.decoder_sparse_step}")

# Check for mlp_only_layers
mlp_only_layers = [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
print(f"mlp_only_layers: {mlp_only_layers}")

print("\n" + "="*80)
print("Layer Analysis")
print("="*80)

moe_layers = []
mlp_layers = []

for layer_idx in range(config.num_hidden_layers):
    # This is the condition from Qwen3MoeDecoderLayer.__init__
    is_moe_layer = (
        (layer_idx not in mlp_only_layers) and 
        (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)
    )
    
    if is_moe_layer:
        moe_layers.append(layer_idx)
        print(f"Layer {layer_idx:2d}: MoE (has {config.num_experts} experts)")
    else:
        mlp_layers.append(layer_idx)
        print(f"Layer {layer_idx:2d}: Standard MLP")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Total layers: {config.num_hidden_layers}")
print(f"MoE layers: {len(moe_layers)} (indices: {moe_layers})")
print(f"Standard MLP layers: {len(mlp_layers)}")
print(f"\nPercentage of MoE layers: {100 * len(moe_layers) / config.num_hidden_layers:.1f}%")
