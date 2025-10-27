# Training script for fine-tuning unsafe experts
# This script demonstrates how to use the prepared data to train only unsafe expert MLPs

import os
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from typing import List, Dict, Tuple
import numpy as np


def load_prepared_data(model_name: str) -> Tuple[List[Dict], List[Tuple[int, int]], np.ndarray]:
    """
    Load the prepared routing data, unsafe experts, and SFT dataset.
    
    Args:
        model_name: Model name (with slashes replaced by --)
        
    Returns:
        Tuple of (sft_dataset, unsafe_experts, expert_mask)
    """
    model_name_clean = model_name.replace('/', '--')
    
    # Load SFT dataset
    sft_path = f"sft_dataset_{model_name_clean}.pkl"
    sft_dataset = pd.read_pickle(sft_path)
    print(f"Loaded SFT dataset: {len(sft_dataset)} examples")
    
    # Load risk diff to get unsafe experts
    risk_diff_path = f"risk_diff_{model_name_clean}.pkl"
    risk_diff_df = pd.read_pickle(risk_diff_path)
    
    # Extract unsafe experts (threshold and top_k can be adjusted)
    threshold = 0.05
    top_k = 50
    unsafe_df = risk_diff_df[risk_diff_df["risk_diff"] > threshold].head(top_k)
    unsafe_experts = [(int(row["layer"]), int(row["expert"])) 
                     for _, row in unsafe_df.iterrows()]
    print(f"Loaded {len(unsafe_experts)} unsafe experts")
    
    # Create expert mask
    num_layers = risk_diff_df["layer"].max() + 1
    n_experts = risk_diff_df.groupby("layer")["expert"].count().iloc[0]
    expert_mask = np.zeros((num_layers, n_experts), dtype=np.float32)
    for layer, expert in unsafe_experts:
        expert_mask[layer, expert] = 1.0
    
    return sft_dataset, unsafe_experts, expert_mask


def freeze_model_except_unsafe_experts(
    model: AutoModelForCausalLM,
    unsafe_experts: List[Tuple[int, int]]
) -> AutoModelForCausalLM:
    """
    Freeze all model parameters except unsafe expert MLPs.
    
    Args:
        model: HuggingFace model
        unsafe_experts: List of (layer, expert) tuples
        
    Returns:
        Model with frozen parameters
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze unsafe expert MLPs
    unfrozen_count = 0
    for layer_idx, expert_idx in unsafe_experts:
        try:
            # For Qwen3-MoE and similar architectures
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[layer_idx]
                
                # Try different MoE architecture patterns
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                    
                    # Pattern 1: mlp.experts[expert_idx]
                    if hasattr(mlp, "experts"):
                        expert = mlp.experts[expert_idx]
                        for param in expert.parameters():
                            param.requires_grad = True
                        unfrozen_count += 1
                    
                    # Pattern 2: mlp.block_sparse_moe.experts[expert_idx] (Mixtral)
                    elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "experts"):
                        expert = mlp.block_sparse_moe.experts[expert_idx]
                        for param in expert.parameters():
                            param.requires_grad = True
                        unfrozen_count += 1
                    
                    # Pattern 3: Direct expert access
                    elif hasattr(layer, f"expert_{expert_idx}"):
                        expert = getattr(layer, f"expert_{expert_idx}")
                        for param in expert.parameters():
                            param.requires_grad = True
                        unfrozen_count += 1
                        
        except Exception as e:
            print(f"Warning: Could not unfreeze Layer {layer_idx}, Expert {expert_idx}: {e}")
    
    # Verify freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameter freezing complete:")
    print(f"  Unfrozen {unfrozen_count} experts")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    return model


def prepare_dataset_for_training(
    sft_dataset: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Convert SFT dataset to HuggingFace Dataset format.
    
    Args:
        sft_dataset: List of examples with 'messages' field
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset
    """
    def tokenize_function(examples):
        # Format: <user>prompt</user><assistant>response</assistant>
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Convert to HF Dataset
    dataset_dict = {"messages": [ex["messages"] for ex in sft_dataset]}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def train_unsafe_experts(
    model_name: str,
    output_dir: str = "./unsafe_expert_finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
):
    """
    Main training function for unsafe expert fine-tuning.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
    """
    print("=" * 80)
    print("Loading prepared data...")
    print("=" * 80)
    sft_dataset, unsafe_experts, expert_mask = load_prepared_data(model_name)
    
    print("\n" + "=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better training stability
        device_map="auto",
    )
    
    print("\n" + "=" * 80)
    print("Freezing model parameters...")
    print("=" * 80)
    model = freeze_model_except_unsafe_experts(model, unsafe_experts)
    
    print("\n" + "=" * 80)
    print("Preparing dataset...")
    print("=" * 80)
    train_dataset = prepare_dataset_for_training(sft_dataset, tokenizer)
    print(f"Dataset size: {len(train_dataset)}")
    
    print("\n" + "=" * 80)
    print("Setting up training...")
    print("=" * 80)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,  # Use bfloat16
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save expert mask for later use
    expert_mask_path = os.path.join(output_dir, "expert_mask.npy")
    np.save(expert_mask_path, expert_mask)
    print(f"Saved expert mask to: {expert_mask_path}")
    
    # Save unsafe experts list
    unsafe_experts_path = os.path.join(output_dir, "unsafe_experts.txt")
    with open(unsafe_experts_path, "w") as f:
        for layer, expert in unsafe_experts:
            f.write(f"{layer},{expert}\n")
    print(f"Saved unsafe experts to: {unsafe_experts_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train unsafe experts")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./unsafe_expert_finetuned",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    train_unsafe_experts(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

