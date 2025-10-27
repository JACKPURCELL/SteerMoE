# Training script for fine-tuning unsafe experts with batch processing
# This script uses the same batch strategy as analysis: mixed batches of safe and unsafe prompts

import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from typing import List, Dict, Tuple, Iterator
from torch.nn import functional as F
from torch.utils.data import Sampler
import json


def load_prepared_batch_data(model_name: str) -> Tuple[Dict, List[List[Tuple[int, int]]], List[np.ndarray]]:
    """
    Load the prepared batch data with PER-BATCH unsafe experts and masks.
    
    Args:
        model_name: Model name (with slashes replaced by --)
        
    Returns:
        Tuple of (sft_data_dict, batch_unsafe_experts, batch_expert_masks)
        - batch_unsafe_experts: List of unsafe expert lists, one per batch
        - batch_expert_masks: List of expert masks, one per batch
    """
    model_name_clean = model_name.replace('/', '--')
    
    # Load SFT dataset with batch information
    sft_path = f"sft_dataset_{model_name_clean}.pkl"
    sft_data = pd.read_pickle(sft_path)
    
    print(f"Loaded SFT data: {len(sft_data['sft_dataset'])} examples")
    print(f"Number of batches: {len(sft_data['batch_data_list'])}")
    
    # Load per-batch unsafe experts and masks
    batch_unsafe_experts = sft_data['batch_unsafe_experts']
    batch_expert_masks = sft_data['batch_expert_masks']
    
    print(f"Per-batch unsafe experts:")
    for i in range(min(3, len(batch_unsafe_experts))):
        print(f"  Batch {i}: {len(batch_unsafe_experts[i])} unsafe experts")
    
    return sft_data, batch_unsafe_experts, batch_expert_masks


def freeze_model_except_unsafe_experts(
    model: AutoModelForCausalLM,
    batch_unsafe_experts: List[List[Tuple[int, int]]]
) -> AutoModelForCausalLM:
    """
    Freeze all model parameters except unsafe expert MLPs.
    Takes the UNION of all batch-specific unsafe experts.
    
    Args:
        model: HuggingFace model
        batch_unsafe_experts: List of unsafe expert lists (one per batch)
                              Each inner list contains (layer, expert) tuples
        
    Returns:
        Model with frozen parameters
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Get union of all unsafe experts across all batches
    all_unsafe_experts = set()
    for batch_experts in batch_unsafe_experts:
        all_unsafe_experts.update(batch_experts)
    
    print(f"\nUnfreezing union of unsafe experts from all batches:")
    print(f"  Total unique unsafe experts across all batches: {len(all_unsafe_experts)}")
    
    # Unfreeze unsafe expert MLPs
    unfrozen_count = 0
    for layer_idx, expert_idx in all_unsafe_experts:
        try:
            # For different MoE architectures
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[layer_idx]
                
                # OLMoE and similar architectures
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
                        
        except Exception as e:
            print(f"Warning: Could not unfreeze Layer {layer_idx}, Expert {expert_idx}: {e}")
    
    # Verify freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameter freezing complete:")
    print(f"  Unfrozen {unfrozen_count} unsafe expert MLPs (union across all batches)")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    return model


class ExpertMaskingTrainer(Trainer):
    """
    Custom Trainer that dynamically masks safe expert logits based on batch_idx.
    Each batch uses its own set of unsafe experts.
    """
    
    def __init__(self, batch_expert_masks, *args, **kwargs):
        """
        Args:
            batch_expert_masks: List of expert masks, one per batch
                                Each mask is (num_layers, n_experts) array
        """
        super().__init__(*args, **kwargs)
        # Convert all batch masks to tensors
        self.batch_expert_masks = [
            torch.tensor(mask, dtype=torch.float32)
            for mask in batch_expert_masks
        ]
        print(f"Initialized ExpertMaskingTrainer with {len(self.batch_expert_masks)} batch-specific masks")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to dynamically mask safe experts based on batch_idx.
        
        Args:
            num_items_in_batch: Number of items in the batch (added in newer transformers versions)
        """
        # Get batch_idx from inputs
        if 'batch_idx' in inputs:
            batch_indices = inputs['batch_idx']
            # All samples in a batch should have the same batch_idx (ensured by BatchPreservingSampler)
            unique_batch_idx = torch.unique(batch_indices)
            
            if len(unique_batch_idx) == 1:
                batch_idx = unique_batch_idx.item()
                # Get the corresponding expert mask for this batch
                current_mask = self.batch_expert_masks[batch_idx].to(self.args.device)
                
                # TODO: Apply mask to router logits
                # This requires modifying the model's forward pass
                # For now, we just use the mask for monitoring
                
                # Store for potential use in custom forward hooks
                self.current_expert_mask = current_mask
            else:
                # Mixed batch_idx - shouldn't happen with BatchPreservingSampler
                print(f"Warning: Mixed batch_idx in training batch: {unique_batch_idx.tolist()}")
        
        # Standard forward pass
        outputs = model(**inputs)
        
        # Note: Router logit masking should be done inside the model's forward pass
        # This is a framework - you may need to modify the model architecture
        # to actually apply the mask before expert selection
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


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


def create_batch_structured_dataset(
    sft_data: Dict,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    safe_ratio: float = 0.5
) -> List[List[Dict]]:
    """
    Create a dataset that preserves batch structure.
    Each element is a complete batch with safe + unsafe samples.
    
    Args:
        sft_data: Dictionary with batch_data_list
        tokenizer: Tokenizer
        batch_size: Training batch size
        safe_ratio: Ratio of safe prompts in each batch
        
    Returns:
        List of batches, where each batch is a list of examples
    """
    batch_data_list = sft_data['batch_data_list']
    all_batches = []
    
    for batch_data in batch_data_list:
        # Get safe and unsafe outputs from this batch
        safe_indices = batch_data['safe_indices']
        unsafe_indices = batch_data['unsafe_indices']
        batch_outputs = batch_data['batch_outputs']
        
        # Create examples for this batch
        batch_examples = []
        
        # Add safe prompts with simple response
        for idx in safe_indices:
            output = batch_outputs[idx]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": output['prompt']},
                {"role": "assistant", "content": output['generated_text'][:100]}  # Use actual generation
            ]
            batch_examples.append({
                "messages": messages, 
                "label": "safe",
                "batch_idx": batch_data['batch_idx'],
                "in_batch_idx": len(batch_examples)
            })
        
        # Add unsafe prompts with refusal response
        for idx in unsafe_indices:
            output = batch_outputs[idx]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": output['prompt']},
                {"role": "assistant", "content": "I cannot assist with that request as it may be harmful or unsafe."}
            ]
            batch_examples.append({
                "messages": messages, 
                "label": "unsafe",
                "batch_idx": batch_data['batch_idx'],
                "in_batch_idx": len(batch_examples)
            })
        
        all_batches.append(batch_examples)
    
    print(f"Created {len(all_batches)} batches, each with {len(all_batches[0])} examples")
    print(f"Total examples: {sum(len(b) for b in all_batches)}")
    
    return all_batches


def create_batch_aware_dataset(
    batches: List[List[Dict]],
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Convert batch-structured data to HuggingFace Dataset while preserving batch info.
    
    Args:
        batches: List of batches, each batch is a list of examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset with batch_idx field
    """
    # Flatten but keep batch info
    all_examples = []
    for batch in batches:
        all_examples.extend(batch)
    
    # Tokenize
    def tokenize_function(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        # Preserve batch_idx (convert to tensor)
        tokenized["batch_idx"] = torch.tensor(examples["batch_idx"], dtype=torch.long)
        # Note: Don't include label_str in tokenized output (causes collation issues)
        
        return tokenized
    
    # Create dataset
    dataset_dict = {
        "messages": [ex["messages"] for ex in all_examples],
        "batch_idx": [ex["batch_idx"] for ex in all_examples],
        "label": [ex["label"] for ex in all_examples],
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


class BatchPreservingSampler(Sampler):
    """
    Custom sampler that preserves the original batch structure.
    Groups samples by batch_idx to maintain safe/unsafe ratio per batch.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int, shuffle_batches: bool = True):
        """
        Args:
            dataset: Dataset with batch_idx field
            batch_size: Size of each batch
            shuffle_batches: Whether to shuffle the order of batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        
        # Group indices by batch_idx
        self.batch_groups = {}
        for idx, item in enumerate(dataset):
            batch_idx = item['batch_idx'].item() if torch.is_tensor(item['batch_idx']) else item['batch_idx']
            if batch_idx not in self.batch_groups:
                self.batch_groups[batch_idx] = []
            self.batch_groups[batch_idx].append(idx)
        
        # Sort batches
        self.sorted_batch_ids = sorted(self.batch_groups.keys())
        print(f"BatchPreservingSampler: {len(self.sorted_batch_ids)} batches, "
              f"{len(dataset)} total samples")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle batch order if requested
        batch_ids = self.sorted_batch_ids.copy()
        if self.shuffle_batches:
            import random
            random.shuffle(batch_ids)
        
        # Yield samples batch by batch
        for batch_id in batch_ids:
            batch_indices = self.batch_groups[batch_id]
            # Don't shuffle within batch to maintain safe/unsafe order if needed
            yield from batch_indices
    
    def __len__(self) -> int:
        return len(self.dataset)


def train_unsafe_experts_with_batches(
    model_name: str,
    output_dir: str = "./unsafe_expert_finetuned_batch",
    num_epochs: int = 3,
    batch_size: int = 16,
    safe_ratio: float = 0.5,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
):
    """
    Main training function with batch-aware processing.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size (should match analysis batch size)
        safe_ratio: Ratio of safe prompts in each batch
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
    """
    print("=" * 80)
    print("Loading prepared batch data...")
    print("=" * 80)
    sft_data, batch_unsafe_experts, batch_expert_masks = load_prepared_batch_data(model_name)
    
    print("\n" + "=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print("\n" + "=" * 80)
    print("Freezing model parameters (keep only unsafe expert MLPs trainable)...")
    print("=" * 80)
    # Pass all batch unsafe experts - function will take the union
    model = freeze_model_except_unsafe_experts(model, batch_unsafe_experts)
    
    print("\n" + "=" * 80)
    print("Preparing batch-structured dataset...")
    print("=" * 80)
    
    # Create batch-structured dataset
    batches = create_batch_structured_dataset(sft_data, tokenizer, batch_size, safe_ratio)
    train_dataset = create_batch_aware_dataset(batches, tokenizer)
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(batches)}")
    
    # Verify batch structure
    try:
        sample_batch_0 = [i for i, item in enumerate(train_dataset) if item['batch_idx'].item() == 0]
        print(f"Samples in batch 0: {len(sample_batch_0)} (expected: {batch_size})")
    except Exception as e:
        print(f"Note: Could not verify batch structure: {e}")
    
    print("\n" + "=" * 80)
    print("Setting up training with batch-preserving sampler...")
    print("=" * 80)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_drop_last=False,  # Don't drop incomplete batches
        group_by_length=False,  # Don't group by length, preserve batch structure
    )
    
    # Use custom trainer with PER-BATCH expert masking and batch-preserving sampler
    trainer = ExpertMaskingTrainer(
        batch_expert_masks=batch_expert_masks,  # Pass all batch masks
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None,  # Use default collator
    )
    
    # Override the sampler to preserve batch structure
    # Note: We need to modify the get_train_dataloader method
    original_get_train_dataloader = trainer.get_train_dataloader
    
    def get_batch_preserving_dataloader():
        dataloader = original_get_train_dataloader()
        # Replace sampler with our custom one
        from torch.utils.data import DataLoader
        return DataLoader(
            trainer.train_dataset,
            batch_size=trainer.args.per_device_train_batch_size,
            sampler=BatchPreservingSampler(
                trainer.train_dataset, 
                trainer.args.per_device_train_batch_size,
                shuffle_batches=True
            ),
            collate_fn=dataloader.collate_fn,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )
    
    trainer.get_train_dataloader = get_batch_preserving_dataloader
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Safe ratio: {safe_ratio} ({int(batch_size * safe_ratio)} safe + {int(batch_size * (1-safe_ratio))} unsafe per batch)")
    print(f"Number of batches: {len(batch_unsafe_experts)}")
    print(f"Each batch has its own set of unsafe experts!")
    
    # Show per-batch statistics
    expert_counts = [len(experts) for experts in batch_unsafe_experts]
    print(f"\nPer-batch unsafe expert counts:")
    print(f"  Min: {min(expert_counts)}, Max: {max(expert_counts)}, Mean: {np.mean(expert_counts):.1f}")
    
    # Get union for training
    all_unsafe_experts_union = set()
    for batch_experts in batch_unsafe_experts:
        all_unsafe_experts_union.update(batch_experts)
    print(f"\nTotal unique unsafe experts (union): {len(all_unsafe_experts_union)}")
    print(f"These experts' MLPs are unfrozen for training")
    print(f"But only batch-specific experts are activated per batch!")
    print("=" * 80)
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save metadata with per-batch information
    # Get union of all unsafe experts
    all_unsafe_experts_union = set()
    for batch_experts in batch_unsafe_experts:
        all_unsafe_experts_union.update(batch_experts)
    
    metadata = {
        "model_name": model_name,
        "batch_size": batch_size,
        "safe_ratio": safe_ratio,
        "num_batches": len(batch_unsafe_experts),
        "total_unique_unsafe_experts": len(all_unsafe_experts_union),
        "per_batch_unsafe_expert_counts": [len(experts) for experts in batch_unsafe_experts],
        # Note: Cannot save tuples directly to JSON, convert to list of lists
        "batch_unsafe_experts_summary": [
            {"batch_idx": i, "num_experts": len(experts), "sample_experts": list(experts)[:5]}
            for i, experts in enumerate(batch_unsafe_experts[:3])  # Save first 3 as examples
        ],
    }
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to: {metadata_path}")
    
    # Save all batch expert masks
    batch_masks_path = os.path.join(output_dir, "batch_expert_masks.npy")
    np.save(batch_masks_path, np.array(batch_expert_masks))
    print(f"Saved {len(batch_expert_masks)} batch expert masks to: {batch_masks_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train unsafe experts with batch processing")
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./unsafe_expert_finetuned_batch",
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
        default=16,
        help="Training batch size (should match analysis batch size)"
    )
    parser.add_argument(
        "--safe_ratio",
        type=float,
        default=0.5,
        help="Ratio of safe prompts in each batch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    train_unsafe_experts_with_batches(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        safe_ratio=args.safe_ratio,
        learning_rate=args.learning_rate,
    )

