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
from typing import List, Dict, Tuple, Iterator, Optional
from torch.nn import functional as F
from torch.utils.data import Sampler
from dataclasses import dataclass
import json


@dataclass
class DataCollatorWithLabelIds:
    """
    Custom data collator that handles label_ids (1D) separately from other fields (2D).
    """
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract label_ids separately
        label_ids = None
        if 'label_ids' in features[0]:
            label_ids = torch.tensor([f['label_ids'].item() if torch.is_tensor(f['label_ids']) else f['label_ids'] 
                                     for f in features], dtype=torch.long)
            # Remove label_ids from features for default collation
            features = [{k: v for k, v in f.items() if k != 'label_ids'} for f in features]
        
        # Use default collator for other fields
        from transformers import default_data_collator
        batch = default_data_collator(features)
        
        # Add label_ids back
        if label_ids is not None:
            batch['label_ids'] = label_ids
        
        return batch


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
    Supports optional router consistency loss between safe and unsafe samples.
    """
    
    def __init__(
        self, 
        batch_expert_masks, 
        use_router_consistency_loss=False,
        router_consistency_weight=0.1,
        router_consistency_type="symmetric",
        router_consistency_layers="all",
        *args, 
        **kwargs
    ):
        """
        Args:
            batch_expert_masks: List of expert masks, one per batch
                                Each mask is (num_layers, n_experts) array
            use_router_consistency_loss: Whether to use router consistency loss
            router_consistency_weight: Weight for router consistency loss
            router_consistency_type: Type of KL divergence ("forward", "reverse", "symmetric")
            router_consistency_layers: Which layers to compute loss on ("all" or "unsafe_only")
        """
        super().__init__(*args, **kwargs)
        # Convert all batch masks to tensors
        self.batch_expert_masks = [
            torch.tensor(mask, dtype=torch.float32)
            for mask in batch_expert_masks
        ]
        print(f"Initialized ExpertMaskingTrainer with {len(self.batch_expert_masks)} batch-specific masks")
        
        # Router consistency loss configuration
        self.use_router_consistency_loss = use_router_consistency_loss
        self.router_consistency_weight = router_consistency_weight
        self.router_consistency_type = router_consistency_type
        self.router_consistency_layers = router_consistency_layers
        
        if self.use_router_consistency_loss:
            print(f"\n{'='*80}")
            print("Router Consistency Loss Configuration:")
            print(f"  Enabled: {self.use_router_consistency_loss}")
            print(f"  Weight: {self.router_consistency_weight}")
            print(f"  Type: {self.router_consistency_type}")
            print(f"  Layers: {self.router_consistency_layers}")
            print(f"{'='*80}\n")
        
        # Storage for router logits (will be populated by forward hooks)
        self.router_logits_storage = []
        self.hooks_registered = False
    
    def _register_router_hooks(self, model):
        """
        Register forward hooks to capture router logits from all MoE layers.
        Supports different MoE architectures (OLMoE, Mixtral, etc.).
        """
        if self.hooks_registered:
            return
        
        print("Registering forward hooks to capture router logits...")
        hook_count = 0
        
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer_idx, layer in enumerate(model.model.layers):
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                    
                    # OLMoE architecture: mlp.gate (router)
                    if hasattr(mlp, "gate"):
                        def make_hook(layer_idx):
                            def hook_fn(module, input, output):
                                # Store router logits WITHOUT detach for gradient flow
                                # Clone to avoid in-place modifications
                                logits = output.clone() if torch.is_tensor(output) else output[0].clone()
                                self.router_logits_storage.append({
                                    'layer_idx': layer_idx,
                                    'logits': logits
                                })
                            return hook_fn
                        
                        mlp.gate.register_forward_hook(make_hook(layer_idx))
                        hook_count += 1
                    
                    # Mixtral architecture: mlp.block_sparse_moe.gate
                    elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "gate"):
                        def make_hook(layer_idx):
                            def hook_fn(module, input, output):
                                # Store router logits WITHOUT detach for gradient flow
                                logits = output.clone() if torch.is_tensor(output) else output[0].clone()
                                self.router_logits_storage.append({
                                    'layer_idx': layer_idx,
                                    'logits': logits
                                })
                            return hook_fn
                        
                        mlp.block_sparse_moe.gate.register_forward_hook(make_hook(layer_idx))
                        hook_count += 1
        
        print(f"Registered {hook_count} router hooks")
        self.hooks_registered = True
    
    def _compute_router_consistency_loss(self, labels):
        """
        Compute KL divergence between safe and unsafe samples' router distributions.
        
        Args:
            labels: Tensor of labels (0 for safe, 1 for unsafe)
            
        Returns:
            KL divergence loss (scalar tensor)
        """
        if len(self.router_logits_storage) == 0:
            return torch.tensor(0.0, device=self.args.device)
        
        # Move labels to the correct device
        labels = labels.to(self.args.device)
        
        # Initialize loss on the target device
        total_kl_loss = torch.tensor(0.0, device=self.args.device)
        num_layers = 0
        
        # Group router logits by layer
        layer_logits = {}
        for item in self.router_logits_storage:
            layer_idx = item['layer_idx']
            logits = item['logits']
            
            if layer_idx not in layer_logits:
                layer_logits[layer_idx] = []
            layer_logits[layer_idx].append(logits)
        
        # Compute KL divergence for each layer
        for layer_idx, logits_list in layer_logits.items():
            # Stack all logits for this layer: [batch_size * seq_len, num_experts]
            all_logits = torch.cat(logits_list, dim=0)
            
            # Average over sequence length to get per-sample distribution
            # Assuming logits shape is [batch_size * seq_len, num_experts]
            # We need to map back to batch samples
            batch_size = len(labels)
            seq_len = all_logits.shape[0] // batch_size
            
            # Reshape and average: [batch_size, seq_len, num_experts] -> [batch_size, num_experts]
            all_logits = all_logits.reshape(batch_size, seq_len, -1).mean(dim=1)
            
            # Ensure labels are on the same device as all_logits
            labels_device = labels.to(all_logits.device)
            
            # Separate safe and unsafe samples (masks are now on the same device as all_logits)
            safe_mask = labels_device == 0  # 0 for safe
            unsafe_mask = labels_device == 1  # 1 for unsafe
            
            if safe_mask.sum() == 0 or unsafe_mask.sum() == 0:
                continue  # Skip if no safe or unsafe samples
            
            safe_logits = all_logits[safe_mask]   # [n_safe, num_experts]
            unsafe_logits = all_logits[unsafe_mask]  # [n_unsafe, num_experts]
            
            # Convert to probabilities
            safe_probs = F.softmax(safe_logits, dim=-1)
            unsafe_probs = F.softmax(unsafe_logits, dim=-1)
            
            # Compute average distributions
            safe_avg_probs = safe_probs.mean(dim=0)  # [num_experts]
            unsafe_avg_probs = unsafe_probs.mean(dim=0)  # [num_experts]
            
            # Add small epsilon for numerical stability
            eps = 1e-8
            safe_avg_probs = safe_avg_probs + eps
            unsafe_avg_probs = unsafe_avg_probs + eps
            safe_avg_probs = safe_avg_probs / safe_avg_probs.sum()
            unsafe_avg_probs = unsafe_avg_probs / unsafe_avg_probs.sum()
            
            # Compute KL divergence
            if self.router_consistency_type == "forward":
                # KL(safe || unsafe): push unsafe to match safe
                kl_div = F.kl_div(
                    unsafe_avg_probs.log(), 
                    safe_avg_probs, 
                    reduction='batchmean'
                )
            elif self.router_consistency_type == "reverse":
                # KL(unsafe || safe): push safe to match unsafe
                kl_div = F.kl_div(
                    safe_avg_probs.log(), 
                    unsafe_avg_probs, 
                    reduction='batchmean'
                )
            else:  # symmetric
                kl_forward = F.kl_div(
                    unsafe_avg_probs.log(), 
                    safe_avg_probs, 
                    reduction='batchmean'
                )
                kl_reverse = F.kl_div(
                    safe_avg_probs.log(), 
                    unsafe_avg_probs, 
                    reduction='batchmean'
                )
                kl_div = (kl_forward + kl_reverse) / 2.0
            
            # Move kl_div to the target device before accumulation (handles multi-GPU)
            total_kl_loss += kl_div.to(self.args.device)
            num_layers += 1
        
        # Average over layers
        if num_layers > 0:
            total_kl_loss = total_kl_loss / num_layers
        
        return total_kl_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to dynamically mask safe experts based on batch_idx.
        Optionally adds router consistency loss between safe and unsafe samples.
        
        Args:
            num_items_in_batch: Number of items in the batch (added in newer transformers versions)
        """
        # Register hooks on first call
        if self.use_router_consistency_loss and not self.hooks_registered:
            self._register_router_hooks(model)
        
        # Clear router logits storage from previous batch
        self.router_logits_storage = []
        
        # Get batch_idx and labels from inputs
        batch_idx = None
        labels_str = None
        
        if 'batch_idx' in inputs:
            batch_indices = inputs['batch_idx']
            unique_batch_idx = torch.unique(batch_indices)
            
            if len(unique_batch_idx) == 1:
                batch_idx = unique_batch_idx.item()
                current_mask = self.batch_expert_masks[batch_idx].to(self.args.device)
                self.current_expert_mask = current_mask
            else:
                print(f"Warning: Mixed batch_idx in training batch: {unique_batch_idx.tolist()}")
        
        # Extract labels for router consistency loss (if present)
        if 'label_ids' in inputs:
            labels_str = inputs['label_ids']
            # Remove from inputs before forward pass
            inputs_copy = {k: v for k, v in inputs.items() if k != 'label_ids'}
        else:
            inputs_copy = inputs
        
        # Standard forward pass
        outputs = model(**inputs_copy)
        
        # Get base loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Add router consistency loss if enabled
        if self.use_router_consistency_loss and labels_str is not None:
            router_loss = self._compute_router_consistency_loss(labels_str)
            
            # Add weighted router consistency loss
            if router_loss.item() > 0:
                loss = loss + self.router_consistency_weight * router_loss
                
                # Log the losses (optional - can be verbose)
                if self.state.global_step % 100 == 0:
                    print(f"Step {self.state.global_step}: "
                          f"CE Loss: {loss.item() - self.router_consistency_weight * router_loss.item():.4f}, "
                          f"Router Loss: {router_loss.item():.4f}, "
                          f"Total Loss: {loss.item():.4f}")
        
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
        HuggingFace Dataset with batch_idx and label_ids fields
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
        
        # Convert label strings to numeric IDs for router consistency loss
        # 0 = safe, 1 = unsafe
        label_ids = torch.tensor([0 if label == "safe" else 1 for label in examples["label"]], dtype=torch.long)
        tokenized["label_ids"] = label_ids
        
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
    finetune_unsafe_experts: bool = True,
    use_router_consistency_loss: bool = False,
    router_consistency_weight: float = 0.1,
    router_consistency_type: str = "symmetric",
    router_consistency_layers: str = "all",
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
        finetune_unsafe_experts: Whether to finetune unsafe expert MLPs (default: True)
        use_router_consistency_loss: Whether to use router consistency loss
        router_consistency_weight: Weight for router consistency loss
        router_consistency_type: Type of KL divergence ("forward", "reverse", "symmetric")
        router_consistency_layers: Which layers to compute loss on ("all" or "unsafe_only")
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
    if finetune_unsafe_experts:
        print("Freezing model parameters (keep only unsafe expert MLPs trainable)...")
        print("=" * 80)
        # Pass all batch unsafe experts - function will take the union
        model = freeze_model_except_unsafe_experts(model, batch_unsafe_experts)
    else:
        print("Not finetuning unsafe experts...")
        if use_router_consistency_loss:
            print("Freezing all parameters EXCEPT routers (for router consistency ablation)...")
            print("=" * 80)
            # Freeze everything first
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze router (gate) parameters only
            unfrozen_routers = 0
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                for layer_idx, layer in enumerate(model.model.layers):
                    if hasattr(layer, "mlp"):
                        mlp = layer.mlp
                        
                        # OLMoE architecture: mlp.gate (router)
                        if hasattr(mlp, "gate"):
                            for param in mlp.gate.parameters():
                                param.requires_grad = True
                            unfrozen_routers += 1
                        
                        # Mixtral architecture: mlp.block_sparse_moe.gate
                        elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "gate"):
                            for param in mlp.block_sparse_moe.gate.parameters():
                                param.requires_grad = True
                            unfrozen_routers += 1
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nParameter freezing complete:")
            print(f"  Unfrozen {unfrozen_routers} router (gate) modules")
            print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
            print(f"  Router consistency loss will update router parameters")
        else:
            print("Freezing ALL model parameters (baseline with frozen model)...")
            print("=" * 80)
            # Freeze everything - this is essentially a baseline
            for param in model.parameters():
                param.requires_grad = False
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nParameter freezing complete:")
            print(f"  All parameters frozen")
            print(f"  Trainable: {trainable_params:,} / {total_params:,} (0.00%)")
            print(f"  WARNING: No trainable parameters - this is a frozen baseline!")
    
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
    
    # Create custom data collator to handle label_ids
    data_collator = DataCollatorWithLabelIds(tokenizer=tokenizer)
    
    # Use custom trainer with PER-BATCH expert masking and batch-preserving sampler
    trainer = ExpertMaskingTrainer(
        batch_expert_masks=batch_expert_masks,  # Pass all batch masks
        use_router_consistency_loss=use_router_consistency_loss,
        router_consistency_weight=router_consistency_weight,
        router_consistency_type=router_consistency_type,
        router_consistency_layers=router_consistency_layers,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use custom collator
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
        # Training configuration
        "finetune_unsafe_experts": finetune_unsafe_experts,
        # Router consistency loss configuration
        "use_router_consistency_loss": use_router_consistency_loss,
        "router_consistency_weight": router_consistency_weight if use_router_consistency_loss else None,
        "router_consistency_type": router_consistency_type if use_router_consistency_loss else None,
        "router_consistency_layers": router_consistency_layers if use_router_consistency_loss else None,
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


def train_full_model_baseline(
    model_name: str,
    output_dir: str = "./full_model_baseline",
    num_epochs: int = 3,
    batch_size: int = 16,
    safe_ratio: float = 0.5,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
):
    """
    Baseline training: fine-tune ALL model parameters (not just unsafe experts).
    This is the comparison experiment to see if selective expert training works better.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        safe_ratio: Ratio of safe prompts in each batch
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
    """
    print("=" * 80)
    print("BASELINE EXPERIMENT: Training ALL model parameters")
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
    print("Parameter configuration: ALL PARAMETERS TRAINABLE (Baseline)")
    print("=" * 80)
    # Keep all parameters trainable (default behavior)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} (100.00%)")
    print("This is the baseline - training the full model for comparison")
    
    print("\n" + "=" * 80)
    print("Preparing batch-structured dataset...")
    print("=" * 80)
    
    # Create batch-structured dataset
    batches = create_batch_structured_dataset(sft_data, tokenizer, batch_size, safe_ratio)
    train_dataset = create_batch_aware_dataset(batches, tokenizer)
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(batches)}")
    
    print("\n" + "=" * 80)
    print("Setting up training (using standard Trainer - no expert masking)...")
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
        dataloader_drop_last=False,
        group_by_length=False,
    )
    
    # Create custom data collator (same as selective mode for consistency)
    data_collator = DataCollatorWithLabelIds(tokenizer=tokenizer)
    
    # Use STANDARD Trainer (no expert masking - this is the baseline)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Use same batch-preserving sampler for consistency
    original_get_train_dataloader = trainer.get_train_dataloader
    
    def get_batch_preserving_dataloader():
        dataloader = original_get_train_dataloader()
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
    print("Starting BASELINE training (ALL parameters)...")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Safe ratio: {safe_ratio} ({int(batch_size * safe_ratio)} safe + {int(batch_size * (1-safe_ratio))} unsafe per batch)")
    print(f"Number of batches: {len(batches)}")
    print(f"Training mode: FULL MODEL (all parameters trainable)")
    print("=" * 80)
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Saving final baseline model...")
    print("=" * 80)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "training_mode": "full_model_baseline",
        "batch_size": batch_size,
        "safe_ratio": safe_ratio,
        "num_batches": len(batches),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": 1.0,
        "note": "Baseline experiment - all parameters trained for comparison with selective expert training",
    }
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("BASELINE training complete!")
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
    parser.add_argument(
        "--mode",
        type=str,
        default="selective",
        choices=["selective", "baseline"],
        help="Training mode: 'selective' (only unsafe experts) or 'baseline' (all parameters)"
    )
    parser.add_argument(
        "--finetune_unsafe_experts",
        action="store_true",
        default=True,
        help="Finetune unsafe expert MLPs (default: True). Set to False for router-only ablation."
    )
    parser.add_argument(
        "--no_finetune_unsafe_experts",
        action="store_false",
        dest="finetune_unsafe_experts",
        help="Do NOT finetune unsafe experts (for router-only ablation study)"
    )
    parser.add_argument(
        "--use_router_consistency_loss",
        action="store_true",
        help="Enable router consistency loss (for ablation study)"
    )
    parser.add_argument(
        "--router_consistency_weight",
        type=float,
        default=0.1,
        help="Weight for router consistency loss"
    )
    parser.add_argument(
        "--router_consistency_type",
        type=str,
        default="symmetric",
        choices=["forward", "reverse", "symmetric"],
        help="Type of KL divergence: 'forward' (safe->unsafe), 'reverse' (unsafe->safe), 'symmetric' (bidirectional)"
    )
    parser.add_argument(
        "--router_consistency_layers",
        type=str,
        default="all",
        choices=["all", "unsafe_only"],
        help="Which layers to compute router consistency loss on"
    )
    
    args = parser.parse_args()
    
    if args.mode == "selective":
        print("\n" + "=" * 80)
        print("Running SELECTIVE EXPERT training")
        print(f"  Finetune Unsafe Experts: {args.finetune_unsafe_experts}")
        if args.use_router_consistency_loss:
            print(f"  Router Consistency Loss: ENABLED (weight={args.router_consistency_weight})")
        else:
            print(f"  Router Consistency Loss: DISABLED")
        
        # Determine experiment type
        if not args.finetune_unsafe_experts and not args.use_router_consistency_loss:
            print("\n  WARNING: Neither expert finetuning nor router consistency is enabled!")
            print("  This will be a baseline with frozen model.")
        elif not args.finetune_unsafe_experts and args.use_router_consistency_loss:
            print("\n  ABLATION: Router Consistency ONLY")
        elif args.finetune_unsafe_experts and not args.use_router_consistency_loss:
            print("\n  ABLATION: Expert Finetuning ONLY")
        else:
            print("\n  COMBINED: Expert Finetuning + Router Consistency")
        print("=" * 80)
        
        train_unsafe_experts_with_batches(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            safe_ratio=args.safe_ratio,
            learning_rate=args.learning_rate,
            finetune_unsafe_experts=args.finetune_unsafe_experts,
            use_router_consistency_loss=args.use_router_consistency_loss,
            router_consistency_weight=args.router_consistency_weight,
            router_consistency_type=args.router_consistency_type,
            router_consistency_layers=args.router_consistency_layers,
        )
    elif args.mode == "baseline":
        print("\n" + "=" * 80)
        print("Running BASELINE (FULL MODEL) training")
        print("=" * 80)
        # Modify output_dir to indicate baseline
        baseline_output_dir = args.output_dir.replace("unsafe_expert_finetuned", "full_model_baseline")
        if baseline_output_dir == args.output_dir:
            baseline_output_dir = args.output_dir + "_baseline"
        
        train_full_model_baseline(
            model_name=args.model_name,
            output_dir=baseline_output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            safe_ratio=args.safe_ratio,
            learning_rate=args.learning_rate,
        )

