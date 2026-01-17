# Training script with gradient-based unsafe expert identification
# 
# NEW STRATEGY:
#   Instead of using routing weights to identify unsafe experts,
#   we use gradient magnitude:
  
#   1. Forward unsafe prompt with safe response
#   2. Compute loss and backward to get gradients
#   3. Calculate gradient magnitude for each expert
#   4. Rank experts by gradient magnitude
#   5. Keep only top-N experts' gradients, zero out others
#   6. Update only those top-N unsafe experts
#   7. Record identified unsafe experts
#   8. Then optimize router consistency (optional)
#
# ADVANTAGES:
#   - Gradient magnitude directly measures which experts need to change most
#   - More direct identification of experts involved in unsafe behavior
#   - No need to run both safe and unsafe prompts separately

import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from typing import List, Dict, Tuple, Optional
from torch.nn import functional as F
from dataclasses import dataclass
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy


@dataclass
class DataCollatorWithLabelIds:
    """Custom data collator that handles label_ids (1D) separately from other fields (2D)."""
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


def load_unsafe_prompts_with_safe_responses(
    dataset_path: str,
    max_samples: int = 128
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load unsafe prompts and prepare safe responses for them.
    
    Args:
        dataset_path: Path to dataset JSON file
        max_samples: Maximum number of samples to use
        
    Returns:
        Tuple of (unsafe_prompts, safe_prompts) where each is a list of message dicts
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} examples from {dataset_path}")
    
    # Filter dataset based on judge success
    filtered_dataset = []
    for item in dataset:
        if item.get("judge_success_gpt4", 0) == 1:
            filtered_dataset.append(item)
    
    filtered_dataset = filtered_dataset[:max_samples]
    print(f"Using {len(filtered_dataset)} unsafe prompts")
    
    unsafe_prompts = []
    safe_prompts = []
    
    for item in filtered_dataset:
        # Unsafe prompt: use complete jailbreak attack
        unsafe_messages = item["all_prompt"]
        unsafe_prompts.append(unsafe_messages)
        
        # Safe prompt: use goal with simple user prompt
        goal = item["goal"]
        safe_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": goal}
        ]
        safe_prompts.append(safe_messages)
    
    return unsafe_prompts, safe_prompts


def create_training_dataset(
    unsafe_prompts: List[List[Dict]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    safe_response: str = "I cannot assist with that request as it may be harmful or unsafe."
) -> Dataset:
    """
    Create training dataset with unsafe prompts paired with safe responses.
    
    Args:
        unsafe_prompts: List of unsafe message lists
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        safe_response: Safe refusal response to train on
        
    Returns:
        HuggingFace Dataset
    """
    training_examples = []
    
    for unsafe_messages in unsafe_prompts:
        # Add safe response to unsafe prompt
        messages = unsafe_messages + [{"role": "assistant", "content": safe_response}]
        training_examples.append({
            "messages": messages,
            "label": "unsafe"  # All are unsafe prompts
        })
    
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
        
        return tokenized
    
    dataset_dict = {
        "messages": [ex["messages"] for ex in training_examples],
        "label": [ex["label"] for ex in training_examples],
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def get_expert_gradients(model: AutoModelForCausalLM) -> Dict[Tuple[int, int], float]:
    """
    Calculate gradient magnitude for each expert in the model.
    
    Args:
        model: Model with computed gradients
        
    Returns:
        Dict mapping (layer_idx, expert_idx) -> gradient magnitude
    """
    expert_gradients = {}
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # OLMoE architecture: mlp.experts
                if hasattr(mlp, "experts"):
                    for expert_idx, expert in enumerate(mlp.experts):
                        total_grad = 0.0
                        num_params = 0
                        
                        for param in expert.parameters():
                            if param.grad is not None:
                                total_grad += param.grad.abs().sum().item()
                                num_params += param.numel()
                        
                        # Average gradient magnitude
                        avg_grad = total_grad / num_params if num_params > 0 else 0.0
                        expert_gradients[(layer_idx, expert_idx)] = avg_grad
                
                # Mixtral architecture: mlp.block_sparse_moe.experts
                elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "experts"):
                    for expert_idx, expert in enumerate(mlp.block_sparse_moe.experts):
                        total_grad = 0.0
                        num_params = 0
                        
                        for param in expert.parameters():
                            if param.grad is not None:
                                total_grad += param.grad.abs().sum().item()
                                num_params += param.numel()
                        
                        # Average gradient magnitude
                        avg_grad = total_grad / num_params if num_params > 0 else 0.0
                        expert_gradients[(layer_idx, expert_idx)] = avg_grad
    
    return expert_gradients


def identify_top_k_unsafe_experts(
    expert_gradients: Dict[Tuple[int, int], float],
    top_k: int
) -> List[Tuple[int, int]]:
    """
    Identify top-k experts with largest gradients.
    
    Args:
        expert_gradients: Dict mapping (layer, expert) -> gradient magnitude
        top_k: Number of top experts to select
        
    Returns:
        List of (layer, expert) tuples for top-k experts
    """
    # Sort by gradient magnitude
    sorted_experts = sorted(expert_gradients.items(), key=lambda x: x[1], reverse=True)
    
    # Take top k
    top_experts = [expert for expert, grad in sorted_experts[:top_k]]
    
    return top_experts


def zero_out_safe_expert_gradients(
    model: AutoModelForCausalLM,
    unsafe_experts: List[Tuple[int, int]]
) -> int:
    """
    Zero out gradients for all experts except the identified unsafe ones.
    
    Args:
        model: Model with computed gradients
        unsafe_experts: List of (layer, expert) tuples to keep
        
    Returns:
        Number of experts whose gradients were zeroed
    """
    unsafe_experts_set = set(unsafe_experts)
    zeroed_count = 0
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # OLMoE architecture
                if hasattr(mlp, "experts"):
                    for expert_idx, expert in enumerate(mlp.experts):
                        if (layer_idx, expert_idx) not in unsafe_experts_set:
                            # Zero out gradients for safe experts
                            for param in expert.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                            zeroed_count += 1
                
                # Mixtral architecture
                elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "experts"):
                    for expert_idx, expert in enumerate(mlp.block_sparse_moe.experts):
                        if (layer_idx, expert_idx) not in unsafe_experts_set:
                            # Zero out gradients for safe experts
                            for param in expert.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                            zeroed_count += 1
    
    return zeroed_count


def freeze_all_except_experts(model: AutoModelForCausalLM):
    """Freeze all parameters except expert MLPs."""
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze all expert MLPs
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                if hasattr(mlp, "experts"):
                    for expert in mlp.experts:
                        for param in expert.parameters():
                            param.requires_grad = True
                
                elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "experts"):
                    for expert in mlp.block_sparse_moe.experts:
                        for param in expert.parameters():
                            param.requires_grad = True


def freeze_all_except_routers(model: AutoModelForCausalLM) -> int:
    """Freeze all parameters except router (gate) parameters. Returns number of unfrozen routers."""
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
    
    return unfrozen_routers


def register_router_hooks(model: AutoModelForCausalLM, storage: List) -> int:
    """Register forward hooks to capture router logits. Returns number of hooks registered."""
    hook_count = 0
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                
                # OLMoE architecture
                if hasattr(mlp, "gate"):
                    def make_hook(layer_idx):
                        def hook_fn(module, input, output):
                            logits = output.clone() if torch.is_tensor(output) else output[0].clone()
                            storage.append({
                                'layer_idx': layer_idx,
                                'logits': logits
                            })
                        return hook_fn
                    
                    mlp.gate.register_forward_hook(make_hook(layer_idx))
                    hook_count += 1
                
                # Mixtral architecture
                elif hasattr(mlp, "block_sparse_moe") and hasattr(mlp.block_sparse_moe, "gate"):
                    def make_hook(layer_idx):
                        def hook_fn(module, input, output):
                            logits = output.clone() if torch.is_tensor(output) else output[0].clone()
                            storage.append({
                                'layer_idx': layer_idx,
                                'logits': logits
                            })
                        return hook_fn
                    
                    mlp.block_sparse_moe.gate.register_forward_hook(make_hook(layer_idx))
                    hook_count += 1
    
    return hook_count


def extract_router_probs_from_storage(
    router_logits_storage: List,
    batch_size: int,
    device: torch.device
) -> Dict[int, torch.Tensor]:
    """
    Extract and process router probabilities from storage.
    
    Returns:
        Dict mapping layer_idx -> averaged router probabilities [batch_size, num_experts]
    """
    layer_probs = {}
    
    # Group router logits by layer
    layer_logits = {}
    for item in router_logits_storage:
        layer_idx = item['layer_idx']
        logits = item['logits']
        
        if layer_idx not in layer_logits:
            layer_logits[layer_idx] = []
        layer_logits[layer_idx].append(logits)
    
    # Process each layer
    for layer_idx, logits_list in layer_logits.items():
        all_logits = torch.cat(logits_list, dim=0)  # [batch_size * seq_len, num_experts]
        
        seq_len = all_logits.shape[0] // batch_size
        
        # Reshape and average over sequence: [batch_size, seq_len, num_experts] -> [batch_size, num_experts]
        all_logits = all_logits.reshape(batch_size, seq_len, -1).mean(dim=1)
        
        # Convert to probabilities
        probs = F.softmax(all_logits, dim=-1)  # [batch_size, num_experts]
        
        layer_probs[layer_idx] = probs.to(device)
    
    return layer_probs


def compute_router_consistency_loss(
    safe_router_probs: Dict[int, torch.Tensor],
    unsafe_router_probs: Dict[int, torch.Tensor],
    device: torch.device,
    kl_type: str = "forward"
) -> torch.Tensor:
    """
    Compute KL divergence between safe (target) and unsafe router distributions.
    
    Args:
        safe_router_probs: Dict[layer_idx -> probs], detached, as target
        unsafe_router_probs: Dict[layer_idx -> probs], trainable
        device: Device for computation
        kl_type: Type of KL divergence
        
    Returns:
        KL divergence loss
    """
    if len(safe_router_probs) == 0 or len(unsafe_router_probs) == 0:
        return torch.tensor(0.0, device=device)
    
    total_kl_loss = torch.tensor(0.0, device=device)
    num_layers = 0
    
    # Compute KL divergence for each layer
    for layer_idx in safe_router_probs.keys():
        if layer_idx not in unsafe_router_probs:
            continue
        
        safe_probs = safe_router_probs[layer_idx]  # [n_safe, num_experts], detached
        unsafe_probs = unsafe_router_probs[layer_idx]  # [n_unsafe, num_experts], trainable
        
        # Compute average distributions
        safe_avg_probs = safe_probs.mean(dim=0)  # [num_experts]
        unsafe_avg_probs = unsafe_probs.mean(dim=0)  # [num_experts]
        
        # Add epsilon for stability
        eps = 1e-8
        safe_avg_probs = safe_avg_probs + eps
        unsafe_avg_probs = unsafe_avg_probs + eps
        safe_avg_probs = safe_avg_probs / safe_avg_probs.sum()
        unsafe_avg_probs = unsafe_avg_probs / unsafe_avg_probs.sum()
        
        # Compute KL divergence
        if kl_type == "forward":
            kl_div = F.kl_div(unsafe_avg_probs.log(), safe_avg_probs, reduction='batchmean')
        elif kl_type == "reverse":
            kl_div = F.kl_div(safe_avg_probs.log(), unsafe_avg_probs, reduction='batchmean')
        else:  # symmetric
            kl_forward = F.kl_div(unsafe_avg_probs.log(), safe_avg_probs, reduction='batchmean')
            kl_reverse = F.kl_div(safe_avg_probs.log(), unsafe_avg_probs, reduction='batchmean')
            kl_div = (kl_forward + kl_reverse) / 2.0
        
        total_kl_loss += kl_div.to(device)
        num_layers += 1
    
    if num_layers > 0:
        total_kl_loss = total_kl_loss / num_layers
    
    return total_kl_loss


def train_one_round_gradient_based(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    round_idx: int,
    top_k_experts: int = 100,
    accumulate_gradients: bool = True
) -> Tuple[float, List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Train for one round using gradient-based expert identification.
    
    Strategy:
        1. Forward pass on unsafe prompts with safe responses
        2. Compute loss and backward
        3. Calculate gradient magnitude for each expert
        4. Identify top-k experts with largest gradients
        5. Zero out gradients for other experts
        6. Update only the top-k experts
        7. Return identified unsafe experts
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        round_idx: Current round index
        top_k_experts: Number of top experts to identify and update
        accumulate_gradients: If True, accumulate gradients across batches before selecting top-k
        
    Returns:
        Tuple of (average_loss, identified_unsafe_experts, expert_gradients_dict)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Storage for gradients across batches
    accumulated_expert_gradients = {}
    
    progress_bar = tqdm(dataloader, desc=f"Round {round_idx} - Gradient-based Expert Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Remove label_ids (not needed for forward)
        inputs = {k: v for k, v in batch.items() if k not in ['label_ids']}
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Get expert gradients for this batch
        batch_expert_gradients = get_expert_gradients(model)
        
        if accumulate_gradients:
            # Accumulate gradients across batches
            for expert_key, grad_value in batch_expert_gradients.items():
                if expert_key not in accumulated_expert_gradients:
                    accumulated_expert_gradients[expert_key] = []
                accumulated_expert_gradients[expert_key].append(grad_value)
        else:
            # Identify top-k unsafe experts for THIS BATCH
            unsafe_experts_batch = identify_top_k_unsafe_experts(batch_expert_gradients, top_k_experts)
            
            # Zero out safe expert gradients
            zeroed_count = zero_out_safe_expert_gradients(model, unsafe_experts_batch)
            
            # Update only unsafe experts
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'batch': f"{batch_idx + 1}/{len(dataloader)}"
        })
    
    # After all batches, identify top-k experts globally (if accumulating)
    if accumulate_gradients:
        # Average gradients across batches for each expert
        averaged_expert_gradients = {
            expert_key: np.mean(grad_list)
            for expert_key, grad_list in accumulated_expert_gradients.items()
        }
        
        # Identify top-k unsafe experts globally
        unsafe_experts_global = identify_top_k_unsafe_experts(averaged_expert_gradients, top_k_experts)
        
        print(f"\n[Round {round_idx}] Identified {len(unsafe_experts_global)} top-k unsafe experts globally")
        print(f"  Top 5 experts by gradient magnitude:")
        sorted_experts = sorted(averaged_expert_gradients.items(), key=lambda x: x[1], reverse=True)
        for i, ((layer, expert), grad) in enumerate(sorted_experts[:5]):
            print(f"    {i+1}. Layer {layer}, Expert {expert}: {grad:.6f}")
        
        final_expert_gradients = averaged_expert_gradients
        final_unsafe_experts = unsafe_experts_global
    else:
        # Return the last batch's results (not ideal, but for consistency)
        final_expert_gradients = batch_expert_gradients
        final_unsafe_experts = unsafe_experts_batch
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, final_unsafe_experts, final_expert_gradients


def train_one_round_router(
    model: AutoModelForCausalLM,
    safe_prompts: List[List[Dict]],
    unsafe_prompts: List[List[Dict]],
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    round_idx: int,
    router_logits_storage: List,
    batch_size: int = 8,
    kl_type: str = "forward"
) -> float:
    """
    Train router for one round using consistency loss.
    
    Strategy:
        1. Forward safe prompts (no grad) to get target router distributions
        2. Forward unsafe prompts (with grad) to get trainable router distributions
        3. Minimize KL divergence to make unsafe routing match safe routing
    
    Args:
        model: Model to train
        safe_prompts: List of safe prompt messages
        unsafe_prompts: List of unsafe prompt messages
        tokenizer: Tokenizer
        optimizer: Optimizer
        device: Device
        round_idx: Current round index
        router_logits_storage: Storage for router logits (via hooks)
        batch_size: Batch size for router training
        kl_type: Type of KL divergence
        
    Returns:
        Average router loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create mini-batches
    num_samples = min(len(safe_prompts), len(unsafe_prompts))
    num_batches_total = (num_samples + batch_size - 1) // batch_size
    
    progress_bar = tqdm(range(num_batches_total), desc=f"Round {round_idx} - Router Training")
    
    for batch_idx in progress_bar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        safe_batch = safe_prompts[start_idx:end_idx]
        unsafe_batch = unsafe_prompts[start_idx:end_idx]
        
        # Tokenize safe prompts
        safe_texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 
                      for msgs in safe_batch]
        safe_inputs = tokenizer(safe_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        safe_inputs = {k: v.to(device) for k, v in safe_inputs.items()}
        
        # Tokenize unsafe prompts
        unsafe_texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 
                        for msgs in unsafe_batch]
        unsafe_inputs = tokenizer(unsafe_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        unsafe_inputs = {k: v.to(device) for k, v in unsafe_inputs.items()}
        
        # Forward safe prompts (no grad) - get target distributions
        router_logits_storage.clear()
        with torch.no_grad():
            _ = model(**safe_inputs)
        safe_router_probs = extract_router_probs_from_storage(
            router_logits_storage, len(safe_batch), device
        )
        
        # Forward unsafe prompts (with grad) - trainable
        router_logits_storage.clear()
        _ = model(**unsafe_inputs)
        unsafe_router_probs = extract_router_probs_from_storage(
            router_logits_storage, len(unsafe_batch), device
        )
        
        # Compute router consistency loss
        router_loss = compute_router_consistency_loss(
            safe_router_probs, unsafe_router_probs, device, kl_type
        )
        
        if router_loss.item() > 0:
            # Backward and update
            optimizer.zero_grad()
            router_loss.backward()
            optimizer.step()
            
            total_loss += router_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'router_loss': router_loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_gradient_based_identification(
    model_name: str,
    dataset_path: str,
    output_dir: str = "./gradient_based_training",
    num_rounds: int = 3,
    top_k_experts: int = 100,
    batch_size: int = 8,
    expert_lr: float = 5e-5,
    router_lr: float = 1e-4,
    max_samples: int = 128,
    skip_router_training: bool = False,
    kl_type: str = "forward",
):
    """
    Main training function with gradient-based unsafe expert identification.
    
    NEW STRATEGY:
        For each round:
        1. Use unsafe prompts with safe responses
        2. Forward and backward to get gradients
        3. Rank experts by gradient magnitude
        4. Keep only top-k experts' gradients
        5. Update those experts
        6. Record identified unsafe experts
        7. (Optional) Optimize router to match safe routing patterns
    
    Args:
        model_name: HuggingFace model name
        dataset_path: Path to dataset JSON file
        output_dir: Output directory for checkpoints
        num_rounds: Number of training rounds
        top_k_experts: Number of top experts to identify and update per round
        batch_size: Training batch size
        expert_lr: Learning rate for expert training
        router_lr: Learning rate for router training
        max_samples: Maximum number of samples to use
        skip_router_training: If True, skip router training
        kl_type: Type of KL divergence for router training
    """
    print("=" * 80)
    print("GRADIENT-BASED UNSAFE EXPERT IDENTIFICATION")
    print(f"  Model: {model_name}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Top-K Experts per Round: {top_k_experts}")
    print(f"  Expert LR: {expert_lr}, Router LR: {router_lr}")
    if skip_router_training:
        print(f"  Router Training: SKIPPED")
    else:
        print(f"  Router Training: ENABLED (KL type: {kl_type})")
    print("=" * 80)
    
    # Load data
    print("\nLoading unsafe prompts with safe responses...")
    unsafe_prompts, safe_prompts = load_unsafe_prompts_with_safe_responses(dataset_path, max_samples)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    device = next(model.parameters()).device
    
    # Prepare dataset
    print("\nPreparing training dataset...")
    train_dataset = create_training_dataset(unsafe_prompts, tokenizer)
    
    # Create dataloader
    data_collator = DataCollatorWithLabelIds(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Register router hooks (for router training)
    router_logits_storage = []
    if not skip_router_training:
        num_hooks = register_router_hooks(model, router_logits_storage)
        print(f"Registered {num_hooks} router hooks")
    
    # Training loop
    training_logs = []
    all_identified_experts = {}  # Store identified experts per round
    
    for round_idx in range(num_rounds):
        print("\n" + "=" * 80)
        print(f"ROUND {round_idx + 1}/{num_rounds}")
        print("=" * 80)
        
        # ====================
        # Step 1: Gradient-based expert identification and training
        # ====================
        print(f"\n[Round {round_idx + 1}] Step 1: Gradient-based expert identification...")
        
        # Unfreeze all experts (we'll select via gradient masking)
        freeze_all_except_experts(model)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
        
        # Create optimizer
        expert_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=expert_lr
        )
        
        # Train with gradient-based selection
        expert_loss, identified_experts, expert_gradients = train_one_round_gradient_based(
            model, dataloader, expert_optimizer, device, round_idx + 1,
            top_k_experts=top_k_experts, accumulate_gradients=False
        )
        
        print(f"  [Round {round_idx + 1}] Expert Training: Loss = {expert_loss:.4f}")
        print(f"  [Round {round_idx + 1}] Identified {len(identified_experts)} unsafe experts")
        
        # Store identified experts
        all_identified_experts[round_idx + 1] = {
            'experts': identified_experts,
            'gradients': expert_gradients,
        }
        
        training_logs.append({
            'round': round_idx + 1,
            'step': 'expert',
            'loss': expert_loss,
            'num_identified_experts': len(identified_experts)
        })
        
        # ====================
        # Step 2: Router training (optional)
        # ====================
        if not skip_router_training:
            print(f"\n[Round {round_idx + 1}] Step 2: Router consistency training...")
            
            # Freeze everything except routers
            unfrozen_routers = freeze_all_except_routers(model)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Unfrozen routers: {unfrozen_routers}")
            print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
            
            # Create optimizer for routers
            router_optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=router_lr
            )
            
            # Train router
            router_loss = train_one_round_router(
                model, safe_prompts, unsafe_prompts, tokenizer, router_optimizer,
                device, round_idx + 1, router_logits_storage, batch_size, kl_type
            )
            
            print(f"  [Round {round_idx + 1}] Router Training: Loss = {router_loss:.4f}")
            
            training_logs.append({
                'round': round_idx + 1,
                'step': 'router',
                'loss': router_loss
            })
        else:
            print(f"\n[Round {round_idx + 1}] Step 2: Router training SKIPPED")
        
        # Save checkpoint after each round
        round_dir = os.path.join(output_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        model.save_pretrained(round_dir)
        tokenizer.save_pretrained(round_dir)
        print(f"\n  Saved checkpoint to {round_dir}")
    
    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    final_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training logs and metadata
    metadata = {
        "model_name": model_name,
        "training_strategy": "gradient_based_identification",
        "num_rounds": num_rounds,
        "top_k_experts": top_k_experts,
        "batch_size": batch_size,
        "expert_lr": expert_lr,
        "router_lr": router_lr if not skip_router_training else None,
        "kl_type": kl_type if not skip_router_training else None,
        "skip_router_training": skip_router_training,
        "max_samples": max_samples,
        "training_logs": training_logs,
        "identified_experts_per_round": {
            round_num: {
                'experts': [(int(l), int(e)) for l, e in data['experts']],
                'gradients': {f"L{l}_E{e}": float(g) for (l, e), g in list(data['gradients'].items())[:10]}  # Save top 10
            }
            for round_num, data in all_identified_experts.items()
        }
    }
    
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Save full expert gradient data
    expert_data_path = os.path.join(output_dir, "expert_gradients.pkl")
    pd.to_pickle(all_identified_experts, expert_data_path)
    print(f"Saved expert gradient data to {expert_data_path}")
    
    print("\n" + "=" * 80)
    print("Gradient-based training complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with gradient-based unsafe expert identification")
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/OLMoE-1B-7B-0125-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/stufs1/jiachliang/FlipAttack/reproduce_result/FlipAttack-FWO-CoT-LangGPT-Few-shot-Qwen3-30B-A3B-advbench-0_519-final.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gradient_based_training",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of training rounds"
    )
    parser.add_argument(
        "--top_k_experts",
        type=int,
        default=100,
        help="Number of top experts to identify and update per round"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--expert_lr",
        type=float,
        default=5e-5,
        help="Learning rate for expert training"
    )
    parser.add_argument(
        "--router_lr",
        type=float,
        default=1e-6,
        help="Learning rate for router training"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=128,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--skip_router_training",
        action="store_true",
        help="Skip router training, only train experts"
    )
    parser.add_argument(
        "--kl_type",
        type=str,
        default="forward",
        choices=["forward", "reverse", "symmetric"],
        help="Type of KL divergence for router consistency"
    )
    
    args = parser.parse_args()
    
    train_gradient_based_identification(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        top_k_experts=args.top_k_experts,
        batch_size=args.batch_size,
        expert_lr=args.expert_lr,
        router_lr=args.router_lr,
        max_samples=args.max_samples,
        skip_router_training=args.skip_router_training,
        kl_type=args.kl_type,
    )
