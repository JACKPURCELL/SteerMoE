#!/bin/bash
# Comparison experiments between routing-based and gradient-based methods
#
# This script runs both methods on the same model and dataset for comparison

# Configuration
MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
DATASET_PATH="/home/stufs1/jiachliang/FlipAttack/reproduce_result/FlipAttack-FWO-CoT-LangGPT-Few-shot-Qwen3-30B-A3B-advbench-0_519-final.json"
NUM_ROUNDS=3
BATCH_SIZE=8
EXPERT_LR=5e-5
ROUTER_LR=1e-4
MAX_SAMPLES=128

# For Qwen model, use smaller batch size
# MODEL_NAME="Qwen/Qwen3-30B-A3B"
# BATCH_SIZE=4

echo "========================================"
echo "Running Comparison Experiments"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Num Rounds: $NUM_ROUNDS"
echo "Batch Size: $BATCH_SIZE"
echo "========================================"

# Experiment 1: Gradient-Based Method (NEW)
echo ""
echo "========================================"
echo "Experiment 1: Gradient-Based Method"
echo "========================================"
python train_gradient_based_expert_identification.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir ./comparison_gradient_based \
    --num_rounds $NUM_ROUNDS \
    --top_k_experts 100 \
    --batch_size $BATCH_SIZE \
    --expert_lr $EXPERT_LR \
    --router_lr $ROUTER_LR \
    --max_samples $MAX_SAMPLES

# Experiment 2: Routing-Based Method (ORIGINAL)
echo ""
echo "========================================"
echo "Experiment 2: Routing-Based Method"
echo "========================================"
python train_alternating_optimization.py \
    --model_name "$MODEL_NAME" \
    --output_dir ./comparison_routing_based \
    --num_rounds $NUM_ROUNDS \
    --epochs_per_round 1 \
    --batch_size 16 \
    --safe_ratio 0.5 \
    --expert_lr $EXPERT_LR \
    --router_lr $ROUTER_LR

# Experiment 3: Gradient-Based (Expert Only)
echo ""
echo "========================================"
echo "Experiment 3: Gradient-Based (Expert Only)"
echo "========================================"
python train_gradient_based_expert_identification.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir ./comparison_gradient_expert_only \
    --num_rounds $NUM_ROUNDS \
    --top_k_experts 100 \
    --batch_size $BATCH_SIZE \
    --expert_lr $EXPERT_LR \
    --max_samples $MAX_SAMPLES \
    --skip_router_training

# Experiment 4: Routing-Based (Expert Only)
echo ""
echo "========================================"
echo "Experiment 4: Routing-Based (Expert Only)"
echo "========================================"
python train_alternating_optimization.py \
    --model_name "$MODEL_NAME" \
    --output_dir ./comparison_routing_expert_only \
    --num_rounds $NUM_ROUNDS \
    --epochs_per_round 1 \
    --batch_size 16 \
    --safe_ratio 0.5 \
    --expert_lr $EXPERT_LR \
    --skip_router_training

echo ""
echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  1. ./comparison_gradient_based/"
echo "  2. ./comparison_routing_based/"
echo "  3. ./comparison_gradient_expert_only/"
echo "  4. ./comparison_routing_expert_only/"
echo ""
echo "Next steps:"
echo "  1. Evaluate all 4 models on your test set"
echo "  2. Compare performance metrics"
echo "  3. Analyze identified unsafe experts"
echo ""
