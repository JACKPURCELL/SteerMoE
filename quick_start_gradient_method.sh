#!/bin/bash
# Quick start script for gradient-based method
#
# This script demonstrates how to quickly train using the new gradient-based method

set -e  # Exit on error

echo "========================================"
echo "Quick Start: Gradient-Based Method"
echo "========================================"
echo ""

# Step 1: Check if required files exist
echo "Step 1: Checking dataset..."
DATASET_PATH=/home/stufs1/jiachliang/FlipAttack/reproduce_result/processed_dataset/OLMOE-merged-train.json

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please update DATASET_PATH in this script"
    exit 1
fi
echo "✓ Dataset found"

# Step 2: Train with gradient-based method
echo ""
echo "Step 2: Training with gradient-based method..."
echo "This will:"
echo "  - Identify unsafe experts using gradient magnitude"
echo "  - Update only the top-100 unsafe experts"
echo "  - Run for 3 rounds"
echo "  - Save results to ./quick_start_results/"
echo ""

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_gradient_based_expert_identification.py \
    --model_name "allenai/OLMoE-1B-7B-0125-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir ./olmoe_merge_gradient_based_training-5e4100-noaccu \
    --num_rounds 3 \
    --top_k_experts 100 \
    --batch_size 8 \
    --expert_lr 5e-4 \
    --router_lr 1e-6 \
    --max_samples 1000

python /home/stufs1/jiachliang/SteerMoE/3_evaluation/evaluation_olmoe_fwo_example.py
# Step 3: Analyze results
echo ""
echo "Step 3: Analyzing identified experts..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python analyze_identified_experts.py \
    --gradient_dir ./quick_start_results \
    --output_dir ./quick_start_analysis

# Step 4: Summary
echo ""
echo "========================================"
echo "Quick Start Complete!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  - Training checkpoints: ./quick_start_results/"
echo "  - Analysis: ./quick_start_analysis/"
echo ""
echo "Files generated:"
echo "  1. ./quick_start_results/round_1/ - Round 1 checkpoint"
echo "  2. ./quick_start_results/round_2/ - Round 2 checkpoint"
echo "  3. ./quick_start_results/round_3/ - Round 3 checkpoint"
echo "  4. ./quick_start_results/final_model/ - Final trained model"
echo "  5. ./quick_start_results/training_metadata.json - Training logs and identified experts"
echo "  6. ./quick_start_results/expert_gradients.pkl - Detailed gradient data"
echo "  7. ./quick_start_analysis/gradient_based_round3.png - Visualization"
echo ""
echo "Next steps:"
echo "  1. Evaluate the final model on your test set"
echo "  2. Compare with the original routing-based method"
echo "  3. Analyze the identified unsafe experts"
echo ""
echo "To evaluate the model, use:"
echo "  cd 3_evaluation"
echo "  python evaluation_example.py --model_path ../quick_start_results/final_model"
echo ""
