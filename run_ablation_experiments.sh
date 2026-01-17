#!/bin/bash
# 完整的消融实验脚本
# 按照推荐顺序运行三个实验

MODEL_NAME="allenai/OLMoE-1B-7B-0125-Instruct"
NUM_EPOCHS=3
BATCH_SIZE=16

echo "========================================================================"
echo "Starting Ablation Experiments"
echo "========================================================================"
echo "Model: $MODEL_NAME"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo ""
echo "Experiment Order:"
echo "  1. Expert Only (Baseline)"
echo "  2. Router Only (New Method - Isolated)"
echo "  3. Combined (Expert + Router)"
echo "========================================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# ============================================================================
# 实验 1: Expert Only (Baseline)
# ============================================================================
echo ""
echo "========================================================================"
echo "EXPERIMENT 1/3: Expert Only (Baseline)"
echo "========================================================================"
echo "Training only unsafe expert MLPs (original method)"
echo "Output: ./exp_1_expert_only"
echo "Start time: $(date)"
echo "========================================================================"

python train_batch_unsafe_experts.py \
    --model_name "$MODEL_NAME" \
    --mode selective \
    --output_dir ./exp_1_expert_only \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --logging_steps 10 \
    --save_steps 200

EXP1_EXIT_CODE=$?
if [ $EXP1_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Experiment 1 failed with exit code $EXP1_EXIT_CODE"
    exit $EXP1_EXIT_CODE
fi

echo ""
echo "✅ Experiment 1 completed successfully!"
echo "End time: $(date)"
echo ""

# ============================================================================
# 实验 2: Router Only (New Method)
# ============================================================================
echo ""
echo "========================================================================"
echo "EXPERIMENT 2/3: Router Only (New Method - Isolated)"
echo "========================================================================"
echo "Training only routers with consistency loss"
echo "Output: ./exp_2_router_only"
echo "Start time: $(date)"
echo "========================================================================"

python train_batch_unsafe_experts.py \
    --model_name "$MODEL_NAME" \
    --mode selective \
    --no_finetune_unsafe_experts \
    --use_router_consistency_loss \
    --router_consistency_weight 0.5 \
    --router_consistency_type symmetric \
    --output_dir ./exp_2_router_only \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --logging_steps 10 \
    --save_steps 200

EXP2_EXIT_CODE=$?
if [ $EXP2_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Experiment 2 failed with exit code $EXP2_EXIT_CODE"
    exit $EXP2_EXIT_CODE
fi

echo ""
echo "✅ Experiment 2 completed successfully!"
echo "End time: $(date)"
echo ""

# ============================================================================
# 实验 3: Combined (Expert + Router)
# ============================================================================
echo ""
echo "========================================================================"
echo "EXPERIMENT 3/3: Combined (Expert + Router)"
echo "========================================================================"
echo "Training both unsafe experts and routers"
echo "Output: ./exp_3_combined"
echo "Start time: $(date)"
echo "========================================================================"

python train_batch_unsafe_experts.py \
    --model_name "$MODEL_NAME" \
    --mode selective \
    --use_router_consistency_loss \
    --router_consistency_weight 0.1 \
    --router_consistency_type symmetric \
    --output_dir ./exp_3_combined \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --logging_steps 10 \
    --save_steps 200

EXP3_EXIT_CODE=$?
if [ $EXP3_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Experiment 3 failed with exit code $EXP3_EXIT_CODE"
    exit $EXP3_EXIT_CODE
fi

echo ""
echo "✅ Experiment 3 completed successfully!"
echo "End time: $(date)"
echo ""

# ============================================================================
# 总结
# ============================================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✅ Experiment 1: Expert Only - DONE"
echo "  ✅ Experiment 2: Router Only - DONE"
echo "  ✅ Experiment 3: Combined - DONE"
echo ""
echo "Output directories:"
echo "  1. ./exp_1_expert_only/"
echo "  2. ./exp_2_router_only/"
echo "  3. ./exp_3_combined/"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Next steps:"
echo "  1. Evaluate all three models on your test set"
echo "  2. Compare metrics (accuracy, safety, etc.)"
echo "  3. Analyze router distributions"
echo ""
echo "Run evaluation:"
echo "  python eval_model.py --model_path ./exp_1_expert_only/final_model"
echo "  python eval_model.py --model_path ./exp_2_router_only/final_model"
echo "  python eval_model.py --model_path ./exp_3_combined/final_model"
echo "========================================================================"

