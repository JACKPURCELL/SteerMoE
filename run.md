CUDA_VISIBLE_DEVICES=0,1,4,5 python train_alternating_optimization.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./olmoe1222-deep-4 \
    --num_rounds 3 \
    --epochs_per_round 2 \
    --expert_lr 5e-5 \
    --router_lr 1e-6 \
    --kl_type forward

    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_alternating_optimization.py \
    --model_name allenai/OLMoE-1B-7B-0125-Instruct \
    --output_dir ./olmoe1124-expert_only_results \
    --num_rounds 3 \
    --epochs_per_round 1 \
    --expert_lr 5e-5 \
    --skip_router_training  # 添加这个参数！


    # 步骤1: 识别unsafe experts并准备数据
CUDA_VISIBLE_DEVICES=0,1,4,5 python demo_refactored.py

# 步骤2: 交替优化训练
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_alternating_optimization.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --output_dir ./qwen1209 \
    --num_rounds 3 \
    --epochs_per_round 1 \
    --expert_lr 5e-5 \
    --router_lr 1e-6 \
    --kl_type forward