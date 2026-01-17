#!/usr/bin/env python3
"""
Quick verification script to check ablation experiment setup.
This shows what will be trained with different parameter combinations.
"""

def verify_setup(finetune_experts=True, use_router_loss=False, router_weight=0.1):
    """Verify and display the training setup."""
    print("=" * 80)
    print("TRAINING SETUP VERIFICATION")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  --finetune_unsafe_experts: {finetune_experts}")
    print(f"  --use_router_consistency_loss: {use_router_loss}")
    print(f"  --router_consistency_weight: {router_weight}")
    
    print(f"\nTrainable Components:")
    
    if finetune_experts and use_router_loss:
        print("  ✅ Unsafe Expert MLPs (finetuned)")
        print("  ✅ Routers (via consistency loss)")
        print("\n📝 Experiment Type: COMBINED (Expert + Router)")
        print("💡 This is the recommended full approach")
        
    elif finetune_experts and not use_router_loss:
        print("  ✅ Unsafe Expert MLPs (finetuned)")
        print("  ❌ Routers (frozen)")
        print("\n📝 Experiment Type: EXPERT ONLY")
        print("💡 This is the original method (baseline)")
        
    elif not finetune_experts and use_router_loss:
        print("  ❌ Unsafe Expert MLPs (frozen)")
        print("  ✅ Routers (via consistency loss)")
        print("\n📝 Experiment Type: ROUTER ONLY")
        print("💡 This tests router consistency loss in isolation")
        
    else:
        print("  ❌ Unsafe Expert MLPs (frozen)")
        print("  ❌ Routers (frozen)")
        print("\n📝 Experiment Type: FROZEN BASELINE")
        print("⚠️  WARNING: No parameters will be trained!")
        print("   This is only useful as a sanity check")
    
    print("\nLosses:")
    if finetune_experts:
        print("  • Cross-Entropy Loss → updates expert parameters")
    if use_router_loss:
        print(f"  • Router Consistency Loss (weight={router_weight}) → updates router parameters")
    if not finetune_experts and not use_router_loss:
        print("  • None (frozen model)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT CONFIGURATIONS")
    print("=" * 80)
    
    configs = [
        {
            "name": "Expert Only (Original Method)",
            "finetune_experts": True,
            "use_router_loss": False,
            "router_weight": 0.1,
            "command": "python train_batch_unsafe_experts.py --mode selective"
        },
        {
            "name": "Router Only (Ablation)",
            "finetune_experts": False,
            "use_router_loss": True,
            "router_weight": 0.1,
            "command": "python train_batch_unsafe_experts.py --mode selective --no_finetune_unsafe_experts --use_router_consistency_loss"
        },
        {
            "name": "Combined (Recommended)",
            "finetune_experts": True,
            "use_router_loss": True,
            "router_weight": 0.1,
            "command": "python train_batch_unsafe_experts.py --mode selective --use_router_consistency_loss"
        },
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {i}: {config['name']}")
        print(f"{'='*80}")
        verify_setup(
            finetune_experts=config["finetune_experts"],
            use_router_loss=config["use_router_loss"],
            router_weight=config["router_weight"]
        )
        print(f"\nCommand to run:")
        print(f"  {config['command']}")
        print()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
推荐的实验顺序:
1. 先运行 "Expert Only" - 这是 baseline，看原方法效果
2. 再运行 "Router Only" - 看单独优化 router 的效果
3. 最后运行 "Combined" - 看两者结合的效果

对比三个实验的结果，可以回答:
- Router consistency loss 单独有多少贡献？
- Expert finetuning 单独有多少贡献？
- 两者结合是否有协同效应？
""")
    print("=" * 80)

