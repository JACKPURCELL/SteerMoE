#!/usr/bin/env python3
"""
Example usage of OLMOE-FWO dataset evaluation.
"""

import os
import json
from evaluation_refactored import ModelEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def main():
    """Main function demonstrating OLMOE-FWO dataset evaluation."""
    
    # Configuration
    model_name = "./qwen1209/round_3"
    # model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    prefix = "qwenft"
    output_dir = "./data/results"
    
    # Path to OLMOE-FWO dataset
    olmoe_fwo_path = "qwen_fwo_test_data.json"
    # Use OLMOE-FWO dataset for jailbreak attack evaluation
    selected_datasets = ["olmoe_fwo","strongreject","xstest"]
    # ./qwen1209/round_3
    # selected_datasets = ["mmlu","gsm8k","truthfulqa","olmoe_fwo","strongreject","xstest"]
    
    print("=== Starting OLMOE-FWO Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset path: {olmoe_fwo_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator with OLMOE-FWO dataset path
    evaluator = ModelEvaluator(
        model_name=model_name,
        output_dir=output_dir,
        prefix=prefix,
        vllm=True,
        olmoe_fwo_path=olmoe_fwo_path
    )
    
    # Run evaluation
    scores = evaluator.run_evaluation(
        dataset_names=selected_datasets,
        num_samples_to_eval=100,  # Evaluate 50 samples as in original eval1124.py
        batch_size=8,
        max_workers=3  # Use 3 workers for concurrent judge evaluation
    )
    
    # Save aggregated scores
    scores_file = os.path.join(output_dir, f"full_scores_{prefix}.json")
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            existing_scores = json.load(f)
        existing_scores.update(scores)
        scores = existing_scores
    
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Full evaluation scores: {scores}")
    print(f"Results saved to: {scores_file}")


if __name__ == "__main__":
    main()

