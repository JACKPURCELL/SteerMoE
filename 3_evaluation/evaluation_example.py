#!/usr/bin/env python3
"""
Example usage of the refactored evaluation system.
"""

import os
import json
from evaluation_refactored import ModelEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
def main():
    """Main function demonstrating model evaluation."""
    
    # Configuration
    model_name = "Qwen/Qwen3-30B-A3B"
    # model_name = "huihui-ai/Qwen3-8B-abliterated"
    
    prefix = "Qwen3-30B-A3B"
    output_dir = "./data/results"
    
    # Optional: Path to custom vulnerability report
    # vulnerability_report_path = "./data/raw/vulnerability_report.json"
    vulnerability_report_path = None
    
    # Optional: Path to OLMOE-FWO dataset
    olmoe_fwo_path = "qwen_fwo_test_data.json"
    
    # Available datasets:
    # - Safety: "strongreject", "harmbench", "pku_safe", "xstest", "custom", "olmoe_fwo"
    # - Capability: "mmlu", "gsm8k", "truthfulqa"
    # - Generation: "alpaca_eval"
    # selected_datasets = [ "strongreject","harmbench", "xstest", "truthfulqa","mmlu","gsm8k"]
    selected_datasets = [ "strongreject","harmbench"]
    
    print("=== Starting Model Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Datasets: {', '.join(selected_datasets)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_name=model_name,
        output_dir=output_dir,
        prefix=prefix,
        vllm=True,
        # vulnerability_report_path=vulnerability_report_path if os.path.exists(vulnerability_report_path) else None
        olmoe_fwo_path=olmoe_fwo_path
    )
    
    # Run evaluation
    scores = evaluator.run_evaluation(
        dataset_names=selected_datasets,
        num_samples_to_eval=100,
        batch_size=32,
        max_workers=4
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
