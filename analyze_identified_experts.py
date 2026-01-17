#!/usr/bin/env python3
# Analysis script for identified unsafe experts
#
# This script helps visualize and compare unsafe experts identified by
# different methods (gradient-based vs routing-based)

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


def load_gradient_based_results(output_dir: str) -> Dict:
    """Load results from gradient-based method."""
    metadata_path = Path(output_dir) / "training_metadata.json"
    
    if not metadata_path.exists():
        print(f"Warning: {metadata_path} not found")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def load_routing_based_results(output_dir: str) -> Dict:
    """Load results from routing-based method."""
    metadata_path = Path(output_dir) / "training_metadata.json"
    
    if not metadata_path.exists():
        print(f"Warning: {metadata_path} not found")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def visualize_expert_heatmap(
    identified_experts: List[Tuple[int, int]],
    num_layers: int,
    num_experts: int,
    title: str = "Identified Unsafe Experts",
    save_path: str = None
):
    """
    Visualize identified unsafe experts as a heatmap.
    
    Args:
        identified_experts: List of (layer, expert) tuples
        num_layers: Total number of layers
        num_experts: Total number of experts per layer
        title: Plot title
        save_path: Path to save figure
    """
    # Create heatmap matrix
    heatmap = np.zeros((num_layers, num_experts))
    
    for layer, expert in identified_experts:
        heatmap[layer, expert] = 1
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap,
        cmap="YlOrRd",
        cbar_kws={'label': 'Identified as Unsafe'},
        xticklabels=range(num_experts),
        yticklabels=range(num_layers)
    )
    plt.xlabel("Expert Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_expert_identification(
    gradient_experts: List[Tuple[int, int]],
    routing_experts: List[Tuple[int, int]],
    num_layers: int,
    num_experts: int
) -> Dict:
    """
    Compare experts identified by gradient-based and routing-based methods.
    
    Returns:
        Dict with comparison statistics
    """
    gradient_set = set(gradient_experts)
    routing_set = set(routing_experts)
    
    # Calculate overlaps
    overlap = gradient_set & routing_set
    gradient_only = gradient_set - routing_set
    routing_only = routing_set - gradient_set
    
    # Calculate statistics
    overlap_ratio = len(overlap) / len(gradient_set) if len(gradient_set) > 0 else 0
    
    stats = {
        'gradient_count': len(gradient_set),
        'routing_count': len(routing_set),
        'overlap_count': len(overlap),
        'gradient_only_count': len(gradient_only),
        'routing_only_count': len(routing_only),
        'overlap_ratio': overlap_ratio,
        'overlap_experts': list(overlap),
        'gradient_only_experts': list(gradient_only),
        'routing_only_experts': list(routing_only),
    }
    
    return stats


def visualize_comparison(
    gradient_experts: List[Tuple[int, int]],
    routing_experts: List[Tuple[int, int]],
    num_layers: int,
    num_experts: int,
    save_dir: str = None
):
    """
    Visualize comparison between gradient-based and routing-based methods.
    """
    gradient_set = set(gradient_experts)
    routing_set = set(routing_experts)
    
    # Create comparison matrix
    # 0 = not identified, 1 = gradient only, 2 = routing only, 3 = both
    comparison_matrix = np.zeros((num_layers, num_experts))
    
    for layer, expert in gradient_experts:
        comparison_matrix[layer, expert] = 1
    
    for layer, expert in routing_experts:
        if comparison_matrix[layer, expert] == 1:
            comparison_matrix[layer, expert] = 3  # Both methods
        else:
            comparison_matrix[layer, expert] = 2  # Routing only
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    colors = ['white', 'lightblue', 'lightcoral', 'darkgreen']
    cmap = sns.color_palette(colors)
    
    sns.heatmap(
        comparison_matrix,
        cmap=cmap,
        vmin=0,
        vmax=3,
        cbar_kws={'label': 'Identification Method', 'ticks': [0, 1, 2, 3]},
        xticklabels=range(num_experts),
        yticklabels=range(num_layers)
    )
    
    # Customize colorbar
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticklabels(['Not Identified', 'Gradient Only', 'Routing Only', 'Both Methods'])
    
    plt.xlabel("Expert Index")
    plt.ylabel("Layer Index")
    plt.title("Comparison: Gradient-Based vs Routing-Based Expert Identification")
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / "comparison_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_statistics(stats: Dict):
    """Print comparison statistics."""
    print("\n" + "=" * 60)
    print("EXPERT IDENTIFICATION COMPARISON")
    print("=" * 60)
    print(f"Gradient-Based Method: {stats['gradient_count']} experts")
    print(f"Routing-Based Method:  {stats['routing_count']} experts")
    print(f"Overlap:               {stats['overlap_count']} experts ({stats['overlap_ratio']*100:.1f}%)")
    print(f"Gradient Only:         {stats['gradient_only_count']} experts")
    print(f"Routing Only:          {stats['routing_only_count']} experts")
    print("=" * 60)
    
    if stats['overlap_count'] > 0:
        print("\nOverlapping Experts (first 10):")
        for i, (layer, expert) in enumerate(stats['overlap_experts'][:10]):
            print(f"  {i+1}. Layer {layer}, Expert {expert}")
    
    if stats['gradient_only_count'] > 0:
        print("\nGradient-Only Experts (first 10):")
        for i, (layer, expert) in enumerate(stats['gradient_only_experts'][:10]):
            print(f"  {i+1}. Layer {layer}, Expert {expert}")
    
    if stats['routing_only_count'] > 0:
        print("\nRouting-Only Experts (first 10):")
        for i, (layer, expert) in enumerate(stats['routing_only_experts'][:10]):
            print(f"  {i+1}. Layer {layer}, Expert {expert}")


def analyze_gradient_magnitudes(gradient_metadata: Dict):
    """Analyze gradient magnitudes across rounds."""
    print("\n" + "=" * 60)
    print("GRADIENT MAGNITUDE ANALYSIS")
    print("=" * 60)
    
    identified_per_round = gradient_metadata.get('identified_experts_per_round', {})
    
    for round_num, data in identified_per_round.items():
        print(f"\nRound {round_num}:")
        print(f"  Identified experts: {len(data['experts'])}")
        
        if 'gradients' in data and data['gradients']:
            gradients_dict = data['gradients']
            
            # Sort by gradient value
            sorted_grads = sorted(gradients_dict.items(), key=lambda x: x[1], reverse=True)
            
            print(f"  Top 5 experts by gradient magnitude:")
            for i, (expert_key, grad_value) in enumerate(sorted_grads[:5]):
                print(f"    {i+1}. {expert_key}: {grad_value:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze identified unsafe experts")
    parser.add_argument(
        "--gradient_dir",
        type=str,
        default="./gradient_based_training",
        help="Directory with gradient-based results"
    )
    parser.add_argument(
        "--routing_dir",
        type=str,
        default=None,
        help="Directory with routing-based results (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Specific round to analyze (default: last round)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load gradient-based results
    print("Loading gradient-based results...")
    gradient_results = load_gradient_based_results(args.gradient_dir)
    
    if gradient_results is None:
        print("Error: Could not load gradient-based results")
        return
    
    # Analyze gradient magnitudes
    analyze_gradient_magnitudes(gradient_results)
    
    # Get identified experts for specified round
    identified_per_round = gradient_results.get('identified_experts_per_round', {})
    
    if not identified_per_round:
        print("Error: No identified experts found in metadata")
        return
    
    # Select round
    if args.round is None:
        round_key = str(max([int(k) for k in identified_per_round.keys()]))
    else:
        round_key = str(args.round)
    
    if round_key not in identified_per_round:
        print(f"Error: Round {round_key} not found in results")
        return
    
    gradient_experts = identified_per_round[round_key]['experts']
    print(f"\nAnalyzing Round {round_key}: {len(gradient_experts)} experts identified")
    
    # Estimate model dimensions (from first expert)
    if gradient_experts:
        max_layer = max([layer for layer, _ in gradient_experts])
        max_expert = max([expert for _, expert in gradient_experts])
        num_layers = max_layer + 1
        num_experts = max_expert + 1
    else:
        num_layers = 30
        num_experts = 64
    
    print(f"Model dimensions: {num_layers} layers, {num_experts} experts per layer")
    
    # Visualize gradient-based results
    print("\nGenerating gradient-based heatmap...")
    visualize_expert_heatmap(
        gradient_experts,
        num_layers,
        num_experts,
        title=f"Gradient-Based Method - Round {round_key}",
        save_path=Path(args.output_dir) / f"gradient_based_round{round_key}.png"
    )
    
    # If routing-based results provided, compare
    if args.routing_dir:
        print("\nLoading routing-based results...")
        routing_results = load_routing_based_results(args.routing_dir)
        
        if routing_results is not None:
            # For routing-based method, need to load the risk_diff data
            # This is a simplified version - you may need to adapt based on actual data format
            print("\nNote: Routing-based comparison requires additional implementation")
            print("Please ensure routing-based results include identified expert lists")
    
    print("\nAnalysis complete! Results saved in:", args.output_dir)


if __name__ == "__main__":
    main()




