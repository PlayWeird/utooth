#!/usr/bin/env python3
"""
Analysis and visualization script for seed search results
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def load_seed_results(output_dir: Path) -> Dict:
    """Load all seed search results"""
    final_results_file = output_dir / "reports" / "final_results.json"
    
    if final_results_file.exists():
        with open(final_results_file, 'r') as f:
            return json.load(f)
    
    # Otherwise, collect results from individual seed directories
    results = {'results': []}
    seeds_dir = output_dir / "seeds"
    
    if seeds_dir.exists():
        for seed_dir in seeds_dir.glob("seed_*"):
            results_file = seed_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['results'].append(json.load(f))
    
    return results


def analyze_results(results: Dict, output_dir: Path):
    """Perform comprehensive analysis of seed search results"""
    
    # Filter valid results
    valid_results = [r for r in results.get('results', []) 
                    if 'mean_dice' in r and r['mean_dice'] > 0]
    
    if not valid_results:
        print("No valid results to analyze!")
        return
    
    print("="*80)
    print("SEED SEARCH ANALYSIS REPORT")
    print("="*80)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(valid_results)
    
    # Basic statistics
    print("\n1. OVERALL STATISTICS")
    print("-"*40)
    print(f"Total seeds evaluated: {len(results.get('results', []))}")
    print(f"Successful seeds: {len(valid_results)}")
    print(f"Failed seeds: {len(results.get('results', [])) - len(valid_results)}")
    
    print(f"\nDice Score Statistics:")
    print(f"  Best:  {df['mean_dice'].max():.4f} (Seed {df.loc[df['mean_dice'].idxmax(), 'seed']})")
    print(f"  Mean:  {df['mean_dice'].mean():.4f} ± {df['mean_dice'].std():.4f}")
    print(f"  Worst: {df['mean_dice'].min():.4f} (Seed {df.loc[df['mean_dice'].idxmin(), 'seed']})")
    
    # Top 10 seeds
    print("\n2. TOP 10 SEEDS")
    print("-"*40)
    top_10 = df.nlargest(10, 'mean_dice')[['seed', 'mean_dice', 'std_dice']]
    for idx, row in top_10.iterrows():
        print(f"  Seed {row['seed']:3d}: {row['mean_dice']:.4f} ± {row['std_dice']:.4f}")
    
    # Stability analysis
    print("\n3. STABILITY ANALYSIS")
    print("-"*40)
    print("Seeds with lowest variance (most stable):")
    stable_seeds = df.nsmallest(5, 'std_dice')[['seed', 'mean_dice', 'std_dice']]
    for idx, row in stable_seeds.iterrows():
        print(f"  Seed {row['seed']:3d}: {row['mean_dice']:.4f} ± {row['std_dice']:.4f}")
    
    # Performance consistency
    print("\n4. PERFORMANCE CONSISTENCY")
    print("-"*40)
    if 'min_dice' in df.columns and 'max_dice' in df.columns:
        df['dice_range'] = df['max_dice'] - df['min_dice']
        consistent_seeds = df.nsmallest(5, 'dice_range')[['seed', 'mean_dice', 'min_dice', 'max_dice']]
        print("Seeds with most consistent performance across folds:")
        for idx, row in consistent_seeds.iterrows():
            print(f"  Seed {row['seed']:3d}: [{row['min_dice']:.4f}, {row['max_dice']:.4f}] (mean: {row['mean_dice']:.4f})")
    
    # Best balanced seeds (high performance + low variance)
    print("\n5. BEST BALANCED SEEDS")
    print("-"*40)
    # Calculate a balance score (high dice, low std)
    df['balance_score'] = df['mean_dice'] - 0.5 * df['std_dice']
    balanced_seeds = df.nlargest(5, 'balance_score')[['seed', 'mean_dice', 'std_dice', 'balance_score']]
    print("Seeds with best balance of performance and stability:")
    for idx, row in balanced_seeds.iterrows():
        print(f"  Seed {row['seed']:3d}: Dice={row['mean_dice']:.4f} ± {row['std_dice']:.4f} (score: {row['balance_score']:.4f})")
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Save detailed analysis
    save_detailed_analysis(df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete! Check the reports directory for detailed results.")


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots"""
    
    plots_dir = output_dir / "reports" / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Distribution of Dice scores
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of mean dice scores
    axes[0, 0].hist(df['mean_dice'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['mean_dice'].mean(), color='red', linestyle='--', label=f'Mean: {df["mean_dice"].mean():.4f}')
    axes[0, 0].set_xlabel('Mean Dice Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Mean Dice Scores')
    axes[0, 0].legend()
    
    # Scatter plot: mean vs std
    axes[0, 1].scatter(df['mean_dice'], df['std_dice'], alpha=0.6)
    axes[0, 1].set_xlabel('Mean Dice Score')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Performance vs Stability')
    
    # Annotate top 3 seeds
    top_3 = df.nlargest(3, 'mean_dice')
    for idx, row in top_3.iterrows():
        axes[0, 1].annotate(f"Seed {row['seed']}", 
                           (row['mean_dice'], row['std_dice']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    # Box plot of dice scores if fold results available
    if 'fold_results' in df.columns and df['fold_results'].iloc[0]:
        all_fold_scores = []
        seed_labels = []
        for _, row in df.nlargest(10, 'mean_dice').iterrows():
            if row['fold_results']:
                fold_scores = [f['val_dice'] for f in row['fold_results'] if 'val_dice' in f]
                all_fold_scores.extend(fold_scores)
                seed_labels.extend([f"S{row['seed']}"] * len(fold_scores))
        
        if all_fold_scores:
            fold_df = pd.DataFrame({'Seed': seed_labels, 'Dice': all_fold_scores})
            sns.boxplot(data=fold_df, x='Seed', y='Dice', ax=axes[1, 0])
            axes[1, 0].set_title('Fold-wise Performance (Top 10 Seeds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Ranking plot
    df_sorted = df.sort_values('mean_dice', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    axes[1, 1].plot(df_sorted['rank'], df_sorted['mean_dice'], marker='o', markersize=4)
    axes[1, 1].set_xlabel('Seed Rank')
    axes[1, 1].set_ylabel('Mean Dice Score')
    axes[1, 1].set_title('Seed Performance Ranking')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'seed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create a heatmap if we have enough seeds
    if len(df) >= 20 and 'min_dice' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for heatmap
        heatmap_data = df.nlargest(20, 'mean_dice')[['seed', 'mean_dice', 'std_dice', 'min_dice', 'max_dice']]
        heatmap_data = heatmap_data.set_index('seed')
        
        # Normalize for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
        
        sns.heatmap(heatmap_normalized.T, annot=heatmap_data.T, fmt='.3f', 
                   cmap='YlOrRd', cbar_kws={'label': 'Normalized Score'})
        plt.title('Top 20 Seeds Performance Metrics')
        plt.xlabel('Seed Number')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(plots_dir / 'seed_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualization plots saved to: {plots_dir}")


def save_detailed_analysis(df: pd.DataFrame, output_dir: Path):
    """Save detailed analysis to files"""
    
    reports_dir = output_dir / "reports"
    
    # Save full rankings
    df_sorted = df.sort_values('mean_dice', ascending=False)
    df_sorted.to_csv(reports_dir / 'full_seed_rankings.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_seeds': len(df),
        'dice_statistics': {
            'mean': float(df['mean_dice'].mean()),
            'std': float(df['mean_dice'].std()),
            'min': float(df['mean_dice'].min()),
            'max': float(df['mean_dice'].max()),
            'median': float(df['mean_dice'].median())
        },
        'stability_statistics': {
            'mean_std': float(df['std_dice'].mean()),
            'min_std': float(df['std_dice'].min()),
            'max_std': float(df['std_dice'].max())
        },
        'best_seed': {
            'seed': int(df.loc[df['mean_dice'].idxmax(), 'seed']),
            'mean_dice': float(df['mean_dice'].max()),
            'std_dice': float(df.loc[df['mean_dice'].idxmax(), 'std_dice'])
        },
        'most_stable_seed': {
            'seed': int(df.loc[df['std_dice'].idxmin(), 'seed']),
            'mean_dice': float(df.loc[df['std_dice'].idxmin(), 'mean_dice']),
            'std_dice': float(df['std_dice'].min())
        }
    }
    
    # Add balanced score analysis
    df['balance_score'] = df['mean_dice'] - 0.5 * df['std_dice']
    best_balanced = df.loc[df['balance_score'].idxmax()]
    summary_stats['best_balanced_seed'] = {
        'seed': int(best_balanced['seed']),
        'mean_dice': float(best_balanced['mean_dice']),
        'std_dice': float(best_balanced['std_dice']),
        'balance_score': float(best_balanced['balance_score'])
    }
    
    with open(reports_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Detailed analysis saved to: {reports_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze seed search results")
    parser.add_argument('--dir', type=str, required=True,
                       help='Path to seed search output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.dir)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} not found!")
        return
    
    results = load_seed_results(output_dir)
    analyze_results(results, output_dir)


if __name__ == "__main__":
    main()