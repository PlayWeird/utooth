#!/usr/bin/env python3
"""
Create Loss and Dice Graphs for Original Seed 2113
===================================================

Generates publication-quality figures showing training curves for the original
seed 2113 results with consistent colors across all plots.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

def get_consistent_colors(n_folds=10):
    """Get consistent, distinct colors for each fold"""
    # Use a colormap that provides distinct colors
    cmap = plt.cm.tab10  # 10 distinct colors
    colors = [cmap(i) for i in range(n_folds)]
    return colors

def load_fold_metrics(seed_dir, fold_idx):
    """Load metrics for a specific fold"""
    fold_dir = Path(seed_dir) / f"fold_{fold_idx}"
    metrics_file = fold_dir / "metrics.csv"
    
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        # Get only validation rows (where val_dice is not null)
        val_df = df[df['val_dice'].notna()].copy()
        
        if len(val_df) == 0:
            return None
            
        return val_df
    except Exception as e:
        print(f"Error loading fold {fold_idx}: {e}")
        return None

def create_individual_fold_plots(seed_dir, output_dir, colors):
    """Create individual plots for each fold"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating individual fold plots...")
    
    for fold_idx in range(10):
        metrics_df = load_fold_metrics(seed_dir, fold_idx)
        if metrics_df is None:
            print(f"Skipping fold {fold_idx} - no data")
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = metrics_df['epoch'].values
        val_loss = metrics_df['val_loss'].values
        val_dice = metrics_df['val_dice'].values
        
        # Plot validation loss
        ax1.plot(epochs, val_loss, color=colors[fold_idx], linewidth=2, label=f'Fold {fold_idx}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title(f'Validation Loss - Fold {fold_idx}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add best point
        best_idx = np.argmax(val_dice)
        best_epoch = epochs[best_idx]
        best_loss = val_loss[best_idx]
        ax1.scatter(best_epoch, best_loss, color=colors[fold_idx], s=100, 
                   marker='*', edgecolor='black', linewidth=1, zorder=5)
        ax1.annotate(f'Best: {best_loss:.4f}\n@epoch {best_epoch}', 
                    xy=(best_epoch, best_loss), xytext=(10, 10), 
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot validation dice
        ax2.plot(epochs, val_dice, color=colors[fold_idx], linewidth=2, label=f'Fold {fold_idx}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Dice Score')
        ax2.set_title(f'Validation Dice Score - Fold {fold_idx}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add best point
        best_dice = val_dice[best_idx]
        ax2.scatter(best_epoch, best_dice, color=colors[fold_idx], s=100, 
                   marker='*', edgecolor='black', linewidth=1, zorder=5)
        ax2.annotate(f'Best: {best_dice:.4f}\n@epoch {best_epoch}', 
                    xy=(best_epoch, best_dice), xytext=(10, 10), 
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fold_{fold_idx}_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'fold_{fold_idx}_curves.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Saved plots for fold {fold_idx}")

def create_combined_plots(seed_dir, output_dir, colors):
    """Create combined plots showing all folds together"""
    output_dir = Path(output_dir)
    
    print("Creating combined plots...")
    
    # Create combined loss plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    all_best_points = []
    
    for fold_idx in range(10):
        metrics_df = load_fold_metrics(seed_dir, fold_idx)
        if metrics_df is None:
            continue
        
        epochs = metrics_df['epoch'].values
        val_loss = metrics_df['val_loss'].values
        val_dice = metrics_df['val_dice'].values
        
        # Plot validation loss
        ax.plot(epochs, val_loss, color=colors[fold_idx], linewidth=1.5, 
               alpha=0.8, label=f'Fold {fold_idx}')
        
        # Mark best point
        best_idx = np.argmax(val_dice)
        best_epoch = epochs[best_idx]
        best_loss = val_loss[best_idx]
        best_dice = val_dice[best_idx]
        
        ax.scatter(best_epoch, best_loss, color=colors[fold_idx], s=80, 
                  marker='*', edgecolor='black', linewidth=1, zorder=5)
        
        all_best_points.append({
            'fold': fold_idx,
            'epoch': best_epoch,
            'loss': best_loss,
            'dice': best_dice
        })
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss - All Folds (Seed 2113)\n* markers show best Dice epoch for each fold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_loss_curves.pdf', bbox_inches='tight')
    plt.close()
    
    # Create combined dice plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for fold_idx in range(10):
        metrics_df = load_fold_metrics(seed_dir, fold_idx)
        if metrics_df is None:
            continue
        
        epochs = metrics_df['epoch'].values
        val_dice = metrics_df['val_dice'].values
        
        # Plot validation dice
        ax.plot(epochs, val_dice, color=colors[fold_idx], linewidth=1.5, 
               alpha=0.8, label=f'Fold {fold_idx}')
        
        # Mark best point
        best_idx = np.argmax(val_dice)
        best_epoch = epochs[best_idx]
        best_dice = val_dice[best_idx]
        
        ax.scatter(best_epoch, best_dice, color=colors[fold_idx], s=80, 
                  marker='*', edgecolor='black', linewidth=1, zorder=5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Dice Score')
    ax.set_title('Validation Dice Score - All Folds (Seed 2113)\n* markers show best Dice score for each fold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_dice_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_dice_curves.pdf', bbox_inches='tight')
    plt.close()
    
    return all_best_points

def create_summary_plots(all_best_points, output_dir, colors):
    """Create summary plots showing best performance analysis"""
    output_dir = Path(output_dir)
    
    print("Creating summary plots...")
    
    # Extract data
    folds = [p['fold'] for p in all_best_points]
    epochs = [p['epoch'] for p in all_best_points]
    losses = [p['loss'] for p in all_best_points]
    dices = [p['dice'] for p in all_best_points]
    
    # Create summary figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Best Dice scores by fold
    bars1 = ax1.bar(folds, dices, color=[colors[f] for f in folds], alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Best Validation Dice Score')
    ax1.set_title('Best Dice Score by Fold')
    ax1.set_xticks(folds)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, dice in zip(bars1, dices):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{dice:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Best epochs by fold
    bars2 = ax2.bar(folds, epochs, color=[colors[f] for f in folds], alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Best Epoch')
    ax2.set_title('Epoch of Best Performance by Fold')
    ax2.set_xticks(folds)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, epoch in zip(bars2, epochs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{epoch}', ha='center', va='bottom', fontsize=9)
    
    # Dice vs Epoch scatter
    ax3.scatter(epochs, dices, c=[colors[f] for f in folds], s=100, alpha=0.8, edgecolor='black')
    for i, fold in enumerate(folds):
        ax3.annotate(f'F{fold}', (epochs[i], dices[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Best Epoch')
    ax3.set_ylabel('Best Dice Score')
    ax3.set_title('Best Dice vs Best Epoch')
    ax3.grid(True, alpha=0.3)
    
    # Distribution plots
    ax4.hist(dices, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(dices), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dices):.4f}')
    ax4.axvline(np.median(dices), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(dices):.4f}')
    ax4.set_xlabel('Best Dice Score')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Best Dice Scores')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Create statistics summary
    stats = {
        'mean_dice': np.mean(dices),
        'std_dice': np.std(dices),
        'min_dice': np.min(dices),
        'max_dice': np.max(dices),
        'mean_epoch': np.mean(epochs),
        'std_epoch': np.std(epochs),
        'min_epoch': np.min(epochs),
        'max_epoch': np.max(epochs)
    }
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Mean Dice: {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}")
    print(f"Range: {stats['min_dice']:.4f} - {stats['max_dice']:.4f}")
    print(f"Mean Best Epoch: {stats['mean_epoch']:.1f} ± {stats['std_epoch']:.1f}")
    print(f"Epoch Range: {stats['min_epoch']} - {stats['max_epoch']}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Create graphs for original seed 2113')
    parser.add_argument('--seed_dir', type=str, 
                        default='/home/user/utooth/outputs/seed_search/seed_search_20250815_104509/seeds/seed_2113',
                        help='Path to seed 2113 directory')
    parser.add_argument('--output_dir', type=str, default='outputs/seed_2113_graphs',
                        help='Output directory for graphs')
    parser.add_argument('--individual', action='store_true', 
                        help='Create individual fold plots (can be many files)')
    
    args = parser.parse_args()
    
    seed_dir = Path(args.seed_dir)
    output_dir = Path(args.output_dir)
    
    if not seed_dir.exists():
        print(f"Error: Seed directory {seed_dir} does not exist!")
        return
    
    print(f"Creating graphs for seed 2113")
    print(f"Input directory: {seed_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get consistent colors
    colors = get_consistent_colors(10)
    
    # Create combined plots (always)
    all_best_points = create_combined_plots(seed_dir, output_dir, colors)
    
    # Create summary analysis
    stats = create_summary_plots(all_best_points, output_dir, colors)
    
    # Create individual plots (optional)
    if args.individual:
        individual_dir = output_dir / "individual_folds"
        create_individual_fold_plots(seed_dir, individual_dir, colors)
    
    print(f"\nGraphs saved to: {output_dir}")
    print("Files created:")
    print("  - combined_loss_curves.png/pdf")
    print("  - combined_dice_curves.png/pdf") 
    print("  - summary_analysis.png/pdf")
    if args.individual:
        print("  - individual_folds/fold_X_curves.png/pdf (for each fold)")

if __name__ == "__main__":
    main()