import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse

# Set style
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.3

def load_corrected_metrics():
    """Load the corrected metrics analysis data."""
    analysis_path = Path("/home/gaetano/utooth/outputs/analysis/corrected_metrics_analysis.json")
    folds_path = Path("/home/gaetano/utooth/outputs/analysis/corrected_metrics_analysis_folds.csv")
    
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    
    folds_df = pd.read_csv(folds_path)
    
    return analysis_data, folds_df

def load_run_data(run_name):
    """Load training metrics and corrected evaluation data for a given run."""
    base_path = Path(f"/home/gaetano/utooth/outputs/runs/{run_name}")
    
    # Load fold training metrics
    fold_metrics = {}
    n_folds = 10 if '10f' in run_name else 5
    
    for i in range(n_folds):
        try:
            df = pd.read_csv(base_path / f"metrics/fold_{i}/metrics.csv")
            # Clean up the data
            df = df.dropna(subset=['epoch'])
            df = df.groupby('epoch').last().reset_index()
            fold_metrics[i] = df
        except:
            continue  # Fold not found
    
    return fold_metrics

def create_visualizations(run_name, output_dir=None):
    """Create comprehensive visualizations for a training run with corrected metrics."""
    
    if output_dir is None:
        output_dir = Path(f"/home/gaetano/utooth/outputs/visualizations/{run_name}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    fold_metrics = load_run_data(run_name)
    analysis_data, folds_df = load_corrected_metrics()
    
    # Get corrected metrics for this run
    run_analysis = None
    for exp in analysis_data:
        if exp['experiment'] == run_name:
            run_analysis = exp
            break
    
    if run_analysis is None:
        print(f"No corrected metrics found for {run_name}")
        return None
    
    run_folds_df = folds_df[folds_df['experiment'] == run_name]
    
    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_metrics)))
    
    # Create figure with 8 subplots (2x4 layout)
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Validation Loss Across All Folds
    ax1 = plt.subplot(3, 3, 1)
    for idx, (fold_idx, df) in enumerate(fold_metrics.items()):
        if 'val_loss' in df.columns:
            ax1.plot(df['epoch'], df['val_loss'], label=f'Fold {fold_idx}', 
                    linewidth=2.5, color=colors[idx])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Training Validation Loss Progression', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Accuracy Across All Folds (Old Metric)
    ax2 = plt.subplot(3, 3, 2)
    for idx, (fold_idx, df) in enumerate(fold_metrics.items()):
        if 'val_accu' in df.columns:
            ax2.plot(df['epoch'], df['val_accu'], label=f'Fold {fold_idx}', 
                    linewidth=2.5, color=colors[idx])
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Old Validation Accuracy', fontsize=12)
    ax2.set_title('Training Old Accuracy Progression', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Corrected IoU Scores by Fold (Bar Chart)
    ax3 = plt.subplot(3, 3, 3)
    fold_indices = run_folds_df['fold_idx'].values
    corrected_ious = run_folds_df['corrected_iou'].values
    dice_scores = run_folds_df['dice'].values
    
    x = np.arange(len(fold_indices))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, corrected_ious, width, label='Corrected IoU', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x + width/2, dice_scores, width, label='Dice Score', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Fold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Final Corrected IoU & Dice Scores by Fold', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'F{i}' for i in fold_indices])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Comparison: Old vs Corrected Metrics
    ax4 = plt.subplot(3, 3, 4)
    old_accuracy = run_folds_df['old_accuracy'].values
    
    ax4.scatter(old_accuracy, corrected_ious, alpha=0.7, s=100, color='blue', label='IoU vs Old Acc')
    ax4.scatter(old_accuracy, dice_scores, alpha=0.7, s=100, color='red', label='Dice vs Old Acc')
    
    # Add diagonal line for reference
    min_val = min(old_accuracy.min(), corrected_ious.min())
    max_val = max(old_accuracy.max(), dice_scores.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax4.set_xlabel('Old Accuracy', fontsize=12)
    ax4.set_ylabel('Corrected Metrics', fontsize=12)
    ax4.set_title('Old vs Corrected Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution of Corrected Metrics
    ax5 = plt.subplot(3, 3, 5)
    
    metrics_data = [corrected_ious, dice_scores, run_folds_df['binary_iou'].values]
    labels = ['Corrected IoU', 'Dice Score', 'Binary IoU']
    colors_box = ['skyblue', 'lightcoral', 'lightgreen']
    
    bp = ax5.boxplot(metrics_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Distribution of Corrected Metrics', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Validation Loss vs Final Performance
    ax6 = plt.subplot(3, 3, 6)
    val_losses = run_folds_df['val_loss'].values
    
    ax6.scatter(val_losses, corrected_ious, alpha=0.7, s=100, color='blue', label='IoU')
    ax6.scatter(val_losses, dice_scores, alpha=0.7, s=100, color='red', label='Dice')
    
    ax6.set_xlabel('Final Validation Loss', fontsize=12)
    ax6.set_ylabel('Performance Score', fontsize=12)
    ax6.set_title('Validation Loss vs Final Performance', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Training Loss Curves (Best and Worst Folds)
    ax7 = plt.subplot(3, 3, 7)
    
    # Find best and worst folds by Dice score
    best_fold_idx = run_folds_df.loc[run_folds_df['dice'].idxmax(), 'fold_idx']
    worst_fold_idx = run_folds_df.loc[run_folds_df['dice'].idxmin(), 'fold_idx']
    
    for fold_idx, label_prefix, color in [(best_fold_idx, 'Best', 'green'), (worst_fold_idx, 'Worst', 'red')]:
        if fold_idx in fold_metrics:
            df = fold_metrics[fold_idx]
            if 'train_loss_epoch' in df.columns and 'val_loss' in df.columns:
                ax7.plot(df['epoch'], df['train_loss_epoch'], 
                        label=f'{label_prefix} Fold {fold_idx} Train', 
                        linewidth=2.5, color=color)
                ax7.plot(df['epoch'], df['val_loss'], 
                        label=f'{label_prefix} Fold {fold_idx} Val', 
                        linewidth=2.5, linestyle='--', color=color)
    
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Loss', fontsize=12)
    ax7.set_title(f'Training Curves: Best vs Worst Folds', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Ranking
    ax8 = plt.subplot(3, 3, 8)
    
    # Sort folds by Dice score
    sorted_folds = run_folds_df.sort_values('dice', ascending=True)
    y_pos = np.arange(len(sorted_folds))
    
    # Create horizontal bar chart
    bars = ax8.barh(y_pos, sorted_folds['dice'], alpha=0.8, color='lightcoral')
    ax8.barh(y_pos, sorted_folds['corrected_iou'], alpha=0.6, color='skyblue')
    
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([f'Fold {int(idx)}' for idx in sorted_folds['fold_idx']])
    ax8.set_xlabel('Score', fontsize=12)
    ax8.set_title('Fold Performance Ranking', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (dice, iou) in enumerate(zip(sorted_folds['dice'], sorted_folds['corrected_iou'])):
        ax8.text(dice + 0.01, i, f'{dice:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 9. Summary Statistics Text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary stats
    summary_text = f"""
    {run_name.upper()} SUMMARY
    
    Number of Folds: {len(run_folds_df)}
    
    CORRECTED IoU:
    Mean: {run_analysis['metrics']['corrected_iou']['mean']:.1%}
    Std:  {run_analysis['metrics']['corrected_iou']['std']:.1%}
    Best: {run_analysis['metrics']['corrected_iou']['max']:.1%}
    
    DICE SCORE:
    Mean: {run_analysis['metrics']['dice']['mean']:.1%}
    Std:  {run_analysis['metrics']['dice']['std']:.1%}
    Best: {run_analysis['metrics']['dice']['max']:.1%}
    
    BINARY IoU:
    Mean: {run_analysis['metrics']['binary_iou']['mean']:.1%}
    Std:  {run_analysis['metrics']['binary_iou']['std']:.1%}
    Best: {run_analysis['metrics']['binary_iou']['max']:.1%}
    
    OLD ACCURACY:
    Mean: {run_analysis['metrics']['old_accuracy']['mean']:.1%}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{run_name}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison table
    fig2, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in run_folds_df.iterrows():
        table_data.append([
            f"Fold {int(row['fold_idx'])}",
            f"{row['old_accuracy']:.3f}",
            f"{row['corrected_iou']:.3f}",
            f"{row['dice']:.3f}",
            f"{row['binary_iou']:.3f}",
            f"{row['val_loss']:.4f}"
        ])
    
    # Add summary row
    table_data.append([
        "MEAN",
        f"{run_analysis['metrics']['old_accuracy']['mean']:.3f}",
        f"{run_analysis['metrics']['corrected_iou']['mean']:.3f}",
        f"{run_analysis['metrics']['dice']['mean']:.3f}",
        f"{run_analysis['metrics']['binary_iou']['mean']:.3f}",
        f"{run_folds_df['val_loss'].mean():.4f}"
    ])
    
    headers = ['Fold', 'Old Acc', 'Corrected IoU', 'Dice Score', 'Binary IoU', 'Val Loss']
    
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.15, 0.18, 0.18, 0.17, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the summary row
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#FFC107')
        table[(len(table_data), i)].set_text_props(weight='bold')
    
    # Add alternating row colors
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(f'{run_name} - Detailed Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / f'{run_name}_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}:")
    print(f"- {run_name}_analysis.png")
    print(f"- {run_name}_summary_table.png")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Visualize training run metrics with corrected analysis')
    parser.add_argument('run_name', help='Name of the run to visualize')
    parser.add_argument('--output-dir', help='Output directory for visualizations', default=None)
    
    args = parser.parse_args()
    
    try:
        create_visualizations(args.run_name, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # Default behavior - show available runs
        print("Available runs:")
        runs_path = Path("/home/gaetano/utooth/outputs/runs")
        for run_dir in sorted(runs_path.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith('utooth_'):
                print(f"  {run_dir.name}")
        print("\nUsage: python visualize_run.py <run_name>")
    else:
        sys.exit(main())