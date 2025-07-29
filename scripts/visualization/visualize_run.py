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

def load_run_data(run_name):
    """Load all metrics and summary data for a given run."""
    base_path = Path(f"/home/gaetano/utooth/outputs/runs/{run_name}")
    
    # Load fold metrics
    fold_metrics = {}
    for i in range(10):  # Try up to 10 folds
        try:
            df = pd.read_csv(base_path / f"metrics/fold_{i}/metrics.csv")
            # Clean up the data
            df = df.dropna(subset=['epoch'])
            df = df.groupby('epoch').last().reset_index()
            fold_metrics[i] = df
        except:
            continue  # No more folds
    
    # Load summary data
    with open(base_path / "results_summary.json", 'r') as f:
        summary_data = json.load(f)
    
    return fold_metrics, summary_data

def create_visualizations(run_name, output_dir=None):
    """Create comprehensive visualizations for a training run."""
    
    if output_dir is None:
        output_dir = Path(f"/home/gaetano/utooth/outputs/visualizations/{run_name}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    fold_metrics, summary_data = load_run_data(run_name)
    
    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_metrics)))
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Validation Loss Across All Folds
    ax1 = plt.subplot(3, 2, 1)
    for idx, (fold_idx, df) in enumerate(fold_metrics.items()):
        if 'val_loss' in df.columns:
            ax1.plot(df['epoch'], df['val_loss'], label=f'Fold {fold_idx}', 
                    linewidth=2.5, color=colors[idx])
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss Progression Across All Folds', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Accuracy Across All Folds
    ax2 = plt.subplot(3, 2, 2)
    for idx, (fold_idx, df) in enumerate(fold_metrics.items()):
        if 'val_accu' in df.columns:
            ax2.plot(df['epoch'], df['val_accu'], label=f'Fold {fold_idx}', 
                    linewidth=2.5, color=colors[idx])
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Progression Across All Folds', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training vs Validation Loss Comparison (Best and Worst Folds)
    ax3 = plt.subplot(3, 2, 3)
    fold_results = summary_data['fold_results']
    best_fold = min(fold_results, key=lambda x: x['best_val_loss'])['fold_idx']
    worst_fold = max(fold_results, key=lambda x: x['best_val_loss'])['fold_idx']
    
    for fold_idx, label_prefix, color in [(best_fold, 'Best', 'green'), (worst_fold, 'Worst', 'red')]:
        df = fold_metrics[fold_idx]
        if 'train_loss_epoch' in df.columns and 'val_loss' in df.columns:
            ax3.plot(df['epoch'], df['train_loss_epoch'], 
                    label=f'{label_prefix} Fold {fold_idx} Train', 
                    linewidth=2.5, color=color)
            ax3.plot(df['epoch'], df['val_loss'], 
                    label=f'{label_prefix} Fold {fold_idx} Val', 
                    linewidth=2.5, linestyle='--', color=color)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title(f'Training vs Validation Loss: Best and Worst Performing Folds', 
                  fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box Plot of Final Metrics Across Folds
    ax4 = plt.subplot(3, 2, 4)
    final_val_losses = [fold['best_val_loss'] for fold in fold_results]
    
    # Create box plot
    bp = ax4.boxplot([final_val_losses], labels=['Best Val Loss'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Add individual points
    x = np.random.normal(1, 0.04, size=len(final_val_losses))
    ax4.scatter(x, final_val_losses, alpha=0.7, s=50, color='darkblue')
    
    # Add mean line
    ax4.axhline(y=np.mean(final_val_losses), color='red', linestyle='--', 
                label=f'Mean: {np.mean(final_val_losses):.4f}')
    
    ax4.set_ylabel('Validation Loss', fontsize=12)
    ax4.set_title('Distribution of Best Validation Loss Across Folds', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Time and Early Stopping Analysis
    ax5 = plt.subplot(3, 2, 5)
    fold_indices = [f['fold_idx'] for f in fold_results]
    training_times = [f['training_time_seconds']/60 for f in fold_results]  # Convert to minutes
    final_epochs = [f['final_epoch'] for f in fold_results]
    early_stopped = [f['early_stopped'] for f in fold_results]
    
    x = np.arange(len(fold_indices))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, training_times, width, label='Training Time (min)', alpha=0.8)
    bars2 = ax5.bar(x + width/2, final_epochs, width, label='Final Epoch', alpha=0.8)
    
    # Color early stopped folds differently
    for i, (bar1, bar2, stopped) in enumerate(zip(bars1, bars2, early_stopped)):
        if stopped:
            bar1.set_edgecolor('red')
            bar1.set_linewidth(3)
            bar2.set_edgecolor('red')
            bar2.set_linewidth(3)
    
    ax5.set_xlabel('Fold', fontsize=12)
    ax5.set_ylabel('Value', fontsize=12)
    ax5.set_title('Training Time and Epochs per Fold (Red Border = Early Stopped)', 
                  fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'Fold {i}' for i in fold_indices])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Learning Curve Convergence Rate
    ax6 = plt.subplot(3, 2, 6)
    for idx, (fold_idx, df) in enumerate(fold_metrics.items()):
        if 'train_loss_epoch' in df.columns and len(df) > 10:
            # Calculate moving average to smooth the curve
            window = 5
            train_loss_smooth = df['train_loss_epoch'].rolling(window=window, min_periods=1).mean()
            
            # Calculate convergence rate (derivative)
            convergence_rate = -np.gradient(train_loss_smooth)
            ax6.plot(df['epoch'][1:], convergence_rate[1:], 
                    label=f'Fold {fold_idx}', linewidth=2.5, color=colors[idx])
    
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Convergence Rate (Loss Decrease per Epoch)', fontsize=12)
    ax6.set_title('Training Convergence Rate Across Folds', fontsize=14, fontweight='bold')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(bottom=-0.01)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{run_name}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    summary_stats = {
        'Metric': ['Mean Val Loss', 'Std Val Loss', 'Min Val Loss', 'Max Val Loss', 
                   'Total Training Hours', 'Folds w/ Early Stop', 'Total Folds'],
        'Value': [
            f"{summary_data['summary']['mean_val_loss']:.4f}",
            f"{summary_data['summary']['std_val_loss']:.4f}",
            f"{summary_data['summary']['min_val_loss']:.4f}",
            f"{summary_data['summary']['max_val_loss']:.4f}",
            f"{summary_data['summary']['total_training_time_hours']:.2f}",
            f"{sum(early_stopped)}/{len(fold_results)}",
            str(len(fold_results))
        ]
    }
    
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[[row[0], row[1]] for row in zip(summary_stats['Metric'], summary_stats['Value'])],
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Add alternating row colors
    for i in range(1, len(summary_stats['Metric']) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(f'{run_name} Training Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / f'{run_name}_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}:")
    print(f"- {run_name}_analysis.png")
    print(f"- {run_name}_summary_table.png")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Visualize training run metrics')
    parser.add_argument('run_name', help='Name of the run to visualize')
    parser.add_argument('--output-dir', help='Output directory for visualizations', default=None)
    
    args = parser.parse_args()
    
    try:
        create_visualizations(args.run_name, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # If no arguments provided, default to production_v1
    import sys
    if len(sys.argv) == 1:
        create_visualizations('production_v1')
    else:
        sys.exit(main())