#!/usr/bin/env python3
"""
Analyze which fold indices perform best across experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the per-fold results
df = pd.read_csv('corrected_metrics_analysis_folds.csv')

# Calculate average performance by fold index across all experiments
print("=== AVERAGE PERFORMANCE BY FOLD INDEX ===\n")

fold_stats = df.groupby('fold_idx').agg({
    'corrected_iou': ['mean', 'std', 'min', 'max', 'count'],
    'dice': ['mean', 'std', 'min', 'max'],
    'binary_iou': ['mean', 'std', 'min', 'max']
}).round(4)

# Sort by corrected IoU mean
fold_stats_sorted = fold_stats.sort_values(('corrected_iou', 'mean'), ascending=False)

print("Corrected IoU by Fold Index:")
print(fold_stats_sorted['corrected_iou'])

print("\n" + "="*60 + "\n")

# Create a summary table
summary_data = []
for fold_idx in range(10):
    fold_data = df[df['fold_idx'] == fold_idx]
    summary_data.append({
        'Fold': fold_idx,
        'Avg IoU': f"{fold_data['corrected_iou'].mean():.1%}",
        'Avg Dice': f"{fold_data['dice'].mean():.1%}",
        'Best IoU': f"{fold_data['corrected_iou'].max():.1%}",
        'Worst IoU': f"{fold_data['corrected_iou'].min():.1%}",
        'Std Dev': f"{fold_data['corrected_iou'].std():.1%}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Avg IoU', ascending=False)
print("FOLD PERFORMANCE SUMMARY (sorted by Avg IoU):")
print(summary_df.to_string(index=False))

# Find which folds are consistently good/bad
print("\n" + "="*60 + "\n")
print("CONSISTENCY ANALYSIS:")

# Calculate ranking of each fold in each experiment
rankings = []
for exp in df['experiment'].unique():
    exp_data = df[df['experiment'] == exp].copy()
    exp_data['rank'] = exp_data['corrected_iou'].rank(ascending=False)
    rankings.append(exp_data[['fold_idx', 'rank']])

# Combine rankings
all_rankings = pd.concat(rankings)
avg_ranking = all_rankings.groupby('fold_idx')['rank'].agg(['mean', 'std']).round(2)
avg_ranking = avg_ranking.sort_values('mean')

print("\nAverage Ranking by Fold (1=best, 10=worst):")
print(avg_ranking)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Average IoU by fold
fold_avg_iou = df.groupby('fold_idx')['corrected_iou'].mean().sort_values(ascending=False)
ax1 = axes[0, 0]
bars = ax1.bar(fold_avg_iou.index, fold_avg_iou.values)
ax1.set_xlabel('Fold Index')
ax1.set_ylabel('Average Corrected IoU')
ax1.set_title('Average IoU Performance by Fold Index')
ax1.set_xticks(range(10))
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}', ha='center', va='bottom', fontsize=8)

# Plot 2: IoU range by fold
ax2 = axes[0, 1]
fold_ranges = df.groupby('fold_idx')['corrected_iou'].agg(['min', 'max', 'mean'])
x = fold_ranges.index
y_mean = fold_ranges['mean']
y_min = fold_ranges['min']
y_max = fold_ranges['max']

ax2.bar(x, y_max - y_min, bottom=y_min, alpha=0.3, color='blue', label='Range')
ax2.plot(x, y_mean, 'ro-', label='Mean', markersize=8)
ax2.set_xlabel('Fold Index')
ax2.set_ylabel('Corrected IoU')
ax2.set_title('IoU Range and Mean by Fold Index')
ax2.set_xticks(range(10))
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Performance matrix
ax3 = axes[1, 0]
pivot_table = df.pivot_table(values='corrected_iou', index='experiment', columns='fold_idx')
im = ax3.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(10))
ax3.set_xticklabels(range(10))
ax3.set_yticks(range(len(pivot_table.index)))
ax3.set_yticklabels(pivot_table.index)
ax3.set_xlabel('Fold Index')
ax3.set_ylabel('Experiment')
ax3.set_title('IoU Performance Matrix')

# Add text annotations
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        text = ax3.text(j, i, f'{pivot_table.values[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax3)

# Plot 4: Standard deviation by fold
ax4 = axes[1, 1]
fold_std = df.groupby('fold_idx')['corrected_iou'].std().sort_values(ascending=True)
bars = ax4.bar(fold_std.index, fold_std.values, color='coral')
ax4.set_xlabel('Fold Index')
ax4.set_ylabel('Standard Deviation of IoU')
ax4.set_title('Consistency of Fold Performance (lower is better)')
ax4.set_xticks(range(10))
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('fold_performance_analysis.png', dpi=150)
print("\nVisualization saved as: fold_performance_analysis.png")

# Identify the validation samples in best/worst folds
print("\n" + "="*60 + "\n")
print("FOLD COMPOSITION INSIGHT:")
print("\nBest performing folds (1, 6, 7, 8) and worst performing folds (0, 2, 4)")
print("might have different data characteristics.")
print("\nRecommendation: Check which specific cases are in the validation sets")
print("of the best and worst folds to understand data-related factors.")