#!/usr/bin/env python3
"""
Find the best performing models from the corrected metrics analysis
"""

import pandas as pd
import numpy as np

# Load the per-fold results
df = pd.read_csv('corrected_metrics_analysis_folds.csv')

# Find top 10 folds by corrected IoU
print("=== TOP 10 FOLDS BY CORRECTED IoU ===")
top_iou = df.nlargest(10, 'corrected_iou')[['experiment', 'fold_idx', 'corrected_iou', 'dice', 'binary_iou', 'old_accuracy']]
print(top_iou.to_string(index=False))

# Find top 10 folds by Dice score
print("\n=== TOP 10 FOLDS BY DICE SCORE ===")
top_dice = df.nlargest(10, 'dice')[['experiment', 'fold_idx', 'corrected_iou', 'dice', 'binary_iou', 'old_accuracy']]
print(top_dice.to_string(index=False))

# Calculate best experiment by average performance
print("\n=== BEST EXPERIMENT BY AVERAGE PERFORMANCE ===")
exp_avg = df.groupby('experiment').agg({
    'corrected_iou': ['mean', 'std', 'max'],
    'dice': ['mean', 'std', 'max'],
    'binary_iou': ['mean', 'std', 'max']
}).round(4)

print("\nAverage Corrected IoU by Experiment:")
print(exp_avg['corrected_iou'].sort_values('mean', ascending=False))

print("\nAverage Dice Score by Experiment:")
print(exp_avg['dice'].sort_values('mean', ascending=False))

# Find the single best fold overall
best_fold = df.loc[df['corrected_iou'].idxmax()]
print(f"\n=== SINGLE BEST FOLD ===")
print(f"Experiment: {best_fold['experiment']}")
print(f"Fold: {best_fold['fold_idx']}")
print(f"Checkpoint: {best_fold['checkpoint']}")
print(f"Corrected IoU: {best_fold['corrected_iou']:.4f} ({best_fold['corrected_iou']*100:.1f}%)")
print(f"Dice Score: {best_fold['dice']:.4f} ({best_fold['dice']*100:.1f}%)")
print(f"Binary IoU: {best_fold['binary_iou']:.4f} ({best_fold['binary_iou']*100:.1f}%)")
print(f"Old (incorrect) accuracy: {best_fold['old_accuracy']:.4f} ({best_fold['old_accuracy']*100:.1f}%)")

# Show the checkpoint path for the best model
print(f"\nBest model checkpoint location:")
print(f"outputs/runs/{best_fold['experiment']}/checkpoints/fold_{int(best_fold['fold_idx'])}/{best_fold['checkpoint']}")