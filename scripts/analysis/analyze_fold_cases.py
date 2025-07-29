#!/usr/bin/env python3
"""
Analyze which cases are in the best and worst performing folds
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

# Configuration for the experiments (they all use the same seed)
data_path = '/home/gaetano/utooth/DATA/'
n_folds = 10
random_seed = 2025  # From utooth_10f_v3 which is one of the best

# Get all case folders
data_dirs = sorted([d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
case_names = [d.name for d in data_dirs]
n_samples = len(data_dirs)

print(f"Total cases: {n_samples}")
print(f"Cases: {', '.join(case_names[:5])}... (showing first 5)")

# Create fold indices
indices = np.arange(n_samples)
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
fold_splits = list(kfold.split(indices))

# Define best and worst folds based on our analysis
best_folds = [6, 8, 1, 7]  # Average IoU: 80.4%, 79.3%, 77.2%, 75.6%
worst_folds = [0, 3, 2]     # Average IoU: 65.9%, 69.4%, 70.0%

print("\n" + "="*60)
print("BEST PERFORMING FOLDS")
print("="*60)

for fold_idx in best_folds:
    _, val_indices = fold_splits[fold_idx]
    val_cases = [case_names[i] for i in val_indices]
    print(f"\nFold {fold_idx} (Avg IoU: varies by experiment):")
    print(f"Validation cases ({len(val_cases)}): {', '.join(val_cases)}")

print("\n" + "="*60)
print("WORST PERFORMING FOLDS")
print("="*60)

for fold_idx in worst_folds:
    _, val_indices = fold_splits[fold_idx]
    val_cases = [case_names[i] for i in val_indices]
    print(f"\nFold {fold_idx} (Avg IoU: varies by experiment):")
    print(f"Validation cases ({len(val_cases)}): {', '.join(val_cases)}")

# Analyze case frequency in best vs worst folds
print("\n" + "="*60)
print("CASE DIFFICULTY ANALYSIS")
print("="*60)

# Count how often each case appears in best/worst validation sets
case_in_best = {}
case_in_worst = {}

for case in case_names:
    case_in_best[case] = 0
    case_in_worst[case] = 0

for fold_idx in best_folds:
    _, val_indices = fold_splits[fold_idx]
    for idx in val_indices:
        case_in_best[case_names[idx]] += 1

for fold_idx in worst_folds:
    _, val_indices = fold_splits[fold_idx]
    for idx in val_indices:
        case_in_worst[case_names[idx]] += 1

# Find cases that appear only in worst folds
difficult_cases = [case for case in case_names if case_in_worst[case] > 0 and case_in_best[case] == 0]
easy_cases = [case for case in case_names if case_in_best[case] > 0 and case_in_worst[case] == 0]

print(f"\nPotentially difficult cases (appear in worst folds but not best):")
if difficult_cases:
    for case in difficult_cases:
        print(f"  - {case}")
else:
    print("  None found")

print(f"\nPotentially easy cases (appear in best folds but not worst):")
if easy_cases:
    for case in easy_cases[:10]:  # Show first 10
        print(f"  - {case}")
    if len(easy_cases) > 10:
        print(f"  ... and {len(easy_cases) - 10} more")
else:
    print("  None found")

# Check for overlap
overlap_cases = [case for case in case_names if case_in_best[case] > 0 and case_in_worst[case] > 0]
print(f"\nCases appearing in both best and worst folds: {len(overlap_cases)}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("The fold performance differences might be due to:")
print("1. Random variation in data difficulty")
print("2. Specific challenging cases in validation sets")
print("3. Class imbalance in certain folds")
print("\nTo improve model performance:")
print("- Consider stratified k-fold to ensure balanced class distribution")
print("- Analyze the difficult cases for common patterns")
print("- Use ensemble of best folds for production")