#!/usr/bin/env python3
"""
Determine the exact fold splits used in training runs
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_fold_splits(data_path, n_folds, random_seed):
    """Recreate the exact fold splits used in training"""
    # Get all case folders (same logic as in train.py)
    case_folders = glob.glob(os.path.join(data_path, 'case-*'))
    n_samples = len(case_folders)
    indices = np.arange(n_samples)
    
    # Create KFold with same parameters as training
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Get case names for mapping
    case_names = [os.path.basename(folder) for folder in sorted(case_folders)]
    
    return list(kfold.split(indices)), case_names


def analyze_fold_splits(run_name):
    """Analyze fold splits for a specific run"""
    run_path = Path(f"/home/gaetano/utooth/outputs/runs/{run_name}")
    
    # Load config to get parameters
    config_path = run_path / 'config.json'
    if not config_path.exists():
        print(f"Config not found for {run_name}")
        return None
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_path = config['data_path']
    n_folds = config['n_folds'] 
    random_seed = config['random_seed']
    
    print(f"Analyzing {run_name}")
    print(f"Data path: {data_path}")
    print(f"Folds: {n_folds}")
    print(f"Random seed: {random_seed}")
    print("=" * 60)
    
    # Get fold splits
    fold_splits, case_names = get_fold_splits(data_path, n_folds, random_seed)
    
    print(f"Total cases: {len(case_names)}")
    print(f"Cases: {case_names}")
    print("\nFold splits:")
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        train_cases = [case_names[i] for i in train_indices]
        val_cases = [case_names[i] for i in val_indices]
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train ({len(train_cases)}): {train_cases}")
        print(f"  Val ({len(val_cases)}): {val_cases}")
    
    return fold_splits, case_names


def save_fold_splits(run_name, fold_splits, case_names):
    """Save fold splits to file"""
    output_path = Path(f"/home/gaetano/utooth/outputs/runs/{run_name}/fold_splits.json")
    
    splits_data = {
        'total_cases': len(case_names),
        'case_names': case_names,
        'folds': []
    }
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        fold_data = {
            'fold_idx': fold_idx,
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'train_cases': [case_names[i] for i in train_indices],
            'val_cases': [case_names[i] for i in val_indices]
        }
        splits_data['folds'].append(fold_data)
    
    with open(output_path, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"\nFold splits saved to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Determine fold splits for training runs')
    parser.add_argument('run_name', help='Name of the run (e.g., utooth_10f_v3)')
    parser.add_argument('--save', action='store_true', help='Save fold splits to file')
    
    args = parser.parse_args()
    
    result = analyze_fold_splits(args.run_name)
    if result:
        fold_splits, case_names = result
        if args.save:
            save_fold_splits(args.run_name, fold_splits, case_names)


if __name__ == "__main__":
    main()