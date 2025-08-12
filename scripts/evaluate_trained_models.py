#!/usr/bin/env python3
"""
Re-evaluate trained models with corrected IoU and Dice metrics
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import argparse

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet
from src.models.accuracy_metrics import calculate_multiclass_iou, calculate_dice_coefficient, calculate_binary_iou
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from torch import sigmoid


def find_best_checkpoint(fold_dir):
    """Find the best checkpoint in a fold directory based on validation loss"""
    checkpoints = list(fold_dir.glob("utooth-epoch=*-val_loss=*.ckpt"))
    if not checkpoints:
        return None
    
    # Sort by validation loss (extract from filename)
    best_ckpt = min(checkpoints, key=lambda x: float(x.stem.split('val_loss=')[1]))
    return best_ckpt


def evaluate_checkpoint(checkpoint_path, data_module, fold_idx, val_indices, device='cuda'):
    """Evaluate a single checkpoint and return metrics"""
    # Load model
    model = UNet.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    # Get validation dataloader for this fold
    val_dataloader = data_module.val_dataloader()
    
    # Metrics storage
    all_ious = []
    all_dice = []
    all_binary_ious = []
    all_losses = []
    
    with torch.no_grad():
        for x, y in tqdm(val_dataloader, desc=f"Evaluating fold {fold_idx}"):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            pred_sigmoid = sigmoid(logits)
            
            # Calculate loss
            loss = model.focal_tversky_loss(logits, y)
            all_losses.append(loss.item())
            
            # Calculate metrics
            iou = calculate_multiclass_iou(pred_sigmoid, y, num_classes=4, threshold=0.5)
            dice = calculate_dice_coefficient(pred_sigmoid, y, num_classes=4, threshold=0.5)
            binary_iou = calculate_binary_iou(pred_sigmoid, y, threshold=0.5)
            
            all_ious.append(iou.item())
            all_dice.append(dice.item())
            all_binary_ious.append(binary_iou.item())
    
    # Calculate mean metrics
    metrics = {
        'iou': np.mean(all_ious),
        'dice': np.mean(all_dice),
        'binary_iou': np.mean(all_binary_ious),
        'val_loss': np.mean(all_losses),
        'iou_std': np.std(all_ious),
        'dice_std': np.std(all_dice),
        'binary_iou_std': np.std(all_binary_ious)
    }
    
    return metrics


def evaluate_run(run_path, data_path='/home/gaetano/utooth/DATA', device='cuda'):
    """Evaluate all folds in a training run"""
    import glob
    from sklearn.model_selection import KFold
    
    run_path = Path(run_path)
    checkpoints_dir = run_path / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print(f"No checkpoints directory found in {run_path}")
        return None
    
    # Determine number of folds
    fold_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    n_folds = len(fold_dirs)
    
    if n_folds == 0:
        print(f"No fold directories found in {checkpoints_dir}")
        return None
    
    # Create k-fold indices
    case_folders = glob.glob(os.path.join(data_path, 'case-*'))
    n_samples = len(case_folders)
    indices = np.arange(n_samples)
    
    # Determine random seed from experiment state if available
    random_seed = 42  # default
    experiment_state_path = run_path / 'experiment_state.json'
    if experiment_state_path.exists():
        with open(experiment_state_path, 'r') as f:
            state = json.load(f)
            random_seed = state.get('random_seed', 42)
    
    # Create fold splits
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_splits = list(kfold.split(indices))
    
    # Evaluate each fold
    results = {}
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split('_')[1])
        
        # Find best checkpoint
        best_ckpt = find_best_checkpoint(fold_dir)
        if best_ckpt is None:
            print(f"No checkpoint found for fold {fold_idx}")
            continue
        
        print(f"\nEvaluating {run_path.name}, Fold {fold_idx}")
        print(f"Checkpoint: {best_ckpt.name}")
        
        # Get fold indices
        train_indices, val_indices = fold_splits[fold_idx]
        
        # Create data module for this fold
        data_module = CTScanDataModuleKFold(
            data_dir=data_path,
            batch_size=1,
            num_workers=4,
            train_indices=train_indices.tolist(),
            val_indices=val_indices.tolist()
        )
        data_module.setup()
        
        # Evaluate
        metrics = evaluate_checkpoint(best_ckpt, data_module, fold_idx, val_indices, device)
        results[fold_idx] = {
            'checkpoint': best_ckpt.name,
            **metrics
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate trained models with corrected metrics')
    parser.add_argument('--run', help='Specific run to evaluate (e.g., utooth_10f_v3)', default=None)
    parser.add_argument('--all', action='store_true', help='Evaluate all runs')
    parser.add_argument('--data-path', default='/home/gaetano/utooth/DATA/', help='Path to data directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output-dir', default='/home/gaetano/utooth/outputs/analysis', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = Path('/home/gaetano/utooth/outputs/runs')
    
    # Determine which runs to evaluate
    if args.run:
        runs_to_evaluate = [runs_dir / args.run]
    elif args.all:
        runs_to_evaluate = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('utooth_')])
    else:
        print("Please specify --run or --all")
        return
    
    # Evaluate runs
    all_results = {}
    for run_path in runs_to_evaluate:
        if not run_path.exists():
            print(f"Run directory {run_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating run: {run_path.name}")
        print(f"{'='*60}")
        
        results = evaluate_run(run_path, args.data_path, args.device)
        if results:
            all_results[run_path.name] = results
    
    # Save detailed results
    with open(output_dir / 'corrected_metrics_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary
    summary = []
    for run_name, fold_results in all_results.items():
        run_metrics = {
            'run': run_name,
            'n_folds': len(fold_results),
            'mean_iou': np.mean([m['iou'] for m in fold_results.values()]),
            'std_iou': np.std([m['iou'] for m in fold_results.values()]),
            'mean_dice': np.mean([m['dice'] for m in fold_results.values()]),
            'std_dice': np.std([m['dice'] for m in fold_results.values()]),
            'mean_binary_iou': np.mean([m['binary_iou'] for m in fold_results.values()]),
            'best_fold_iou': max(fold_results.items(), key=lambda x: x[1]['iou'])[0],
            'best_iou': max(m['iou'] for m in fold_results.values()),
            'best_fold_dice': max(fold_results.items(), key=lambda x: x[1]['dice'])[0],
            'best_dice': max(m['dice'] for m in fold_results.values())
        }
        summary.append(run_metrics)
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'corrected_metrics_summary.csv', index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run':<20} {'Folds':<8} {'Mean IoU':<12} {'Best IoU':<12} {'Mean Dice':<12} {'Best Dice':<12}")
    print(f"{'-'*80}")
    
    for run in summary:
        print(f"{run['run']:<20} {run['n_folds']:<8} "
              f"{run['mean_iou']:.4f}±{run['std_iou']:.4f}  "
              f"{run['best_iou']:.4f}       "
              f"{run['mean_dice']:.4f}±{run['std_dice']:.4f}  "
              f"{run['best_dice']:.4f}")
    
    print(f"\nResults saved to {output_dir}")
    print(f"- Detailed results: corrected_metrics_detailed.json")
    print(f"- Summary: corrected_metrics_summary.csv")


if __name__ == "__main__":
    main()