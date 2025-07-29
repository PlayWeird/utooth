#!/usr/bin/env python3
"""
Analyze existing training runs with corrected accuracy metrics
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.accuracy_metrics import calculate_multiclass_iou, calculate_dice_coefficient, calculate_binary_iou
from torchmetrics.functional import jaccard_index


def evaluate_checkpoint(checkpoint_path, data_path, val_indices, device):
    """Evaluate a single checkpoint with corrected metrics"""
    
    try:
        # Load model
        model = UNet.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        model = model.to(device)
        
        # Create data loader
        dataset = CTScanDataModuleKFold(
            data_dir=data_path,
            batch_size=2,
            train_indices=[],
            val_indices=val_indices,
            num_workers=2
        )
        dataset.setup()
        val_loader = dataset.val_dataloader()
        
        # Metrics storage
        old_accus = []
        new_ious = []
        dice_scores = []
        binary_ious = []
        val_losses = []
        
        # Process validation set
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                # Forward pass
                logits = model(x)
                pred = torch.sigmoid(logits)
                
                # Calculate loss
                loss = model.focal_tversky_loss(logits, y)
                val_losses.append(loss.item())
                
                # Old accuracy (current implementation)
                old_accu = jaccard_index(pred, y.squeeze(1), task='multiclass', num_classes=4, threshold=0.5)
                old_accus.append(old_accu.item())
                
                # New corrected IoU
                new_iou = calculate_multiclass_iou(pred, y, num_classes=4, threshold=0.5)
                new_ious.append(new_iou.item())
                
                # Dice coefficient
                dice = calculate_dice_coefficient(pred, y, num_classes=4, threshold=0.5)
                dice_scores.append(dice.item())
                
                # Binary IoU
                binary_iou = calculate_binary_iou(pred, y, threshold=0.5)
                binary_ious.append(binary_iou.item())
        
        return {
            'val_loss': np.mean(val_losses),
            'old_accuracy': np.mean(old_accus),
            'corrected_iou': np.mean(new_ious),
            'dice': np.mean(dice_scores),
            'binary_iou': np.mean(binary_ious)
        }
    
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
        return None


def analyze_experiment(experiment_dir, data_path, device):
    """Analyze all folds of an experiment"""
    
    print(f"\nAnalyzing experiment: {os.path.basename(experiment_dir)}")
    
    # Load experiment config
    config_path = os.path.join(experiment_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"  Config not found, skipping...")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load existing results
    results_path = os.path.join(experiment_dir, 'results_summary.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = None
    
    # Create fold indices
    data_dirs = sorted([d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
    n_samples = len(data_dirs)
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_seed'])
    fold_splits = list(kfold.split(indices))
    
    # Analyze each fold
    fold_results = []
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        print(f"  No checkpoints found, skipping...")
        return None
    
    for fold_idx in range(config['n_folds']):
        fold_dir = os.path.join(checkpoints_dir, f'fold_{fold_idx}')
        
        if not os.path.exists(fold_dir):
            print(f"  Fold {fold_idx} not found, skipping...")
            continue
        
        # Find best checkpoint (by val_loss)
        checkpoints = [f for f in os.listdir(fold_dir) if f.endswith('.ckpt') and f != 'last.ckpt' and 'val_loss=' in f]
        
        if not checkpoints:
            # Try to use last.ckpt if no other checkpoints
            if os.path.exists(os.path.join(fold_dir, 'last.ckpt')):
                checkpoints = ['last.ckpt']
            else:
                print(f"  Fold {fold_idx}: No valid checkpoints found")
                continue
        else:
            checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]) if 'val_loss=' in x else float('inf'))
        
        best_checkpoint = os.path.join(fold_dir, checkpoints[0])
        _, val_indices = fold_splits[fold_idx]
        
        print(f"  Evaluating fold {fold_idx}...")
        metrics = evaluate_checkpoint(best_checkpoint, data_path, val_indices, device)
        
        if metrics:
            fold_result = {
                'fold_idx': fold_idx,
                'checkpoint': checkpoints[0],
                **metrics
            }
            
            # Add original reported metrics if available
            if existing_results and 'fold_results' in existing_results:
                for old_fold in existing_results['fold_results']:
                    if old_fold['fold_idx'] == fold_idx:
                        fold_result['original_reported_accu'] = old_fold.get('best_val_accu', None)
                        fold_result['original_reported_loss'] = old_fold.get('best_val_loss', None)
                        break
            
            fold_results.append(fold_result)
    
    if not fold_results:
        return None
    
    # Calculate summary statistics
    summary = {
        'experiment': os.path.basename(experiment_dir),
        'n_folds': len(fold_results),
        'metrics': {
            'corrected_iou': {
                'mean': np.mean([f['corrected_iou'] for f in fold_results]),
                'std': np.std([f['corrected_iou'] for f in fold_results]),
                'min': np.min([f['corrected_iou'] for f in fold_results]),
                'max': np.max([f['corrected_iou'] for f in fold_results])
            },
            'dice': {
                'mean': np.mean([f['dice'] for f in fold_results]),
                'std': np.std([f['dice'] for f in fold_results]),
                'min': np.min([f['dice'] for f in fold_results]),
                'max': np.max([f['dice'] for f in fold_results])
            },
            'binary_iou': {
                'mean': np.mean([f['binary_iou'] for f in fold_results]),
                'std': np.std([f['binary_iou'] for f in fold_results]),
                'min': np.min([f['binary_iou'] for f in fold_results]),
                'max': np.max([f['binary_iou'] for f in fold_results])
            },
            'old_accuracy': {
                'mean': np.mean([f['old_accuracy'] for f in fold_results]),
                'std': np.std([f['old_accuracy'] for f in fold_results]),
                'min': np.min([f['old_accuracy'] for f in fold_results]),
                'max': np.max([f['old_accuracy'] for f in fold_results])
            }
        },
        'fold_results': fold_results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze existing uTooth runs with corrected metrics')
    parser.add_argument('--data_path', type=str, default='/home/gaetano/utooth/DATA/',
                        help='Path to data directory')
    parser.add_argument('--experiments', nargs='*', 
                        help='Specific experiments to analyze (default: all)')
    parser.add_argument('--output', type=str, default='corrected_metrics_analysis.json',
                        help='Output file for analysis results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find experiments to analyze
    runs_dir = 'outputs/runs'
    if args.experiments:
        experiment_dirs = [os.path.join(runs_dir, exp) for exp in args.experiments]
    else:
        experiment_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                          if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('utooth_')]
    
    # Sort experiments
    experiment_dirs.sort()
    
    print(f"Found {len(experiment_dirs)} experiments to analyze")
    
    # Analyze each experiment
    all_results = []
    for exp_dir in experiment_dirs:
        result = analyze_experiment(exp_dir, args.data_path, device)
        if result:
            all_results.append(result)
    
    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY OF CORRECTED METRICS")
    print("="*80)
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Experiment': result['experiment'],
            'Folds': result['n_folds'],
            'Old Accuracy': f"{result['metrics']['old_accuracy']['mean']:.1%} ± {result['metrics']['old_accuracy']['std']:.1%}",
            'Corrected IoU': f"{result['metrics']['corrected_iou']['mean']:.1%} ± {result['metrics']['corrected_iou']['std']:.1%}",
            'Dice Score': f"{result['metrics']['dice']['mean']:.1%} ± {result['metrics']['dice']['std']:.1%}",
            'Binary IoU': f"{result['metrics']['binary_iou']['mean']:.1%} ± {result['metrics']['binary_iou']['std']:.1%}"
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save summary as CSV
        csv_output = args.output.replace('.json', '_summary.csv')
        df.to_csv(csv_output, index=False)
        print(f"\nSummary table saved to: {csv_output}")
    
    # Create detailed per-fold CSV
    all_folds = []
    for result in all_results:
        for fold in result['fold_results']:
            fold_data = {
                'experiment': result['experiment'],
                **fold
            }
            all_folds.append(fold_data)
    
    if all_folds:
        folds_df = pd.DataFrame(all_folds)
        folds_csv = args.output.replace('.json', '_folds.csv')
        folds_df.to_csv(folds_csv, index=False)
        print(f"Per-fold results saved to: {folds_csv}")


if __name__ == "__main__":
    main()