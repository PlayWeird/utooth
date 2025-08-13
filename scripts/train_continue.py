#!/usr/bin/env python3
"""
Continue training from existing checkpoints, ignoring completion status
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train_fold, create_fold_indices

def main():
    parser = argparse.ArgumentParser(description='Continue uTooth training from checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment to continue')
    parser.add_argument('--data_path', type=str, default='/home/gaetano/utooth/DATA/', help='Path to data')
    parser.add_argument('--max_epochs', type=int, default=80, help='Maximum epochs to train to')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--random_seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--no_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--specific_folds', type=int, nargs='+', help='Only train specific folds (e.g., --specific_folds 3 4 5 7)')
    
    args = parser.parse_args()
    
    # Force resume mode
    args.resume = True
    args.auto_resume = True
    args.force_restart = False
    args.no_resume = False
    args.test_run = False
    
    run_dir = os.path.join('outputs/runs', args.experiment_name)
    
    print(f"Continuing training for: {args.experiment_name}")
    print(f"Target epochs: {args.max_epochs}")
    
    # Create fold indices
    fold_indices = create_fold_indices(args.data_path, n_folds=args.n_folds, random_seed=args.random_seed)
    
    # Determine which folds to train
    if args.specific_folds:
        folds_to_train = args.specific_folds
        print(f"Training only folds: {folds_to_train}")
    else:
        folds_to_train = range(args.n_folds)
        print(f"Training all {args.n_folds} folds")
    
    # Train each fold
    fold_results = []
    total_start_time = time.time()
    
    for fold_idx in folds_to_train:
        train_indices, val_indices = fold_indices[fold_idx]
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")
        
        # Remove fold statistics to force retraining
        fold_stats_path = os.path.join(run_dir, 'fold_statistics', f'fold_{fold_idx}_stats.json')
        if os.path.exists(fold_stats_path):
            os.remove(fold_stats_path)
            print(f"Removed existing fold statistics to enable continuation")
        
        try:
            fold_stats, _ = train_fold(fold_idx, train_indices, val_indices, args.data_path, args, run_dir)
            fold_results.append(fold_stats)
            
            print(f"\nFold {fold_idx + 1} completed:")
            print(f"  Best validation loss: {fold_stats['best_val_loss']:.4f}")
            print(f"  Best validation accuracy: {fold_stats['best_val_accu']:.4f}")
            print(f"  Best epoch: {fold_stats['best_epoch']}")
            print(f"  Training time: {fold_stats['training_time_seconds']/60:.1f} minutes")
            
        except Exception as e:
            print(f"Error training fold {fold_idx}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Print summary
    if fold_results:
        val_losses = [stats['best_val_loss'] for stats in fold_results]
        val_accus = [stats['best_val_accu'] for stats in fold_results]
        
        print(f"\n{'='*70}")
        print("CONTINUATION TRAINING RESULTS")
        print(f"{'='*70}")
        for stats in fold_results:
            print(f"Fold {stats['fold_idx'] + 1}: Val Loss = {stats['best_val_loss']:.4f} | "
                  f"Val Accu = {stats['best_val_accu']:.4f} | "
                  f"Best Epoch = {stats['best_epoch']} | "
                  f"Time = {stats['training_time_seconds']/60:.1f} min")
        print(f"{'='*70}")
        print(f"Average validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Average validation accuracy: {np.mean(val_accus):.4f} ± {np.std(val_accus):.4f}")
        print(f"Total continuation time: {total_time/3600:.2f} hours")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()