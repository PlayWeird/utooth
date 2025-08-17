#!/usr/bin/env python3
"""
Monitor Retraining Progress
============================

Monitors the progress of seed retraining in real-time.
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def get_fold_progress(fold_dir):
    """Get current progress for a single fold"""
    metrics_file = fold_dir / "metrics.csv"
    results_file = fold_dir / "results.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        # Read metrics CSV
        df = pd.read_csv(metrics_file)
        val_df = df[df['val_dice'].notna()]
        
        if len(val_df) == 0:
            return {
                'status': 'training',
                'current_epoch': 0,
                'best_dice': 0,
                'best_epoch': 0,
                'last_dice': 0
            }
        
        current_epoch = val_df['epoch'].iloc[-1]
        best_idx = val_df['val_dice'].idxmax()
        best_dice = val_df.loc[best_idx, 'val_dice']
        best_epoch = val_df.loc[best_idx, 'epoch']
        last_dice = val_df['val_dice'].iloc[-1]
        
        # Check if completed
        status = 'training'
        if results_file.exists():
            status = 'completed'
        
        return {
            'status': status,
            'current_epoch': int(current_epoch),
            'best_dice': float(best_dice),
            'best_epoch': int(best_epoch),
            'last_dice': float(last_dice),
            'val_loss': float(val_df['val_loss'].iloc[-1]) if 'val_loss' in val_df.columns else None
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def monitor_training(output_dir, refresh_interval=10):
    """Monitor training progress"""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist!")
        return
    
    # Load config
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        seed = config.get('seed', 'unknown')
        max_epochs = config.get('max_epochs', 150)
    else:
        seed = 'unknown'
        max_epochs = 150
    
    # Load original results for this seed
    original_results_path = Path(f"outputs/seed_search/seed_search_20250815_104509/seeds/seed_{seed}/results.json")
    
    if original_results_path.exists():
        with open(original_results_path) as f:
            original_data = json.load(f)
        
        # Extract fold scores and epochs from original data
        fold_scores = []
        fold_epochs = []
        for fold_metric in original_data['fold_metrics']:
            fold_scores.append(fold_metric['val_dice'])
            fold_epochs.append(fold_metric['epoch'])
        
        original_results = {
            'mean_dice': original_data['mean_dice'],
            'std_dice': original_data['std_dice'],
            'fold_scores': fold_scores,
            'fold_epochs': fold_epochs
        }
    else:
        print(f"Warning: Could not find original results for seed {seed}")
        print(f"Expected path: {original_results_path}")
        # Fallback to empty results
        original_results = {
            'mean_dice': 0,
            'std_dice': 0,
            'fold_scores': [0] * 10,
            'fold_epochs': [0] * 10
        }
    
    print(f"Monitoring retraining of seed {seed} ({max_epochs} epochs)")
    print(f"Output directory: {output_dir}")
    print(f"Original mean Dice: {original_results['mean_dice']:.4f}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen (works on Unix-like systems)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"{'='*80}")
            print(f"RETRAINING MONITOR - Seed {seed} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            # Get all fold directories
            fold_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
            
            if not fold_dirs:
                print("No fold directories found yet. Waiting for training to start...")
            else:
                all_results = []
                completed_folds = 0
                
                # Print header with original comparison
                print(f"{'Fold':<6} {'Original':<10} {'OrigEp':<8} {'Status':<12} {'Progress':<15} {'Current':<10} {'Best':<10} {'vs Orig':<10} {'NewEp':<8}")
                print("-" * 105)
                
                for fold_dir in fold_dirs:
                    fold_num = int(fold_dir.name.split('_')[1])
                    progress = get_fold_progress(fold_dir)
                    
                    # Get original score for this fold
                    original_score = original_results['fold_scores'][fold_num] if fold_num < len(original_results['fold_scores']) else 0
                    original_epoch = original_results['fold_epochs'][fold_num] if fold_num < len(original_results['fold_epochs']) else 0
                    
                    if progress is None:
                        print(f"{fold_num:<6} {original_score:<10.4f} {original_epoch:<8} {'waiting':<12} {'--':<15} {'--':<10} {'--':<10} {'--':<10} {'--':<8}")
                    elif progress['status'] == 'error':
                        print(f"{fold_num:<6} {original_score:<10.4f} {original_epoch:<8} {'ERROR':<12} {progress['error'][:15]:<15} {'--':<10} {'--':<10} {'--':<10} {'--':<8}")
                    else:
                        epoch_progress = f"{progress['current_epoch']}/{max_epochs}"
                        pct_complete = (progress['current_epoch'] / max_epochs) * 100
                        progress_str = f"{epoch_progress} ({pct_complete:.1f}%)"
                        
                        if progress['status'] == 'completed':
                            completed_folds += 1
                            status_str = "✓ DONE"
                        else:
                            status_str = "training"
                        
                        # Calculate improvement vs original
                        if progress['best_dice'] > 0:
                            improvement = progress['best_dice'] - original_score
                            vs_orig_str = f"{improvement:+.4f}"
                            if improvement >= 0:
                                vs_orig_str += "✓"
                            else:
                                vs_orig_str += "✗"
                        else:
                            vs_orig_str = "--"
                        
                        print(f"{fold_num:<6} {original_score:<10.4f} {original_epoch:<8} {status_str:<12} {progress_str:<15} "
                              f"{progress['last_dice']:<10.4f} {progress['best_dice']:<10.4f} "
                              f"{vs_orig_str:<10} {progress['best_epoch']:<8}")
                        
                        if progress['best_dice'] > 0:
                            all_results.append(progress['best_dice'])
                
                # Print summary statistics
                print(f"\n{'-'*105}")
                if all_results:
                    mean_dice = np.mean(all_results)
                    std_dice = np.std(all_results)
                    
                    # Count folds better than original
                    better_folds = 0
                    for i, result_dice in enumerate(all_results):
                        if i < len(original_results['fold_scores']):
                            if result_dice >= original_results['fold_scores'][i]:
                                better_folds += 1
                    
                    print(f"SUMMARY COMPARISON:")
                    print(f"{'Metric':<20} {'Original':<12} {'Current':<12} {'Improvement':<12}")
                    print("-" * 60)
                    
                    # Mean comparison
                    mean_improvement = mean_dice - original_results['mean_dice']
                    mean_indicator = "✓" if mean_improvement >= 0 else "✗"
                    print(f"{'Mean Dice':<20} {original_results['mean_dice']:<12.4f} {mean_dice:<12.4f} {mean_improvement:+.4f} {mean_indicator}")
                    
                    # Std deviation comparison
                    std_improvement = std_dice - original_results['std_dice']
                    std_indicator = "✓" if std_improvement <= 0 else "✗"  # Lower std is better
                    print(f"{'Std Deviation':<20} {original_results['std_dice']:<12.4f} {std_dice:<12.4f} {std_improvement:+.4f} {std_indicator}")
                    
                    # Fold count comparison
                    print(f"{'Folds Better':<20} {'--':<12} {better_folds}/{len(all_results):<11} {'--':<12}")
                    print(f"{'Completed':<20} {'10/10':<12} {completed_folds}/10{'':<7} {'--':<12}")
                    
                    # Progress indicator
                    if completed_folds == 10:
                        if mean_improvement > 0:
                            print(f"\n🎉 RETRAINING SUCCESSFUL! Improved by {mean_improvement:.4f} Dice score!")
                        else:
                            print(f"\n📊 RETRAINING COMPLETE. Difference: {mean_improvement:.4f} Dice score.")
                    else:
                        print(f"\n⏳ Training in progress... {completed_folds}/10 folds completed")
                
                # Check if all completed
                if completed_folds == 10:
                    print(f"\n{'='*80}")
                    print("ALL FOLDS COMPLETED!")
                    
                    # Load final results if available
                    final_results_path = output_dir / "final_results.json"
                    if final_results_path.exists():
                        with open(final_results_path) as f:
                            final = json.load(f)
                        print(f"Final Mean Dice: {final['mean_dice']:.4f} ± {final['std_dice']:.4f}")
                    break
            
            print(f"\n(Refreshing every {refresh_interval} seconds...)")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description='Monitor retraining progress')
    parser.add_argument('output_dir', type=str, help='Output directory to monitor')
    parser.add_argument('--interval', type=int, default=10, 
                        help='Refresh interval in seconds (default: 10)')
    
    args = parser.parse_args()
    monitor_training(args.output_dir, args.interval)


if __name__ == "__main__":
    main()