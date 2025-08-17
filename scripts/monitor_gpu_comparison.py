#!/usr/bin/env python3
"""
Monitor GPU Comparison Training Progress
=========================================

Monitors the progress of seed 2113 training across multiple GPUs in real-time,
comparing each GPU's performance to the original seed 2113 results.
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def get_gpu_progress(gpu_dir, gpu_id):
    """Get current progress for a single GPU (all its folds)"""
    gpu_dir = Path(gpu_dir)
    
    if not gpu_dir.exists():
        return None
    
    # Check for overall GPU results
    gpu_results_file = gpu_dir / "gpu_results.json"
    if gpu_results_file.exists():
        with open(gpu_results_file) as f:
            return json.load(f)
    
    # Otherwise, collect progress from individual folds
    fold_dirs = sorted([d for d in gpu_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        return {
            'gpu_id': gpu_id,
            'status': 'waiting',
            'completed_folds': 0,
            'current_fold': None,
            'fold_results': []
        }
    
    fold_results = []
    completed_folds = 0
    current_fold = None
    
    for fold_dir in fold_dirs:
        fold_num = int(fold_dir.name.split('_')[1])
        
        # Check for fold completion
        fold_results_file = fold_dir / "results.json"
        if fold_results_file.exists():
            with open(fold_results_file) as f:
                fold_result = json.load(f)
                fold_results.append(fold_result)
                completed_folds += 1
        else:
            # Check if fold is currently training
            metrics_file = fold_dir / "metrics.csv"
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    val_df = df[df['val_dice'].notna()]
                    
                    if len(val_df) > 0:
                        current_epoch = val_df['epoch'].iloc[-1]
                        current_dice = val_df['val_dice'].iloc[-1]
                        best_dice = val_df['val_dice'].max()
                        
                        current_fold = {
                            'fold': fold_num,
                            'status': 'training',
                            'current_epoch': int(current_epoch),
                            'current_dice': float(current_dice),
                            'best_dice': float(best_dice)
                        }
                    else:
                        current_fold = {
                            'fold': fold_num,
                            'status': 'starting',
                            'current_epoch': 0,
                            'current_dice': 0,
                            'best_dice': 0
                        }
                except Exception:
                    current_fold = {
                        'fold': fold_num,
                        'status': 'error',
                        'current_epoch': 0,
                        'current_dice': 0,
                        'best_dice': 0
                    }
                break  # Only track the first active fold
    
    # Calculate current stats from completed folds
    if fold_results:
        dice_scores = [r['val_dice'] for r in fold_results]
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
    else:
        mean_dice = 0
        std_dice = 0
    
    return {
        'gpu_id': gpu_id,
        'status': 'completed' if completed_folds == 10 else 'training',
        'completed_folds': completed_folds,
        'current_fold': current_fold,
        'fold_results': fold_results,
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'timestamp': datetime.now().isoformat()
    }


def monitor_gpu_comparison(output_dir, refresh_interval=15):
    """Monitor GPU comparison training progress"""
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
        n_gpus = config.get('n_gpus', 3)
    else:
        seed = 'unknown'
        max_epochs = 150
        n_gpus = 3
    
    # Original seed 2113 results for comparison
    original_results = {
        'mean_dice': 0.8310940742492676,
        'std_dice': 0.060907994079069036,
        'fold_scores': [0.8785, 0.8097, 0.8654, 0.8836, 0.8086, 0.7570, 0.7038, 0.8158, 0.8914, 0.8973]
    }
    
    print(f"Monitoring GPU comparison for seed {seed} ({max_epochs} epochs)")
    print(f"Output directory: {output_dir}")
    print(f"Tracking {n_gpus} GPUs")
    print(f"Original mean Dice: {original_results['mean_dice']:.4f}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"{'='*100}")
            print(f"GPU COMPARISON MONITOR - Seed {seed} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*100}")
            print(f"Original Seed {seed}: {original_results['mean_dice']:.4f} ± {original_results['std_dice']:.4f}")
            print(f"{'='*100}\n")
            
            # Get GPU directories
            gpu_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('gpu_')])
            
            if not gpu_dirs:
                print("No GPU directories found yet. Waiting for training to start...")
            else:
                all_gpu_results = []
                completed_gpus = 0
                
                # Print header
                print(f"{'GPU':<5} {'Status':<12} {'Folds':<8} {'Current':<25} {'Mean Dice':<12} {'vs Original':<12}")
                print("-" * 100)
                
                for gpu_dir in gpu_dirs:
                    gpu_id = int(gpu_dir.name.split('_')[1])
                    progress = get_gpu_progress(gpu_dir, gpu_id)
                    
                    if progress is None:
                        print(f"{gpu_id:<5} {'waiting':<12} {'--':<8} {'--':<25} {'--':<12} {'--':<12}")
                        continue
                    
                    status = progress['status']
                    completed_folds = progress['completed_folds']
                    current_fold = progress.get('current_fold')
                    
                    # Format current activity
                    if current_fold:
                        current_str = f"Fold {current_fold['fold']} ep{current_fold.get('current_epoch', 0)} ({current_fold.get('current_dice', 0):.3f})"
                    elif status == 'completed':
                        current_str = "✓ ALL COMPLETE"
                    else:
                        current_str = "waiting..."
                    
                    # Format mean dice and comparison
                    if completed_folds > 0:
                        mean_dice = progress['mean_dice']
                        improvement = mean_dice - original_results['mean_dice']
                        
                        mean_dice_str = f"{mean_dice:.4f}"
                        vs_original_str = f"{'+' if improvement > 0 else ''}{improvement:.4f}"
                        
                        all_gpu_results.append(progress)
                    else:
                        mean_dice_str = "--"
                        vs_original_str = "--"
                    
                    if status == 'completed':
                        completed_gpus += 1
                        status_str = "✓ DONE"
                    else:
                        status_str = status
                    
                    folds_str = f"{completed_folds}/10"
                    
                    print(f"{gpu_id:<5} {status_str:<12} {folds_str:<8} {current_str:<25} {mean_dice_str:<12} {vs_original_str:<12}")
                
                # Print detailed fold comparison for ALL GPUs (including in-progress)
                print(f"\n{'-'*100}")
                print("FOLD-BY-FOLD COMPARISON (All GPUs vs Original):")
                
                # Get all GPU results that have at least some data
                all_gpu_data = [r for r in all_gpu_results if r['completed_folds'] > 0 or r.get('current_fold')]
                
                if all_gpu_data:
                    # Header
                    print(f"{'Fold':<6} {'Original':<10}", end="")
                    for gpu_result in all_gpu_data:
                        print(f"GPU {gpu_result['gpu_id']:<10}", end="")
                    print()
                    
                    print("-" * (16 + len(all_gpu_data) * 11))
                    
                    # Print each fold
                    for fold_idx in range(10):
                        original_score = original_results['fold_scores'][fold_idx]
                        print(f"{fold_idx:<6} {original_score:<10.4f}", end="")
                        
                        for gpu_result in all_gpu_data:
                            # Check if this fold is completed
                            fold_score = None
                            fold_status = "pending"
                            
                            # First check completed folds
                            for fold_res in gpu_result.get('fold_results', []):
                                if fold_res['fold'] == fold_idx:
                                    fold_score = fold_res['val_dice']
                                    fold_status = "done"
                                    break
                            
                            # If not completed, check if it's currently training
                            current_fold = gpu_result.get('current_fold')
                            if fold_score is None and current_fold and current_fold['fold'] == fold_idx:
                                if current_fold['status'] == 'training':
                                    fold_score = current_fold.get('best_dice', current_fold.get('current_dice', 0))
                                    fold_status = f"ep{current_fold.get('current_epoch', 0)}"
                                elif current_fold['status'] == 'starting':
                                    fold_status = "starting"
                            
                            # Format output
                            if fold_score is not None and fold_score > 0:
                                if fold_status == "done":
                                    # Completed fold - show score and improvement
                                    improvement = fold_score - original_score
                                    color_code = "✓" if improvement >= 0 else "✗"
                                    print(f"{fold_score:<6.4f}{color_code:<4}", end="")
                                else:
                                    # Currently training - show current best
                                    print(f"{fold_score:<6.4f}~{fold_status:<3}", end="")
                            elif fold_status == "starting":
                                print(f"{'start':<10}", end="")
                            elif fold_status == "pending":
                                print(f"{'--':<10}", end="")
                            else:
                                print(f"{'--':<10}", end="")
                        print()
                    
                    # Summary statistics
                    print()
                    print(f"{'Progress':<6} {'10/10':<10}", end="")
                    for gpu_result in all_gpu_data:
                        completed = gpu_result['completed_folds']
                        current_fold = gpu_result.get('current_fold')
                        if current_fold and current_fold['status'] in ['training', 'starting']:
                            progress_str = f"{completed}+1/10"
                        else:
                            progress_str = f"{completed}/10"
                        print(f"{progress_str:<10}", end="")
                    print()
                    
                    # Current mean (for completed folds only) with improvement indicators
                    print(f"{'CurMean':<6} {original_results['mean_dice']:<10.4f}", end="")
                    for gpu_result in all_gpu_data:
                        if gpu_result['completed_folds'] > 0:
                            mean_dice = gpu_result['mean_dice']
                            improvement = mean_dice - original_results['mean_dice']
                            indicator = "✓" if improvement >= 0 else "✗"
                            print(f"{mean_dice:<6.4f}{indicator:<4}", end="")
                        else:
                            print(f"{'--':<10}", end="")
                    print()
                    
                    # Detailed improvement vs original
                    print(f"{'Δ Mean':<6} {'0.0000':<10}", end="")
                    for gpu_result in all_gpu_data:
                        if gpu_result['completed_folds'] > 0:
                            improvement = gpu_result['mean_dice'] - original_results['mean_dice']
                            print(f"{improvement:<+10.4f}", end="")
                        else:
                            print(f"{'--':<10}", end="")
                    print()
                    
                    # Standard deviation comparison
                    print(f"{'StdDev':<6} {original_results['std_dice']:<10.4f}", end="")
                    for gpu_result in all_gpu_data:
                        if gpu_result['completed_folds'] > 1:  # Need at least 2 folds for std
                            print(f"{gpu_result['std_dice']:<10.4f}", end="")
                        else:
                            print(f"{'--':<10}", end="")
                    print()
                    
                    # Fold improvement summary (how many folds are better/worse)
                    print(f"{'Better':<6} {'--':<10}", end="")
                    for gpu_result in all_gpu_data:
                        if gpu_result['completed_folds'] > 0:
                            better_count = 0
                            total_count = 0
                            for fold_res in gpu_result.get('fold_results', []):
                                fold_idx = fold_res['fold']
                                original_fold_score = original_results['fold_scores'][fold_idx]
                                if fold_res['val_dice'] >= original_fold_score:
                                    better_count += 1
                                total_count += 1
                            print(f"{better_count}/{total_count}    ", end="")
                        else:
                            print(f"{'--':<10}", end="")
                    print()
                
                # Show completion status
                completed_gpu_results = [r for r in all_gpu_results if r['status'] == 'completed']
                
                # Check variance across GPUs
                if len(completed_gpu_results) >= 2:
                    mean_dices = [r['mean_dice'] for r in completed_gpu_results]
                    variance_across_gpus = np.var(mean_dices)
                    
                    print(f"\nVariance across completed GPUs: {variance_across_gpus:.6f}")
                    if variance_across_gpus < 1e-6:
                        print("→ Results are essentially identical across GPUs")
                    else:
                        print("→ Results vary across GPUs - GPU assignment matters!")
                
                # Check if all completed
                if completed_gpus == n_gpus:
                    print(f"\n{'='*100}")
                    print("ALL GPUs COMPLETED!")
                    
                    # Load final comparison if available
                    final_comparison_path = output_dir / "comparison_results.json"
                    if final_comparison_path.exists():
                        with open(final_comparison_path) as f:
                            final = json.load(f)
                        print(f"Final comparison saved to: {final_comparison_path}")
                    break
            
            print(f"\n(Refreshing every {refresh_interval} seconds...)")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU comparison training progress')
    parser.add_argument('output_dir', type=str, help='Output directory to monitor')
    parser.add_argument('--interval', type=int, default=15, 
                        help='Refresh interval in seconds (default: 15)')
    
    args = parser.parse_args()
    monitor_gpu_comparison(args.output_dir, args.interval)


if __name__ == "__main__":
    main()