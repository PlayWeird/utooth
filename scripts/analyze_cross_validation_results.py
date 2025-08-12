#!/usr/bin/env python3
"""
Analyze cross-validation results from uTooth training runs
"""

import os
import csv
import numpy as np
from pathlib import Path


def extract_final_metrics(metrics_file):
    """Extract the final validation accuracy and loss from a metrics CSV file"""
    if not os.path.exists(metrics_file):
        return None, None
    
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find the last row with validation metrics
    for row in reversed(rows):
        if row.get('val_accu') and row.get('val_loss'):
            try:
                val_acc = float(row['val_accu'])
                val_loss = float(row['val_loss'])
                return val_acc, val_loss
            except ValueError:
                continue
    
    return None, None


def analyze_run(run_path):
    """Analyze a single training run with multiple folds"""
    metrics_dir = Path(run_path) / 'metrics'
    if not metrics_dir.exists():
        return None
    
    results = {}
    for fold_dir in sorted(metrics_dir.glob('fold_*')):
        fold_num = int(fold_dir.name.split('_')[1])
        metrics_file = fold_dir / 'metrics.csv'
        
        val_acc, val_loss = extract_final_metrics(metrics_file)
        if val_acc is not None:
            results[fold_num] = {
                'accuracy': val_acc,
                'loss': val_loss,
                'dice': val_acc  # Note: In this implementation, val_accu is actually the Dice score
            }
    
    return results


def main():
    runs_dir = Path('/home/gaetano/utooth/outputs/runs')
    
    # Analyze each run
    all_runs = {}
    for run_dir in sorted(runs_dir.glob('utooth_*')):
        if run_dir.is_dir():
            results = analyze_run(run_dir)
            if results:
                all_runs[run_dir.name] = results
    
    # Print detailed results
    for run_name, fold_results in all_runs.items():
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"{'='*60}")
        
        if not fold_results:
            print("No results found")
            continue
        
        # Collect metrics for statistics
        dice_scores = [metrics['dice'] for metrics in fold_results.values()]
        losses = [metrics['loss'] for metrics in fold_results.values()]
        
        # Print individual fold results
        print("\nIndividual Fold Results:")
        print("-" * 40)
        print(f"{'Fold':<6} {'Dice Score':<12} {'Val Loss':<12}")
        print("-" * 40)
        
        for fold_num in sorted(fold_results.keys()):
            metrics = fold_results[fold_num]
            print(f"{fold_num:<6} {metrics['dice']:.6f}     {metrics['loss']:.6f}")
        
        # Calculate statistics
        if dice_scores:
            best_fold = max(fold_results.items(), key=lambda x: x[1]['dice'])
            worst_fold = min(fold_results.items(), key=lambda x: x[1]['dice'])
            
            print("\n" + "-" * 40)
            print("Statistics:")
            print("-" * 40)
            print(f"Number of completed folds: {len(fold_results)}")
            print(f"\nBest Fold:  Fold {best_fold[0]} - Dice: {best_fold[1]['dice']:.6f}")
            print(f"Worst Fold: Fold {worst_fold[0]} - Dice: {worst_fold[1]['dice']:.6f}")
            print(f"\nAverage Dice Score: {np.mean(dice_scores):.6f} ± {np.std(dice_scores):.6f}")
            print(f"Average Val Loss:   {np.mean(losses):.6f} ± {np.std(losses):.6f}")
            
            print(f"\nDice Score Range: {min(dice_scores):.6f} - {max(dice_scores):.6f}")


if __name__ == "__main__":
    main()