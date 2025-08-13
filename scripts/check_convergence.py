#!/usr/bin/env python3
"""
Check if training runs have converged by analyzing validation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

def analyze_convergence(metrics_file, window=5):
    """Analyze if a model has converged based on validation metrics"""
    df = pd.read_csv(metrics_file)
    
    # Filter for validation rows
    val_df = df[df['val_loss'].notna()].copy()
    
    if len(val_df) < window:
        return None
    
    # Get last N epochs
    last_epochs = val_df.tail(window)
    
    # Calculate statistics
    final_loss = val_df['val_loss'].iloc[-1]
    final_iou = val_df['val_iou'].iloc[-1] if 'val_iou' in val_df.columns else val_df['val_accu'].iloc[-1]
    
    # Check if still improving
    loss_improving = val_df['val_loss'].iloc[-1] < val_df['val_loss'].iloc[-window]
    iou_improving = final_iou > (val_df['val_iou'].iloc[-window] if 'val_iou' in val_df.columns else val_df['val_accu'].iloc[-window])
    
    # Calculate rate of change
    loss_change_rate = (val_df['val_loss'].iloc[-1] - val_df['val_loss'].iloc[-window]) / window
    iou_change_rate = (final_iou - (val_df['val_iou'].iloc[-window] if 'val_iou' in val_df.columns else val_df['val_accu'].iloc[-window])) / window
    
    return {
        'final_epoch': len(val_df) - 1,
        'final_loss': final_loss,
        'final_iou': final_iou,
        'loss_still_improving': loss_improving,
        'iou_still_improving': iou_improving,
        'loss_change_per_epoch': loss_change_rate,
        'iou_change_per_epoch': iou_change_rate,
        'epochs_trained': len(val_df)
    }

def main():
    run_name = "utooth_10f_v3_corrected"
    base_path = Path(f"/home/gaetano/utooth/outputs/runs/{run_name}")
    
    print(f"Analyzing convergence for: {run_name}")
    print("=" * 80)
    
    convergence_results = []
    
    for fold in range(10):
        metrics_file = base_path / "metrics" / f"fold_{fold}" / "metrics.csv"
        
        if not metrics_file.exists():
            print(f"Fold {fold}: No metrics file found")
            continue
            
        result = analyze_convergence(metrics_file)
        
        if result:
            result['fold'] = fold
            convergence_results.append(result)
            
            print(f"\nFold {fold}:")
            print(f"  Final epoch: {result['final_epoch']}")
            print(f"  Final loss: {result['final_loss']:.4f}")
            print(f"  Final IoU: {result['final_iou']:.4f}")
            print(f"  Loss still improving: {'YES' if result['loss_still_improving'] else 'NO'}")
            print(f"  IoU still improving: {'YES' if result['iou_still_improving'] else 'NO'}")
            print(f"  Loss change rate: {result['loss_change_per_epoch']:.6f}/epoch")
            print(f"  IoU change rate: {result['iou_change_per_epoch']:.6f}/epoch")
            
            if result['loss_still_improving'] or result['iou_still_improving']:
                print(f"  ⚠️  NOT CONVERGED - Model was still improving!")
    
    # Summary
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    
    not_converged = sum(1 for r in convergence_results if r['loss_still_improving'] or r['iou_still_improving'])
    print(f"Folds not converged: {not_converged}/{len(convergence_results)}")
    
    if not_converged > 0:
        print("\nRecommendation: Training should continue for more epochs.")
        print("The models were still improving when training stopped.")
    
    # Save detailed results
    output_file = base_path / "convergence_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(convergence_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()