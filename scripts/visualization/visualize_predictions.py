#!/usr/bin/env python3
"""
Visualize model predictions on validation data to verify segmentation quality
"""

import os
import sys
import argparse
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
import src.utils.ct_utils as ct_utils


def load_model_from_checkpoint(checkpoint_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    model = UNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    return model


def visualize_predictions(model, data_path, val_indices, output_dir, num_samples=3):
    """Generate visualizations comparing ground truth with predictions"""
    
    # Create data module with validation indices
    dataset = CTScanDataModuleKFold(
        data_dir=data_path,
        batch_size=1,  # Process one at a time for visualization
        train_indices=[],  # Empty train set
        val_indices=val_indices
    )
    dataset.setup()
    val_loader = dataset.val_dataloader()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process first few samples
    for idx, (x, y) in enumerate(val_loader):
        if idx >= num_samples:
            break
            
        x = x.to(device)
        y = y.to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(x)
            pred = torch.sigmoid(logits)
            pred_binary = (pred > 0.5).float()
        
        # Convert to numpy for visualization
        x_np = x[0, 0].cpu().numpy()  # Shape: (D, H, W)
        
        # Handle the label format: y is (B, 1, C, D, H, W)
        # Remove batch and extra dimension to get (C, D, H, W)
        y_np = y[0, 0].cpu().numpy()  # Shape: (C, D, H, W) where C=4
        
        # Sum across all tooth classes for visualization
        y_sum = y_np.sum(axis=0)  # Shape: (D, H, W)
        
        pred_np = pred_binary[0].cpu().numpy()  # Shape: (C, D, H, W)
        pred_sum = pred_np.sum(axis=0)  # Shape: (D, H, W)
        
        # Create figure with multiple views
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Select slices to show (beginning, middle, end)
        slice_indices = [
            x_np.shape[0] // 4,
            x_np.shape[0] // 2,
            3 * x_np.shape[0] // 4
        ]
        
        for i, slice_idx in enumerate(slice_indices):
            # Original CT
            axes[i, 0].imshow(x_np[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'CT Slice {slice_idx}')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(y_sum[slice_idx], cmap='hot')
            axes[i, 1].set_title(f'Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(pred_sum[slice_idx], cmap='hot')
            axes[i, 2].set_title(f'Prediction')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Sample {idx + 1} - Axial Views', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_axial.png'), dpi=150)
        plt.close()
        
        # Create 3D visualization using maximum intensity projection
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Sagittal view (max projection)
        axes[0, 0].imshow(np.max(x_np, axis=2).T, cmap='gray', origin='lower')
        axes[0, 0].set_title('CT - Sagittal MIP')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.max(y_sum, axis=2).T, cmap='hot', origin='lower')
        axes[0, 1].set_title('Ground Truth - Sagittal MIP')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.max(pred_sum, axis=2).T, cmap='hot', origin='lower')
        axes[0, 2].set_title('Prediction - Sagittal MIP')
        axes[0, 2].axis('off')
        
        # Coronal view (max projection)
        axes[1, 0].imshow(np.max(x_np, axis=1).T, cmap='gray', origin='lower')
        axes[1, 0].set_title('CT - Coronal MIP')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.max(y_sum, axis=1).T, cmap='hot', origin='lower')
        axes[1, 1].set_title('Ground Truth - Coronal MIP')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(np.max(pred_sum, axis=1).T, cmap='hot', origin='lower')
        axes[1, 2].set_title('Prediction - Coronal MIP')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Sample {idx + 1} - Maximum Intensity Projections', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_mip.png'), dpi=150)
        plt.close()
        
        # Calculate and print metrics
        intersection = np.logical_and(y_sum > 0, pred_sum > 0).sum()
        union = np.logical_or(y_sum > 0, pred_sum > 0).sum()
        iou = intersection / (union + 1e-8)
        
        print(f"\nSample {idx + 1} metrics:")
        print(f"  IoU (Jaccard): {iou:.4f}")
        print(f"  Predicted voxels: {(pred_sum > 0).sum()}")
        print(f"  Ground truth voxels: {(y_sum > 0).sum()}")
        print(f"  Intersection voxels: {intersection}")
        
        # Also calculate per-class IoU
        print(f"  Per-class IoU:")
        for class_idx in range(4):
            class_intersection = np.logical_and(y_np[class_idx] > 0, pred_np[class_idx] > 0).sum()
            class_union = np.logical_or(y_np[class_idx] > 0, pred_np[class_idx] > 0).sum()
            class_iou = class_intersection / (class_union + 1e-8) if class_union > 0 else 0
            print(f"    Class {class_idx}: {class_iou:.4f}")
        
        # Save individual class predictions and ground truth comparison
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        middle_slice = x_np.shape[0] // 2
        
        for class_idx in range(4):
            # Ground truth for this class
            axes[0, class_idx].imshow(y_np[class_idx, middle_slice], cmap='hot')
            axes[0, class_idx].set_title(f'Class {class_idx} GT')
            axes[0, class_idx].axis('off')
            
            # Prediction for this class
            axes[1, class_idx].imshow(pred_np[class_idx, middle_slice], cmap='hot')
            axes[1, class_idx].set_title(f'Class {class_idx} Pred')
            axes[1, class_idx].axis('off')
        
        plt.suptitle(f'Sample {idx + 1} - Individual Class Predictions (Slice {middle_slice})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}_classes.png'), dpi=150)
        plt.close()
        
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize uTooth model predictions')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., utooth_10f_v3)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index to visualize (default: 0)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize (default: 3)')
    parser.add_argument('--data_path', type=str, default='/home/gaetano/utooth/DATA/',
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    # Load experiment configuration
    run_dir = os.path.join('outputs', 'runs', args.experiment)
    config_path = os.path.join(run_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"Error: Experiment '{args.experiment}' not found")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find best checkpoint for the fold
    checkpoint_dir = os.path.join(run_dir, 'checkpoints', f'fold_{args.fold}')
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and 'val_loss=' in f]
    
    if not checkpoints:
        print(f"Error: No checkpoints found for fold {args.fold}")
        return
    
    # Sort by validation loss (lower is better)
    checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]))
    best_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    
    print(f"Using checkpoint: {best_checkpoint}")
    
    # Load fold indices
    from sklearn.model_selection import KFold
    data_dirs = sorted([d for d in Path(args.data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
    n_samples = len(data_dirs)
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_seed'])
    fold_splits = list(kfold.split(indices))
    _, val_indices = fold_splits[args.fold]
    
    # Load model
    model = load_model_from_checkpoint(best_checkpoint)
    
    # Create output directory
    output_dir = os.path.join(run_dir, 'visualizations', f'fold_{args.fold}')
    
    # Generate visualizations
    visualize_predictions(model, args.data_path, val_indices, output_dir, args.num_samples)


if __name__ == "__main__":
    main()