#!/usr/bin/env python3
"""
Test script to understand and fix the accuracy calculation issue
"""

import os
import sys
import torch
import numpy as np
from torchmetrics.functional import jaccard_index

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from pathlib import Path
from sklearn.model_selection import KFold

def test_accuracy_calculations():
    """Test different accuracy calculation methods"""
    
    # Setup data
    data_path = '/home/gaetano/utooth/DATA/'
    data_dirs = sorted([d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
    n_samples = len(data_dirs)
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=10, shuffle=True, random_state=2025)
    fold_splits = list(kfold.split(indices))
    _, val_indices = fold_splits[1]  # Use fold 1
    
    # Create data module
    dataset = CTScanDataModuleKFold(
        data_dir=data_path,
        batch_size=2,
        train_indices=[],
        val_indices=val_indices
    )
    dataset.setup()
    val_loader = dataset.val_dataloader()
    
    # Get a batch
    x, y = next(iter(val_loader))
    
    print(f"Input shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Label unique values: {torch.unique(y)}")
    
    # Create fake predictions (similar to real model output)
    batch_size = x.shape[0]
    logits = torch.randn(batch_size, 4, 75, 75, 75)  # Typical model output
    pred = torch.sigmoid(logits)
    
    print(f"\nPrediction shape: {pred.shape}")
    
    # Test current accuracy calculation
    print("\n=== Current Accuracy Calculation ===")
    try:
        # Current method in the code
        accu1 = jaccard_index(pred, y.squeeze(1), task='multiclass', num_classes=4, threshold=0.5)
        print(f"Current method (y.squeeze(1)): {accu1:.4f}")
    except Exception as e:
        print(f"Current method failed: {e}")
    
    # Test alternative calculations
    print("\n=== Alternative Accuracy Calculations ===")
    
    # Method 1: Convert predictions to class indices
    pred_binary = (pred > 0.5).float()
    pred_classes = torch.argmax(pred_binary, dim=1)  # Shape: (B, D, H, W)
    
    # Method 2: Convert labels to class indices
    if y.dim() == 5 and y.shape[1] == 1:
        y_squeezed = y.squeeze(1)  # Remove channel dim
        if y_squeezed.shape[1] == 4:  # If shape is (B, 4, D, H, W)
            # Labels are already one-hot encoded
            y_classes = torch.argmax(y_squeezed, dim=1)
            
            try:
                accu2 = jaccard_index(pred_classes, y_classes, task='multiclass', num_classes=4)
                print(f"Method with argmax: {accu2:.4f}")
            except Exception as e:
                print(f"Argmax method failed: {e}")
    
    # Method 3: Calculate IoU manually per class and average
    ious = []
    for b in range(batch_size):
        batch_ious = []
        for c in range(4):
            pred_c = pred_binary[b, c]
            true_c = y_squeezed[b, c] if 'y_squeezed' in locals() else y[b, 0, c]
            
            intersection = torch.logical_and(pred_c > 0, true_c > 0).sum()
            union = torch.logical_or(pred_c > 0, true_c > 0).sum()
            
            if union > 0:
                iou = intersection.float() / union.float()
                batch_ious.append(iou)
        
        if batch_ious:
            ious.append(torch.mean(torch.stack(batch_ious)))
    
    if ious:
        manual_iou = torch.mean(torch.stack(ious))
        print(f"Manual per-class IoU (averaged): {manual_iou:.4f}")
    
    # Method 4: Binary segmentation accuracy (any tooth vs background)
    pred_any = torch.max(pred_binary, dim=1)[0]  # Any class predicted
    true_any = torch.max(y_squeezed if 'y_squeezed' in locals() else y.squeeze(1), dim=1)[0]  # Any class true
    
    intersection_binary = torch.logical_and(pred_any > 0, true_any > 0).sum()
    union_binary = torch.logical_or(pred_any > 0, true_any > 0).sum()
    binary_iou = intersection_binary.float() / (union_binary.float() + 1e-7)
    print(f"Binary IoU (any tooth vs background): {binary_iou:.4f}")
    
    # Method 5: Pixel accuracy
    if 'y_classes' in locals() and 'pred_classes' in locals():
        correct = (pred_classes == y_classes).sum()
        total = pred_classes.numel()
        pixel_acc = correct.float() / total
        print(f"Pixel accuracy: {pixel_acc:.4f}")


if __name__ == "__main__":
    test_accuracy_calculations()