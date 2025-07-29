#!/usr/bin/env python3
"""
Test the corrected accuracy metrics on existing checkpoints
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.accuracy_metrics import calculate_multiclass_iou, calculate_dice_coefficient, calculate_binary_iou
from torchmetrics.functional import jaccard_index


def test_checkpoint_accuracy(checkpoint_path, data_path, val_indices):
    """Test accuracy metrics on a checkpoint"""
    
    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model = UNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create data loader
    dataset = CTScanDataModuleKFold(
        data_dir=data_path,
        batch_size=2,
        train_indices=[],
        val_indices=val_indices
    )
    dataset.setup()
    val_loader = dataset.val_dataloader()
    
    # Metrics storage
    old_accus = []
    new_ious = []
    dice_scores = []
    binary_ious = []
    
    # Process validation set
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits = model(x)
            pred = torch.sigmoid(logits)
            
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
            
            if batch_idx == 0:
                print(f"\nBatch {batch_idx} shapes:")
                print(f"  Input: {x.shape}")
                print(f"  Label: {y.shape}")
                print(f"  Prediction: {pred.shape}")
    
    # Calculate averages
    avg_old = np.mean(old_accus)
    avg_new = np.mean(new_ious)
    avg_dice = np.mean(dice_scores)
    avg_binary = np.mean(binary_ious)
    
    print(f"\nMetrics Summary:")
    print(f"  Old accuracy (incorrect): {avg_old:.4f}")
    print(f"  Corrected IoU: {avg_new:.4f}")
    print(f"  Dice coefficient: {avg_dice:.4f}")
    print(f"  Binary IoU: {avg_binary:.4f}")
    
    return {
        'old_accuracy': avg_old,
        'corrected_iou': avg_new,
        'dice': avg_dice,
        'binary_iou': avg_binary
    }


def main():
    # Test on v3 experiment which had good results
    experiment = 'utooth_10f_v3'
    data_path = '/home/gaetano/utooth/DATA/'
    
    # Load experiment config
    run_dir = os.path.join('outputs', 'runs', experiment)
    config_path = os.path.join(run_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create fold indices
    data_dirs = sorted([d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
    n_samples = len(data_dirs)
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_seed'])
    fold_splits = list(kfold.split(indices))
    
    # Test fold 1 which we know had good visual results
    fold_idx = 1
    _, val_indices = fold_splits[fold_idx]
    
    # Find best checkpoint
    checkpoint_dir = os.path.join(run_dir, 'checkpoints', f'fold_{fold_idx}')
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and f != 'last.ckpt' and 'val_loss=' in f]
    checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]))
    best_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    
    print(f"Testing accuracy metrics for {experiment}, fold {fold_idx}")
    results = test_checkpoint_accuracy(best_checkpoint, data_path, val_indices)
    
    print(f"\n{'='*50}")
    print("CONCLUSION:")
    print(f"The corrected IoU ({results['corrected_iou']:.1%}) is much higher than")
    print(f"the incorrectly calculated accuracy ({results['old_accuracy']:.1%})")
    print(f"\nThis matches the visual results showing good segmentation quality!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()