"""
Corrected accuracy metrics for uTooth model
"""

import torch
from typing import Tuple


def calculate_multiclass_iou(pred: torch.Tensor, target: torch.Tensor, 
                           num_classes: int = 4, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate IoU for multi-class segmentation with proper handling of the label format.
    
    Args:
        pred: Predictions with shape (B, C, D, H, W) - should be after sigmoid
        target: Ground truth with shape (B, 1, C, D, H, W) where C is number of classes
        num_classes: Number of classes
        threshold: Threshold for binarizing predictions
        
    Returns:
        Average IoU across all classes and batch
    """
    # Ensure predictions are binary
    pred_binary = (pred > threshold).float()
    
    # Handle target shape
    if target.dim() == 6 and target.shape[1] == 1:
        # Remove the extra dimension: (B, 1, C, D, H, W) -> (B, C, D, H, W)
        target = target.squeeze(1)
    
    # Ensure target is binary (it should already be, but just in case)
    target = (target > 0).float()
    
    # Calculate IoU for each class and each sample in batch
    batch_size = pred.shape[0]
    ious = []
    
    for b in range(batch_size):
        sample_ious = []
        for c in range(num_classes):
            pred_c = pred_binary[b, c]
            true_c = target[b, c]
            
            intersection = torch.logical_and(pred_c > 0, true_c > 0).sum()
            union = torch.logical_or(pred_c > 0, true_c > 0).sum()
            
            # Only calculate IoU if there are any positive pixels in either pred or target
            if union > 0:
                iou = intersection.float() / union.float()
                sample_ious.append(iou)
            # If both pred and target are empty for this class, it's a perfect match (IoU = 1)
            elif intersection == 0 and union == 0:
                sample_ious.append(torch.tensor(1.0, device=pred.device))
        
        # Average IoU across classes for this sample
        if sample_ious:
            ious.append(torch.mean(torch.stack(sample_ious)))
    
    # Average across batch
    if ious:
        return torch.mean(torch.stack(ious))
    else:
        return torch.tensor(0.0, device=pred.device)


def calculate_dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                             num_classes: int = 4, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate Dice coefficient for multi-class segmentation.
    
    Args:
        pred: Predictions with shape (B, C, D, H, W) - should be after sigmoid
        target: Ground truth with shape (B, 1, C, D, H, W) where C is number of classes
        num_classes: Number of classes
        threshold: Threshold for binarizing predictions
        
    Returns:
        Average Dice coefficient across all classes and batch
    """
    # Ensure predictions are binary
    pred_binary = (pred > threshold).float()
    
    # Handle target shape
    if target.dim() == 6 and target.shape[1] == 1:
        target = target.squeeze(1)
    
    target = (target > 0).float()
    
    # Calculate Dice for each class and each sample in batch
    batch_size = pred.shape[0]
    dice_scores = []
    
    for b in range(batch_size):
        sample_dice = []
        for c in range(num_classes):
            pred_c = pred_binary[b, c]
            true_c = target[b, c]
            
            intersection = torch.logical_and(pred_c > 0, true_c > 0).sum()
            pred_sum = pred_c.sum()
            true_sum = true_c.sum()
            
            # Dice = 2 * intersection / (pred_sum + true_sum)
            if pred_sum + true_sum > 0:
                dice = 2.0 * intersection.float() / (pred_sum + true_sum).float()
                sample_dice.append(dice)
            elif pred_sum == 0 and true_sum == 0:
                # Both empty - perfect match
                sample_dice.append(torch.tensor(1.0, device=pred.device))
        
        if sample_dice:
            dice_scores.append(torch.mean(torch.stack(sample_dice)))
    
    if dice_scores:
        return torch.mean(torch.stack(dice_scores))
    else:
        return torch.tensor(0.0, device=pred.device)


def calculate_binary_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate binary IoU (any tooth vs background).
    
    Args:
        pred: Predictions with shape (B, C, D, H, W) - should be after sigmoid
        target: Ground truth with shape (B, 1, C, D, H, W)
        threshold: Threshold for binarizing predictions
        
    Returns:
        Binary IoU score
    """
    # Ensure predictions are binary
    pred_binary = (pred > threshold).float()
    
    # Handle target shape
    if target.dim() == 6 and target.shape[1] == 1:
        target = target.squeeze(1)
    
    # Convert to binary: any class present = 1, no class = 0
    pred_any = torch.max(pred_binary, dim=1)[0]  # (B, D, H, W)
    true_any = torch.max(target, dim=1)[0]  # (B, D, H, W)
    
    intersection = torch.logical_and(pred_any > 0, true_any > 0).sum()
    union = torch.logical_or(pred_any > 0, true_any > 0).sum()
    
    if union > 0:
        return intersection.float() / union.float()
    else:
        return torch.tensor(1.0, device=pred.device)