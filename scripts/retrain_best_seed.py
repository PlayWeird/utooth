#!/usr/bin/env python3
"""
Retrain Best Seed with Extended Epochs
========================================

This script retrains a specific seed (2113) with 150 epochs and no early stopping,
distributing folds across available GPUs for parallel training.
"""

import os
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import time
import logging
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import KFold

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components  
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.unet import UNet
from scripts.train import MetricsCallback

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def create_fold_indices(data_path, n_folds=10, random_seed=2113):
    """Create k-fold cross validation indices - MUST match seed search exactly"""
    # Get all case folders
    data_dirs = sorted([d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
    n_samples = len(data_dirs)
    
    print(f"Found {n_samples} samples for {n_folds}-fold cross validation")
    
    # Create indices - CRITICAL: use same seed as in seed search
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    return list(kfold.split(indices))


def train_single_fold(args):
    """Train a single fold on a specific GPU"""
    fold_idx, train_indices, val_indices, gpu_id, seed, output_dir, data_path, max_epochs = args
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create fold-specific output directory
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx} on GPU {gpu_id}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"Output: {fold_dir}")
    print(f"{'='*60}\n")
    
    # Set seed for reproducibility - MUST match original training
    # CRITICAL: Use seed + fold_idx like in original (not just seed)
    torch.manual_seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + fold_idx)
    
    # Create data module with exact same parameters as seed search
    data_module = CTScanDataModuleKFold(
        data_dir=str(data_path),
        batch_size=6,  # Same as seed search
        train_indices=train_indices,
        val_indices=val_indices
    )
    
    # Create model with best hyperparameters from sweep (exact same as seed search)
    model = UNet(
        in_channels=1,
        out_channels=4,
        n_blocks=5,  # CRITICAL: Must be 5, not 4!
        start_filters=32,
        activation='leaky',
        normalization='instance',
        attention=False,
        conv_mode='same',
        dim=3,
        loss_alpha=0.55,
        loss_gamma=2.0,
        learning_rate=0.0023378111371697686
    )
    
    # Callbacks - NO early stopping
    checkpoint = ModelCheckpoint(
        dirpath=fold_dir / "checkpoints",
        filename=f"fold_{fold_idx}_{{epoch:03d}}_{{val_dice:.4f}}",
        monitor='val_dice',
        mode='max',
        save_top_k=3,  # Keep top 3 checkpoints
        save_last=True,
        verbose=True
    )
    
    metrics_callback = MetricsCallback()
    
    # Logger
    logger = CSVLogger(
        save_dir=fold_dir,
        name="",
        version=""
    )
    
    # Trainer - NO early stopping, fixed 150 epochs
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=[0],  # Use GPU 0 within this process
        logger=logger,
        callbacks=[checkpoint, metrics_callback],
        enable_checkpointing=True,
        deterministic='warn',  # Use 'warn' instead of True to handle max_pool3d issue
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    # Train the model
    start_time = time.time()
    trainer.fit(model, data_module)
    train_time = time.time() - start_time
    
    # Get best metrics - use metrics_callback like original
    best_val_dice = metrics_callback.best_val_dice
    best_val_loss = metrics_callback.best_val_loss
    best_epoch = metrics_callback.best_epoch
    
    # Save fold results
    fold_result = {
        'fold': fold_idx,
        'gpu_id': gpu_id,
        'val_dice': best_val_dice,
        'val_loss': best_val_loss,
        'epoch': best_epoch,
        'training_time': train_time,
        'final_epoch': trainer.current_epoch,
        'best_checkpoint': checkpoint.best_model_path
    }
    
    # Save to JSON
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(fold_result, f, indent=2)
    
    return fold_result


def main():
    parser = argparse.ArgumentParser(description='Retrain best seed with extended epochs')
    parser.add_argument('--seed', type=int, default=2113, help='Seed to retrain')
    parser.add_argument('--max_epochs', type=int, default=150, help='Maximum epochs')
    parser.add_argument('--data_path', type=str, default='/home/user/utooth/DATA/', 
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/retrain_seed_XXXX_TIMESTAMP)')
    parser.add_argument('--n_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--specific_folds', type=int, nargs='+', default=None,
                        help='Specific folds to train (default: all 10)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"outputs/retrain_seed_{args.seed}_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'seed': args.seed,
        'max_epochs': args.max_epochs,
        'data_path': args.data_path,
        'n_folds': 10,
        'timestamp': datetime.now().isoformat(),
        'early_stopping': False,
        'patience': None
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get available GPUs
    n_gpus = torch.cuda.device_count() if args.n_gpus is None else min(args.n_gpus, torch.cuda.device_count())
    print(f"Using {n_gpus} GPUs for training")
    
    # Create fold indices - CRITICAL: Must match original seed search exactly
    fold_indices = create_fold_indices(args.data_path, n_folds=10, random_seed=args.seed)
    
    # Prepare training tasks
    tasks = []
    folds_to_train = args.specific_folds if args.specific_folds else list(range(10))
    
    for i, fold_idx in enumerate(folds_to_train):
        train_indices, val_indices = fold_indices[fold_idx]
        gpu_id = i % n_gpus  # Distribute across GPUs
        
        tasks.append((
            fold_idx,
            train_indices,
            val_indices,
            gpu_id,
            args.seed,
            output_dir,
            args.data_path,
            args.max_epochs
        ))
    
    print(f"\nStarting training of {len(tasks)} folds across {n_gpus} GPUs")
    print(f"Seed: {args.seed}, Max epochs: {args.max_epochs}")
    print(f"Output directory: {output_dir}\n")
    
    # Train folds in parallel
    all_results = []
    max_workers = min(n_gpus, len(tasks))  # Don't create more workers than GPUs
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_fold = {executor.submit(train_single_fold, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_fold):
            fold_idx = future_to_fold[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"\nCompleted Fold {fold_idx}: val_dice={result['val_dice']:.4f}")
            except Exception as e:
                print(f"\nError in Fold {fold_idx}: {e}")
    
    # Calculate and save summary statistics
    if all_results:
        dice_scores = [r['val_dice'] for r in all_results]
        
        summary = {
            'seed': args.seed,
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'min_dice': np.min(dice_scores),
            'max_dice': np.max(dice_scores),
            'n_valid_folds': len(all_results),
            'fold_metrics': all_results,
            'max_epochs': args.max_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        with open(output_dir / 'final_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RETRAINING COMPLETE - Seed {args.seed}")
        print(f"{'='*60}")
        print(f"Mean Dice: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"Min Dice: {summary['min_dice']:.4f}")
        print(f"Max Dice: {summary['max_dice']:.4f}")
        print(f"Valid folds: {summary['n_valid_folds']}/10")
        
        # Compare with original if available
        original_results_path = Path(f"outputs/seed_search/seed_search_20250815_104509/seeds/seed_{args.seed}/results.json")
        if original_results_path.exists():
            with open(original_results_path) as f:
                original = json.load(f)
            print(f"\nOriginal mean Dice: {original['mean_dice']:.4f}")
            improvement = summary['mean_dice'] - original['mean_dice']
            print(f"Improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()