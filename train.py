#!/usr/bin/env python3
"""
uTooth Training Script with K-Fold Cross Validation
Based on unet_trainer.ipynb
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import sigmoid, where, cuda
from sklearn.model_selection import KFold

from volume_dataloader_kfold import CTScanDataModuleKFold
from unet import UNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import ct_utils


def create_fold_indices(data_path, n_folds=5, random_seed=42):
    """Create k-fold cross validation indices"""
    # Get all case folders
    data_dirs = sorted([d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('case-')])
    n_samples = len(data_dirs)
    
    print(f"Found {n_samples} samples for {n_folds}-fold cross validation")
    
    # Create indices
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    return list(kfold.split(indices))


def train_fold(fold_idx, train_indices, val_indices, data_path, args):
    """Train a single fold"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"{'='*50}\n")
    
    # Create data module with fold indices
    dataset = CTScanDataModuleKFold(
        data_dir=data_path,
        batch_size=args.batch_size,
        train_indices=train_indices,
        val_indices=val_indices
    )
    
    # Initialize model with same hyperparameters as notebook
    model = UNet(
        in_channels=1,
        out_channels=4,
        n_blocks=4,
        start_filters=32,
        activation='relu',
        normalization='batch',
        conv_mode='same',
        dim=3,
        attention=False,
        loss_alpha=0.5236,  # From notebook
        loss_gamma=1.0,     # From notebook
        learning_rate=2e-3  # From notebook
    )
    
    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/fold_{fold_idx}',
        filename='utooth-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )
    
    # Setup logger
    wandb_logger = WandbLogger(
        project='utooth_kfold',
        name=f'fold_{fold_idx}',
        tags=[f'fold_{fold_idx}', f'{args.n_folds}_fold_cv']
    ) if args.use_wandb else None
    
    # Initialize trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        strategy='auto',
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint],
        logger=wandb_logger,
        enable_progress_bar=True
    )
    
    # Train the model
    trainer.fit(model, dataset)
    
    # Return best validation loss
    return checkpoint.best_model_score.item() if checkpoint.best_model_score else float('inf')


def main():
    parser = argparse.ArgumentParser(description='uTooth Training with K-Fold Cross Validation')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='/home/gaetano/utooth/DATA/',
                        help='Path to data directory')
    
    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size (default: 5)')
    
    # K-fold arguments
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross validation (default: 5)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Other arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--test_run', action='store_true',
                        help='Run with only 2 epochs for testing')
    
    args = parser.parse_args()
    
    # Override epochs for test run
    if args.test_run:
        args.max_epochs = 2
        print("Running in test mode with 2 epochs")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create fold indices
    fold_splits = create_fold_indices(args.data_path, args.n_folds, args.random_seed)
    
    # Train each fold
    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        val_loss = train_fold(fold_idx, train_indices, val_indices, args.data_path, args)
        fold_results.append(val_loss)
        print(f"\nFold {fold_idx + 1} - Best validation loss: {val_loss:.4f}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*50}")
    for i, loss in enumerate(fold_results):
        print(f"Fold {i + 1}: {loss:.4f}")
    print(f"{'='*50}")
    print(f"Average validation loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()