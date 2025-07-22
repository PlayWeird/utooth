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
import json
from datetime import datetime
import pandas as pd
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.unet import UNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import src.utils.ct_utils as ct_utils


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


def train_fold(fold_idx, train_indices, val_indices, data_path, args, run_dir):
    """Train a single fold"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"{'='*50}\n")
    
    # Start timing
    fold_start_time = time.time()
    
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
    
    checkpoint_dir = os.path.join(run_dir, 'checkpoints', f'fold_{fold_idx}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='utooth-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,  # Save top 3 models
        mode='min',
        save_last=True  # Also save last checkpoint
    )
    
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    callbacks = [lr_monitor, checkpoint]
    if not args.no_early_stopping:
        callbacks.append(early_stopping)
    
    # Setup loggers
    loggers = []
    
    # CSV logger for local metrics tracking
    csv_logger = CSVLogger(
        save_dir=run_dir,
        name='metrics',
        version=f'fold_{fold_idx}'
    )
    loggers.append(csv_logger)
    
    # Weights & Biases logger
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project='utooth_kfold',
            name=f'{args.experiment_name}_fold_{fold_idx}',
            tags=[f'fold_{fold_idx}', f'{args.n_folds}_fold_cv', args.experiment_name],
            save_dir=run_dir
        )
        loggers.append(wandb_logger)
    
    # Initialize trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        strategy='auto',
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        # deterministic=True,  # Disabled due to max_pool3d issues
        default_root_dir=run_dir
    )
    
    # Train the model
    trainer.fit(model, dataset)
    
    # Calculate training time
    fold_train_time = time.time() - fold_start_time
    
    # Collect fold statistics
    fold_stats = {
        'fold_idx': fold_idx,
        'best_val_loss': checkpoint.best_model_score.item() if checkpoint.best_model_score else float('inf'),
        'best_epoch': int(checkpoint.best_k_models[checkpoint.best_model_path]) if hasattr(checkpoint, 'best_k_models') and checkpoint.best_model_path in checkpoint.best_k_models else -1,
        'final_epoch': trainer.current_epoch,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'training_time_seconds': fold_train_time,
        'early_stopped': trainer.current_epoch < args.max_epochs - 1,
        'best_model_path': checkpoint.best_model_path
    }
    
    # Save fold-specific statistics
    fold_stats_path = os.path.join(run_dir, 'fold_statistics', f'fold_{fold_idx}_stats.json')
    os.makedirs(os.path.dirname(fold_stats_path), exist_ok=True)
    with open(fold_stats_path, 'w') as f:
        json.dump(fold_stats, f, indent=2)
    
    return fold_stats


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
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment run')
    
    args = parser.parse_args()
    
    # Override epochs for test run
    if args.test_run:
        args.max_epochs = 2
        print("Running in test mode with 2 epochs")
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"utooth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create run directory
    run_dir = os.path.join('outputs', 'runs', args.experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Results will be saved to: {run_dir}")
    print(f"Configuration saved to: {config_path}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    # Create fold indices
    fold_splits = create_fold_indices(args.data_path, args.n_folds, args.random_seed)
    
    # Train each fold
    fold_results = []
    overall_start_time = time.time()
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        fold_stats = train_fold(fold_idx, train_indices, val_indices, args.data_path, args, run_dir)
        fold_results.append(fold_stats)
        print(f"\nFold {fold_idx + 1} completed:")
        print(f"  Best validation loss: {fold_stats['best_val_loss']:.4f}")
        print(f"  Training time: {fold_stats['training_time_seconds']/60:.1f} minutes")
        print(f"  Early stopped: {fold_stats['early_stopped']}")
    
    # Calculate overall statistics
    total_time = time.time() - overall_start_time
    val_losses = [f['best_val_loss'] for f in fold_results]
    
    # Print summary
    print(f"\n{'='*70}")
    print("K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*70}")
    for i, stats in enumerate(fold_results):
        print(f"Fold {i + 1}: Val Loss = {stats['best_val_loss']:.4f} | "
              f"Best Epoch = {stats['best_epoch']} | "
              f"Time = {stats['training_time_seconds']/60:.1f} min")
    print(f"{'='*70}")
    print(f"Average validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")
    
    # Save overall results
    overall_results = {
        'experiment_name': args.experiment_name,
        'config': config,
        'fold_results': fold_results,
        'summary': {
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),
            'min_val_loss': float(np.min(val_losses)),
            'max_val_loss': float(np.max(val_losses)),
            'total_training_time_hours': total_time/3600,
            'completed_at': datetime.now().isoformat()
        }
    }
    
    # Save as JSON
    results_json_path = os.path.join(run_dir, 'results_summary.json')
    with open(results_json_path, 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    # Save as CSV for easy analysis
    results_df = pd.DataFrame(fold_results)
    results_csv_path = os.path.join(run_dir, 'results_summary.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {results_json_path}")
    print(f"  CSV: {results_csv_path}")
    
    # Create a markdown report
    report_path = os.path.join(run_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# uTooth Training Report\n\n")
        f.write(f"**Experiment**: {args.experiment_name}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Data Path**: {args.data_path}\n")
        f.write(f"- **Max Epochs**: {args.max_epochs}\n")
        f.write(f"- **Batch Size**: {args.batch_size}\n")
        f.write(f"- **Number of Folds**: {args.n_folds}\n")
        f.write(f"- **Random Seed**: {args.random_seed}\n")
        f.write(f"- **Early Stopping**: {'Disabled' if args.no_early_stopping else 'Enabled (patience=10)'}\n\n")
        f.write(f"## Results Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| --- | --- |\n")
        f.write(f"| Mean Validation Loss | {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f} |\n")
        f.write(f"| Min Validation Loss | {np.min(val_losses):.4f} |\n")
        f.write(f"| Max Validation Loss | {np.max(val_losses):.4f} |\n")
        f.write(f"| Total Training Time | {total_time/3600:.2f} hours |\n\n")
        f.write(f"## Fold Details\n\n")
        f.write(f"| Fold | Val Loss | Best Epoch | Training Time | Early Stopped |\n")
        f.write(f"| --- | --- | --- | --- | --- |\n")
        for i, stats in enumerate(fold_results):
            f.write(f"| {i+1} | {stats['best_val_loss']:.4f} | {stats['best_epoch']} | "
                   f"{stats['training_time_seconds']/60:.1f} min | "
                   f"{'Yes' if stats['early_stopped'] else 'No'} |\n")
        f.write(f"\n## Model Checkpoints\n\n")
        for i, stats in enumerate(fold_results):
            f.write(f"- Fold {i+1}: `{stats['best_model_path']}`\n")
    
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()