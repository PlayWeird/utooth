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

# Fix matplotlib backend issue that causes "NO POLYGONS TO PRINT" warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.unet import UNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import src.utils.ct_utils as ct_utils


class MetricsCallback(Callback):
    """Custom callback to track best validation metrics"""
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_val_accu = 0.0
        self.best_val_dice = 0.0
        self.best_epoch = 0
        
    def on_validation_end(self, trainer, pl_module):
        # Get current validation metrics
        val_loss = trainer.callback_metrics.get('val_loss')
        val_accu = trainer.callback_metrics.get('val_accu')
        val_dice = trainer.callback_metrics.get('val_dice')
        
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.best_epoch = trainer.current_epoch
            
        if val_accu is not None and val_accu > self.best_val_accu:
            self.best_val_accu = val_accu.item()
            
        if val_dice is not None and val_dice > self.best_val_dice:
            self.best_val_dice = val_dice.item()
            self.best_epoch = trainer.current_epoch


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


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for last.ckpt first (most recent)
    last_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')
    if os.path.exists(last_ckpt):
        return last_ckpt
    
    # Look for best checkpoint files
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and f != 'last.ckpt']
    if not ckpt_files:
        return None
    
    # Sort by modification time (newest first)
    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return os.path.join(checkpoint_dir, ckpt_files[0])

def train_fold(fold_idx, train_indices, val_indices, data_path, args, run_dir):
    """Train a single fold"""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    print(f"{'='*50}\n")
    
    # Check for existing checkpoint to resume from
    checkpoint_dir = os.path.join(run_dir, 'checkpoints', f'fold_{fold_idx}')
    resume_from_checkpoint = None
    
    if args.resume and os.path.exists(checkpoint_dir):
        resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if resume_from_checkpoint:
            print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            print("⚠️  No checkpoint found to resume from, starting fresh")
    elif os.path.exists(checkpoint_dir) and not args.force_restart:
        # Check if this fold was already completed
        fold_stats_path = os.path.join(run_dir, 'fold_statistics', f'fold_{fold_idx}_stats.json')
        if os.path.exists(fold_stats_path):
            print(f"✅ Fold {fold_idx + 1} already completed, skipping...")
            # Load and return existing stats
            with open(fold_stats_path, 'r') as f:
                return json.load(f)
        
        # Offer to resume if checkpoint exists
        resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if resume_from_checkpoint and not args.no_resume:
            print(f"📁 Found existing checkpoint: {resume_from_checkpoint}")
            if not args.auto_resume:
                response = input("Resume from this checkpoint? [Y/n]: ").strip().lower()
                if response in ['', 'y', 'yes']:
                    resume_from_checkpoint = resume_from_checkpoint
                else:
                    resume_from_checkpoint = None
                    print("🔄 Starting fresh (existing checkpoints will be overwritten)")
            else:
                print("🔄 Auto-resuming from checkpoint")
    
    # Start timing
    fold_start_time = time.time()
    
    # Create data module with fold indices
    dataset = CTScanDataModuleKFold(
        data_dir=data_path,
        batch_size=args.batch_size,
        train_indices=train_indices,
        val_indices=val_indices
    )
    
    # Initialize model with optimal hyperparameters from cross-validation
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
        loss_alpha=0.55,    # Optimal from cross-validation
        loss_gamma=1.0,     # Default
        learning_rate=2e-3  # Original working value
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
    
    # Add custom metrics callback
    metrics_callback = MetricsCallback()
    
    callbacks = [lr_monitor, checkpoint, metrics_callback]
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
    
    # Train the model (with optional resume)
    trainer.fit(model, dataset, ckpt_path=resume_from_checkpoint)
    
    # Calculate training time
    fold_train_time = time.time() - fold_start_time
    
    # Collect fold statistics
    # Extract best epoch from checkpoint filename
    best_epoch = -1
    if checkpoint.best_model_path:
        import re
        # Extract epoch number from filename like "utooth-epoch=43-val_loss=0.2227.ckpt"
        epoch_match = re.search(r'epoch=(\d+)', checkpoint.best_model_path)
        if epoch_match:
            best_epoch = int(epoch_match.group(1))
    
    fold_stats = {
        'fold_idx': fold_idx,
        'best_val_loss': checkpoint.best_model_score.item() if checkpoint.best_model_score else float('inf'),
        'best_val_accu': metrics_callback.best_val_accu,
        'best_epoch': best_epoch,
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
    
    return fold_stats, metrics_callback


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
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Automatically resume without asking for confirmation')
    parser.add_argument('--force_restart', action='store_true',
                        help='Force restart training even if checkpoints exist')
    parser.add_argument('--no_resume', action='store_true',
                        help='Never attempt to resume, always start fresh')
    
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
    
    # Check for existing experiment
    config_path = os.path.join(run_dir, 'config.json')
    experiment_state_path = os.path.join(run_dir, 'experiment_state.json')
    
    existing_config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
        
        if not args.force_restart and not args.resume:
            print(f"\n⚠️  Experiment '{args.experiment_name}' already exists!")
            print(f"Started: {existing_config.get('start_time', 'Unknown')}")
            if not args.auto_resume:
                print("Options:")
                print("  --resume: Resume from latest checkpoints")
                print("  --force_restart: Start completely fresh")
                response = input("Resume existing experiment? [Y/n]: ").strip().lower()
                if response in ['', 'y', 'yes']:
                    args.resume = True
                else:
                    args.force_restart = True
            else:
                args.resume = True
    
    # Initialize or load experiment state
    experiment_state = {
        'experiment_name': args.experiment_name,
        'total_folds': args.n_folds,
        'completed_folds': [],
        'failed_folds': [],
        'current_fold': None,
        'start_time': existing_config['start_time'] if existing_config else datetime.now().isoformat(),
        'last_update': datetime.now().isoformat(),
        'status': 'running',
        'resume_count': 0
    }
    
    if os.path.exists(experiment_state_path) and not args.force_restart:
        with open(experiment_state_path, 'r') as f:
            experiment_state.update(json.load(f))
        experiment_state['resume_count'] += 1
        experiment_state['last_update'] = datetime.now().isoformat()
        print(f"🔄 Resuming experiment (resume #{experiment_state['resume_count']})")
    
    # Save configuration (preserve original start_time if resuming)
    config = vars(args)
    config['start_time'] = experiment_state['start_time']
    config['resume_count'] = experiment_state['resume_count']
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save experiment state
    with open(experiment_state_path, 'w') as f:
        json.dump(experiment_state, f, indent=2)
    
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
    
    # Skip already completed folds if resuming
    start_fold = 0
    if args.resume and experiment_state['completed_folds']:
        completed_fold_indices = [f['fold_idx'] for f in experiment_state['completed_folds']]
        start_fold = max(completed_fold_indices) + 1 if completed_fold_indices else 0
        fold_results = experiment_state['completed_folds']
        print(f"📁 Skipping {len(completed_fold_indices)} already completed folds")
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        if fold_idx < start_fold:
            continue
            
        # Update experiment state
        experiment_state['current_fold'] = fold_idx
        experiment_state['last_update'] = datetime.now().isoformat()
        with open(experiment_state_path, 'w') as f:
            json.dump(experiment_state, f, indent=2)
        
        try:
            fold_stats, metrics_callback = train_fold(fold_idx, train_indices, val_indices, args.data_path, args, run_dir)
            fold_results.append(fold_stats)
            
            # Update experiment state with completed fold
            experiment_state['completed_folds'].append(fold_stats)
            experiment_state['current_fold'] = None
            if fold_idx in experiment_state.get('failed_folds', []):
                experiment_state['failed_folds'].remove(fold_idx)
            
            print(f"\nFold {fold_idx + 1} completed:")
            print(f"  Best validation loss: {fold_stats['best_val_loss']:.4f}")
            print(f"  Best validation accuracy: {fold_stats['best_val_accu']:.4f}")
            print(f"  Training time: {fold_stats['training_time_seconds']/60:.1f} minutes")
            print(f"  Early stopped: {fold_stats['early_stopped']}")
            
        except Exception as e:
            print(f"\n❌ Fold {fold_idx + 1} failed with error: {str(e)}")
            experiment_state['failed_folds'].append(fold_idx)
            experiment_state['current_fold'] = None
            raise e
        finally:
            # Always update experiment state
            experiment_state['last_update'] = datetime.now().isoformat()
            with open(experiment_state_path, 'w') as f:
                json.dump(experiment_state, f, indent=2)
    
    # Calculate overall statistics
    total_time = time.time() - overall_start_time
    val_losses = [f['best_val_loss'] for f in fold_results]
    val_accus = [f['best_val_accu'] for f in fold_results]
    
    # Print summary
    print(f"\n{'='*70}")
    print("K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*70}")
    for i, stats in enumerate(fold_results):
        print(f"Fold {i + 1}: Val Loss = {stats['best_val_loss']:.4f} | "
              f"Val Accu = {stats['best_val_accu']:.4f} | "
              f"Best Epoch = {stats['best_epoch']} | "
              f"Time = {stats['training_time_seconds']/60:.1f} min")
    print(f"{'='*70}")
    print(f"Average validation loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Average validation accuracy: {np.mean(val_accus):.4f} ± {np.std(val_accus):.4f}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")
    
    # Mark experiment as completed
    experiment_state['status'] = 'completed'
    experiment_state['completed_at'] = datetime.now().isoformat()
    experiment_state['current_fold'] = None
    with open(experiment_state_path, 'w') as f:
        json.dump(experiment_state, f, indent=2)
    
    # Save overall results
    overall_results = {
        'experiment_name': args.experiment_name,
        'config': config,
        'fold_results': fold_results,
        'experiment_state': experiment_state,
        'summary': {
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),
            'min_val_loss': float(np.min(val_losses)),
            'max_val_loss': float(np.max(val_losses)),
            'mean_val_accu': float(np.mean(val_accus)),
            'std_val_accu': float(np.std(val_accus)),
            'min_val_accu': float(np.min(val_accus)),
            'max_val_accu': float(np.max(val_accus)),
            'total_training_time_hours': total_time/3600,
            'completed_at': datetime.now().isoformat(),
            'resume_count': experiment_state['resume_count']
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
        f.write(f"| Mean Validation Accuracy | {np.mean(val_accus):.4f} ± {np.std(val_accus):.4f} |\n")
        f.write(f"| Min Validation Accuracy | {np.min(val_accus):.4f} |\n")
        f.write(f"| Max Validation Accuracy | {np.max(val_accus):.4f} |\n")
        f.write(f"| Total Training Time | {total_time/3600:.2f} hours |\n\n")
        f.write(f"## Fold Details\n\n")
        f.write(f"| Fold | Val Loss | Val Accuracy | Best Epoch | Training Time | Early Stopped |\n")
        f.write(f"| --- | --- | --- | --- | --- | --- |\n")
        for i, stats in enumerate(fold_results):
            f.write(f"| {i+1} | {stats['best_val_loss']:.4f} | {stats['best_val_accu']:.4f} | {stats['best_epoch']} | "
                   f"{stats['training_time_seconds']/60:.1f} min | "
                   f"{'Yes' if stats['early_stopped'] else 'No'} |\n")
        f.write(f"\n## Model Checkpoints\n\n")
        for i, stats in enumerate(fold_results):
            f.write(f"- Fold {i+1}: `{stats['best_model_path']}`\n")
    
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()