#!/usr/bin/env python3
"""
Hyperparameter sweep using Optuna for uTooth model optimization.
Supports parallel execution across multiple GPUs.
"""

import os
import argparse
import json
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import multiprocessing
from functools import partial

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.unet import UNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from scripts.train import MetricsCallback, create_fold_indices


def objective(trial, args, gpu_id):
    """Objective function for Optuna optimization."""
    
    # Set GPU for this trial
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Suggest hyperparameters
    hp = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'loss_alpha': trial.suggest_float('loss_alpha', 0.3, 0.7, step=0.05),
        'loss_gamma': trial.suggest_float('loss_gamma', 0.75, 2.0, step=0.25),
        'batch_size': trial.suggest_categorical('batch_size', [4, 5, 6, 8]),
        'start_filters': trial.suggest_categorical('start_filters', [16, 32, 64]),
        'n_blocks': trial.suggest_categorical('n_blocks', [3, 4, 5]),
        'normalization': trial.suggest_categorical('normalization', ['batch', 'instance', 'group']),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky', 'silu']),
        'attention': trial.suggest_categorical('attention', [True, False]),
    }
    
    # For sweep, use fewer folds for faster iteration
    n_folds = args.sweep_folds if hasattr(args, 'sweep_folds') else 3
    
    # Create fold indices
    fold_splits = create_fold_indices(args.data_path, n_folds, args.random_seed)
    
    # Track metrics across folds
    fold_val_losses = []
    fold_val_accus = []
    
    # Train on each fold
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        print(f"\nTrial {trial.number} - Fold {fold_idx + 1}/{n_folds}")
        
        # Create data module
        dataset = CTScanDataModuleKFold(
            data_dir=args.data_path,
            batch_size=hp['batch_size'],
            train_indices=train_indices,
            val_indices=val_indices
        )
        
        # Initialize model with suggested hyperparameters
        model = UNet(
            in_channels=1,
            out_channels=4,
            n_blocks=hp['n_blocks'],
            start_filters=hp['start_filters'],
            activation=hp['activation'],
            normalization=hp['normalization'],
            attention=hp['attention'],
            conv_mode='same',
            dim=3,
            loss_alpha=hp['loss_alpha'],
            loss_gamma=hp['loss_gamma'],
            learning_rate=hp['learning_rate']
        )
        
        # Setup callbacks
        checkpoint_dir = os.path.join(args.sweep_dir, f'trial_{trial.number}', f'fold_{fold_idx}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoint_dir,
            filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=False
        )
        
        # Optuna pruning callback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
        
        metrics_callback = MetricsCallback()
        
        callbacks = [checkpoint, early_stopping, pruning_callback, metrics_callback]
        
        # Setup logger
        csv_logger = CSVLogger(
            save_dir=args.sweep_dir,
            name=f'trial_{trial.number}',
            version=f'fold_{fold_idx}'
        )
        
        # Initialize trainer with fewer epochs for sweep
        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=args.sweep_epochs,
            callbacks=callbacks,
            logger=csv_logger,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        
        # Train the model
        trainer.fit(model, dataset)
        
        # Get fold results
        fold_val_losses.append(checkpoint.best_model_score.item())
        fold_val_accus.append(metrics_callback.best_val_accu)
        
        # Check if trial should be pruned
        trial.report(checkpoint.best_model_score.item(), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return average validation loss across folds
    avg_val_loss = np.mean(fold_val_losses)
    avg_val_accu = np.mean(fold_val_accus)
    
    # Log additional metrics
    trial.set_user_attr('avg_val_accu', avg_val_accu)
    trial.set_user_attr('fold_val_losses', fold_val_losses)
    trial.set_user_attr('fold_val_accus', fold_val_accus)
    
    return avg_val_loss


def run_parallel_sweep(args):
    """Run parallel hyperparameter sweep across multiple GPUs."""
    
    # Create study
    study_name = f"utooth_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///{args.sweep_dir}/optuna_study.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        sampler=TPESampler(seed=args.random_seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True
    )
    
    # Add default hyperparameters as the first trial
    study.enqueue_trial({
        'learning_rate': 2e-3,
        'loss_alpha': 0.55,
        'loss_gamma': 1.0,
        'batch_size': 5,
        'start_filters': 32,
        'n_blocks': 4,
        'normalization': 'batch',
        'activation': 'relu',
        'attention': False
    })
    
    # Get available GPUs
    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    print(f"Running sweep on {n_gpus} GPUs")
    
    # Create a pool of GPU workers
    gpu_queue = multiprocessing.Queue()
    for i in range(n_gpus):
        gpu_queue.put(i)
    
    def objective_wrapper(trial):
        # Get available GPU
        gpu_id = gpu_queue.get()
        try:
            result = objective(trial, args, gpu_id)
            return result
        finally:
            # Return GPU to queue
            gpu_queue.put(gpu_id)
    
    # Run optimization
    study.optimize(
        objective_wrapper,
        n_trials=args.n_trials,
        n_jobs=n_gpus,
        show_progress_bar=True
    )
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'study_name': study_name
    }
    
    with open(os.path.join(args.sweep_dir, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*50)
    print(f"Best validation loss: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Generate sweep report
    generate_sweep_report(study, args.sweep_dir)
    
    return study


def generate_sweep_report(study, sweep_dir):
    """Generate a comprehensive sweep report."""
    report_path = os.path.join(sweep_dir, 'sweep_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Hyperparameter Sweep Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Trials**: {len(study.trials)}\n")
        f.write(f"**Completed Trials**: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
        f.write(f"**Pruned Trials**: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n\n")
        
        f.write("## Best Trial\n\n")
        f.write(f"**Trial Number**: {study.best_trial.number}\n")
        f.write(f"**Validation Loss**: {study.best_value:.4f}\n")
        f.write(f"**Validation Accuracy**: {study.best_trial.user_attrs.get('avg_val_accu', 'N/A'):.4f}\n\n")
        
        f.write("### Best Hyperparameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| --- | --- |\n")
        for key, value in study.best_params.items():
            f.write(f"| {key} | {value} |\n")
        
        f.write("\n## Top 10 Trials\n\n")
        f.write("| Trial | Val Loss | Val Accuracy | Status |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        # Sort trials by value
        sorted_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value
        )[:10]
        
        for trial in sorted_trials:
            f.write(f"| {trial.number} | {trial.value:.4f} | "
                   f"{trial.user_attrs.get('avg_val_accu', 'N/A'):.4f} | "
                   f"{trial.state.name} |\n")
        
        f.write("\n## Hyperparameter Importance\n\n")
        try:
            importance = optuna.importance.get_param_importances(study)
            f.write("| Parameter | Importance |\n")
            f.write("| --- | --- |\n")
            for key, value in importance.items():
                f.write(f"| {key} | {value:.4f} |\n")
        except:
            f.write("Could not calculate parameter importance (insufficient trials).\n")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for uTooth model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='/home/gaetano/utooth/DATA/',
                        help='Path to data directory')
    
    # Sweep arguments
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials to run')
    parser.add_argument('--n_gpus', type=int, default=3,
                        help='Number of GPUs to use')
    parser.add_argument('--sweep_epochs', type=int, default=30,
                        help='Maximum epochs per trial')
    parser.add_argument('--sweep_folds', type=int, default=3,
                        help='Number of folds to use in sweep (fewer than full training)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sweep_dir', type=str, default=None,
                        help='Directory to save sweep results')
    
    args = parser.parse_args()
    
    # Create sweep directory
    if args.sweep_dir is None:
        args.sweep_dir = os.path.join('outputs', 'sweeps', 
                                      f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.sweep_dir, exist_ok=True)
    
    # Save sweep configuration
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    with open(os.path.join(args.sweep_dir, 'sweep_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run sweep
    study = run_parallel_sweep(args)
    
    print(f"\nSweep results saved to: {args.sweep_dir}")


if __name__ == "__main__":
    main()