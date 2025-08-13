#!/usr/bin/env python3
"""
Organized Hyperparameter Sweep using Optuna for uTooth Model Optimization
==========================================================================

This script provides a comprehensive hyperparameter sweep system with:
- YAML configuration management
- Parallel GPU execution
- Organized directory structure  
- Comprehensive reporting and monitoring

Directory Structure:
  outputs/sweeps/
    └── {sweep_name}_{timestamp}/
        ├── trials/           # Individual trial data
        ├── checkpoints/      # Model checkpoints (if enabled)
        ├── logs/            # Logging files
        ├── plots/           # Visualization plots
        ├── reports/         # Generated reports
        └── sweep_config.yaml # Configuration used

Usage:
  python scripts/sweep_runner.py [options]
  python scripts/sweep_runner.py --config sweep/configs/default_sweep_config.yaml --n_trials 50
"""

import os
import argparse
import json
import multiprocessing
import sys
from pathlib import Path
from datetime import datetime
import time
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sweep utilities
from scripts.sweep.utils.config_loader import load_sweep_config, create_sweep_directory, validate_config
from scripts.sweep.utils.optuna_helpers import (
    create_optuna_study, suggest_hyperparameters, enqueue_baseline_trial,
    calculate_trial_metrics, log_trial_results, get_study_statistics, setup_logging
)

# Import model components  
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.unet import UNet
from scripts.train import MetricsCallback, create_fold_indices

# Import ML frameworks
import torch
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger


class TrialExecutor:
    """Handles execution of individual trials."""
    
    def __init__(self, config, sweep_dir, logger):
        self.config = config
        self.sweep_dir = sweep_dir
        self.logger = logger
        self.fold_splits = None
        
    def initialize(self):
        """Initialize fold splits and other trial-independent setup."""
        self.fold_splits = create_fold_indices(
            self.config.data_path, 
            self.config.k_folds, 
            self.config.seed
        )
        self.logger.info(f"Created {len(self.fold_splits)} fold splits")
        
    def execute_trial(self, trial, gpu_id):
        """Execute a single trial on specified GPU."""
        
        # Set GPU for this trial
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Suggest hyperparameters
        suggested_params = suggest_hyperparameters(trial, self.config.hyperparameters)
        
        self.logger.info(f"Trial {trial.number} starting on GPU {gpu_id}")
        self.logger.info(f"Parameters: {suggested_params}")
        
        # Train on each fold
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.fold_splits):
            self.logger.info(f"Trial {trial.number} - Fold {fold_idx + 1}/{len(self.fold_splits)}")
            
            try:
                fold_result = self._train_fold(
                    trial, suggested_params, fold_idx, 
                    train_indices, val_indices, gpu_id
                )
                fold_results.append(fold_result)
                
                # Report intermediate result for pruning
                trial.report(fold_result['val_loss'], fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            except Exception as e:
                self.logger.error(f"Trial {trial.number} fold {fold_idx} failed: {str(e)}")
                raise e
        
        # Calculate aggregated metrics
        metrics = calculate_trial_metrics(fold_results)
        
        # Log results
        log_trial_results(trial, metrics, suggested_params)
        
        # Save trial details
        self._save_trial_details(trial, suggested_params, metrics, fold_results)
        
        return metrics['mean_val_loss']
    
    def _train_fold(self, trial, params, fold_idx, train_indices, val_indices, gpu_id):
        """Train a single fold with given parameters."""
        
        # Create data module
        dataset = CTScanDataModuleKFold(
            data_dir=self.config.data_path,
            batch_size=params['batch_size'],
            train_indices=train_indices,
            val_indices=val_indices
        )
        
        # Initialize model with suggested hyperparameters
        model = UNet(
            in_channels=1,
            out_channels=4,
            n_blocks=params['n_blocks'],
            start_filters=params['start_filters'],
            activation=params['activation'],
            normalization=params['normalization'],
            attention=params['attention'],
            conv_mode='same',
            dim=3,
            loss_alpha=params['loss_alpha'],
            loss_gamma=params['loss_gamma'],
            learning_rate=params['learning_rate']
        )
        
        # Setup callbacks
        trial_dir = self.sweep_dir / "trials" / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = []
        
        # Model checkpoint (only if configured)
        if self.config.save_checkpoints:
            checkpoint_dir = self.sweep_dir / "checkpoints" / f"trial_{trial.number}" / f"fold_{fold_idx}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                monitor='val_loss',
                dirpath=checkpoint_dir,
                filename='best',
                save_top_k=1,
                mode='min'
            )
            callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            mode='min',
            verbose=False
        )
        callbacks.append(early_stopping)
        
        # Optuna pruning callback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
        callbacks.append(pruning_callback)
        
        # Metrics callback
        metrics_callback = MetricsCallback()
        callbacks.append(metrics_callback)
        
        # Setup logger
        csv_logger = CSVLogger(
            save_dir=trial_dir,
            name=f'fold_{fold_idx}',
            version=None
        )
        
        # Initialize trainer
        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=self.config.max_epochs,
            callbacks=callbacks,
            logger=csv_logger,
            enable_progress_bar=False,
            log_every_n_steps=1,
            enable_model_summary=False
        )
        
        # Train the model
        trainer.fit(model, dataset)
        
        # Extract results
        if self.config.save_checkpoints and 'checkpoint' in locals():
            best_val_loss = checkpoint.best_model_score.item()
        else:
            # Get from trainer callback metrics if no checkpoint
            best_val_loss = float('inf')
            for callback in trainer.callbacks:
                if hasattr(callback, 'best_model_score') and callback.best_model_score is not None:
                    best_val_loss = callback.best_model_score.item()
                    break
        
        return {
            'fold_idx': fold_idx,
            'val_loss': best_val_loss,
            'val_accu': metrics_callback.best_val_accu,
            'final_epoch': trainer.current_epoch,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices)
        }
    
    def _save_trial_details(self, trial, params, metrics, fold_results):
        """Save detailed trial information."""
        
        trial_details = {
            'trial_number': trial.number,
            'trial_state': trial.state.name,
            'parameters': params,
            'metrics': metrics,
            'fold_results': fold_results,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - trial.datetime_start).total_seconds() if trial.datetime_start else None
        }
        
        trial_file = self.sweep_dir / "trials" / f"trial_{trial.number}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_details, f, indent=2)


class SweepRunner:
    """Main class for running hyperparameter sweeps."""
    
    def __init__(self, config_path=None):
        # Load configuration
        self.config = load_sweep_config(config_path)
        
        # Validate configuration
        if not validate_config(self.config):
            raise ValueError("Invalid configuration")
        
        # Create sweep directory
        self.sweep_dir = create_sweep_directory(self.config)
        
        # Setup logging
        self.logger = setup_logging(self.sweep_dir, verbose=True)
        self.logger.info(f"Sweep directory created: {self.sweep_dir}")
        
        # Initialize trial executor
        self.trial_executor = TrialExecutor(self.config, self.sweep_dir, self.logger)
        self.trial_executor.initialize()
        
        # GPU management
        self.available_gpus = list(range(min(self.config.n_gpus, torch.cuda.device_count())))
        self.gpu_queue = multiprocessing.Queue()
        for gpu_id in self.available_gpus:
            self.gpu_queue.put(gpu_id)
        
        self.logger.info(f"Using GPUs: {self.available_gpus}")
    
    def run_sweep(self, n_trials=None):
        """Run the hyperparameter sweep."""
        
        if n_trials is None:
            n_trials = self.config.trials_per_gpu * len(self.available_gpus)
        
        self.logger.info(f"Starting sweep with {n_trials} trials on {len(self.available_gpus)} GPUs")
        
        # Create Optuna study
        study = create_optuna_study(self.config, self.sweep_dir)
        
        # Enqueue baseline trial
        enqueue_baseline_trial(study, self.config.baseline)
        
        # Define objective function for multiprocessing
        def objective_wrapper(trial):
            # Get available GPU
            gpu_id = self.gpu_queue.get()
            try:
                result = self.trial_executor.execute_trial(trial, gpu_id)
                return result
            except optuna.TrialPruned:
                raise
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {str(e)}")
                raise e
            finally:
                # Return GPU to queue
                self.gpu_queue.put(gpu_id)
        
        # Run optimization
        start_time = time.time()
        study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            n_jobs=len(self.available_gpus),
            show_progress_bar=True
        )
        total_time = time.time() - start_time
        
        # Generate final report
        self._generate_final_report(study, total_time)
        
        return study
    
    def _generate_final_report(self, study, total_time):
        """Generate comprehensive final report."""
        
        self.logger.info("Generating final report...")
        
        # Get study statistics
        stats = get_study_statistics(study)
        
        # Save statistics
        stats_file = self.sweep_dir / "reports" / "sweep_statistics.json"
        stats_file.parent.mkdir(exist_ok=True)
        
        stats['total_time_hours'] = total_time / 3600
        stats['trials_per_hour'] = len(study.trials) / (total_time / 3600) if total_time > 0 else 0
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate markdown report
        self._create_markdown_report(study, stats, total_time)
        
        # Print summary
        print(f"\n{'='*70}")
        print("HYPERPARAMETER SWEEP COMPLETED")
        print(f"{'='*70}")
        print(f"Total trials: {stats['total_trials']}")
        print(f"Completed trials: {stats['completed_trials']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Total time: {total_time/3600:.2f} hours")
        if stats['best_value'] is not None:
            print(f"Best validation loss: {stats['best_value']:.4f}")
            print(f"Best trial: {stats['best_trial_number']}")
        print(f"Results saved to: {self.sweep_dir}")
        print(f"{'='*70}")
    
    def _create_markdown_report(self, study, stats, total_time):
        """Create detailed markdown report."""
        
        report_file = self.sweep_dir / "reports" / "sweep_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# uTooth Hyperparameter Sweep Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Sweep Directory**: `{self.sweep_dir}`\n")
            f.write(f"**Configuration**: `{self.sweep_dir / 'sweep_config.yaml'}`\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- **Total Trials**: {stats['total_trials']}\n")
            f.write(f"- **Completed**: {stats['completed_trials']}\n")
            f.write(f"- **Pruned**: {stats['pruned_trials']}\n")
            f.write(f"- **Failed**: {stats['failed_trials']}\n")
            f.write(f"- **Success Rate**: {stats['success_rate']:.1%}\n")
            f.write(f"- **Total Time**: {total_time/3600:.2f} hours\n")
            f.write(f"- **Trials per Hour**: {stats['trials_per_hour']:.1f}\n\n")
            
            # Best trial
            if stats['best_value'] is not None:
                f.write("## Best Trial\n\n")
                f.write(f"- **Trial Number**: {stats['best_trial_number']}\n")
                f.write(f"- **Validation Loss**: {stats['best_value']:.4f}\n\n")
                
                f.write("### Best Hyperparameters\n\n")
                f.write("| Parameter | Value |\n")
                f.write("| --- | --- |\n")
                for key, value in stats['best_params'].items():
                    f.write(f"| {key} | {value} |\n")
                f.write("\n")
            
            # Parameter importance
            if stats['param_importance']:
                f.write("## Parameter Importance\n\n")
                f.write("| Parameter | Importance |\n")
                f.write("| --- | --- |\n")
                for param, importance in sorted(stats['param_importance'].items(), 
                                              key=lambda x: x[1], reverse=True):
                    f.write(f"| {param} | {importance:.4f} |\n")
                f.write("\n")
            
            # Configuration used
            f.write("## Configuration Used\n\n")
            f.write("```yaml\n")
            with open(self.sweep_dir / "sweep_config.yaml") as config_file:
                f.write(config_file.read())
            f.write("\n```\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter sweep for uTooth model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (default: use built-in config)')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of trials to run (default: trials_per_gpu * n_gpus from config)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data path from config')
    
    args = parser.parse_args()
    
    try:
        # Create sweep runner
        runner = SweepRunner(config_path=args.config)
        
        # Override data path if provided
        if args.data_path:
            runner.config.data_path = args.data_path
        
        # Run sweep
        study = runner.run_sweep(n_trials=args.n_trials)
        
        return 0
        
    except Exception as e:
        print(f"Sweep failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())