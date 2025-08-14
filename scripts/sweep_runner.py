#!/usr/bin/env python3
"""
uTooth Hyperparameter Sweep Runner with Multi-GPU Support
=========================================================

This script runs hyperparameter optimization using Optuna with parallel execution
across multiple GPUs for efficient model training and evaluation.
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
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed

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

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger


def run_single_trial(trial_params):
    """Run a single trial on a specific GPU - this runs in a separate process"""
    
    trial_number, gpu_id, config, sweep_dir, fold_splits, hyperparams = trial_params
    
    # Set this process to use only the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Recreate logger for this process
    logger = logging.getLogger(f"trial_{trial_number}_gpu_{gpu_id}")
    logger.setLevel(logging.INFO)
    
    # Add file handler
    log_file = sweep_dir / "logs" / f"trial_{trial_number}_gpu_{gpu_id}.log"
    log_file.parent.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info(f"Trial {trial_number} starting on GPU {gpu_id}")
    logger.info(f"Parameters: {hyperparams}")
    
    # Train on each fold
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        logger.info(f"Trial {trial_number} - Fold {fold_idx + 1}/{len(fold_splits)}")
        
        try:
            # Create data module
            dataset = CTScanDataModuleKFold(
                data_dir=config.data_path,
                batch_size=hyperparams['batch_size'],
                train_indices=train_indices,
                val_indices=val_indices
            )
            
            # Initialize model with suggested hyperparameters
            model = UNet(
                in_channels=1,
                out_channels=4,
                n_blocks=hyperparams['n_blocks'],
                start_filters=hyperparams['start_filters'],
                activation=hyperparams['activation'],
                normalization=hyperparams['normalization'],
                attention=hyperparams['attention'],
                conv_mode='same',
                dim=3,
                loss_alpha=hyperparams['loss_alpha'],
                loss_gamma=hyperparams['loss_gamma'],
                learning_rate=hyperparams['learning_rate']
            )
            
            # Setup callbacks
            trial_dir = sweep_dir / "trials" / f"trial_{trial_number}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks = []
            
            # Model checkpoint (only if configured)
            if config.save_checkpoints:
                checkpoint_dir = sweep_dir / "checkpoints" / f"trial_{trial_number}" / f"fold_{fold_idx}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint = ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=checkpoint_dir,
                    filename='best',
                    save_top_k=1,
                    mode='min'
                )
                callbacks.append(checkpoint)
            
            # Early stopping (only if patience is reasonable)
            if config.early_stopping_patience < 10000:  # If patience is very high, skip early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=config.early_stopping_patience,
                    mode='min',
                    verbose=False
                )
                callbacks.append(early_stopping)
            
            # Metrics callback
            metrics_callback = MetricsCallback()
            callbacks.append(metrics_callback)
            
            # Setup logger
            csv_logger = CSVLogger(
                save_dir=trial_dir,
                name=f'fold_{fold_idx}',
                version=None
            )
            
            # Initialize trainer - uses GPU 0 since we've set CUDA_VISIBLE_DEVICES
            trainer = Trainer(
                accelerator='gpu',
                devices=1,
                max_epochs=config.max_epochs,
                callbacks=callbacks,
                logger=csv_logger,
                enable_progress_bar=False,
                log_every_n_steps=1,
                enable_model_summary=False
            )
            
            # Train the model
            trainer.fit(model, dataset)
            
            # Extract results
            if config.save_checkpoints and 'checkpoint' in locals():
                best_val_loss = checkpoint.best_model_score.item()
            else:
                best_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
                if hasattr(best_val_loss, 'item'):
                    best_val_loss = best_val_loss.item()
            
            fold_result = {
                'fold_idx': fold_idx,
                'val_loss': best_val_loss,
                'val_accu': metrics_callback.best_val_accu,
                'final_epoch': trainer.current_epoch,
                'train_samples': len(train_indices),
                'val_samples': len(val_indices)
            }
            fold_results.append(fold_result)
            
            # Aggressive memory cleanup after each fold
            del model, trainer, dataset, callbacks, metrics_callback
            if 'checkpoint' in locals():
                del checkpoint
            if 'early_stopping' in locals():
                del early_stopping
            if 'csv_logger' in locals():
                del csv_logger
                
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()  # Clear shared memory
                
        except Exception as e:
            logger.error(f"Trial {trial_number} fold {fold_idx} failed: {str(e)}")
            
            # Aggressive cleanup even on failure
            try:
                if 'model' in locals():
                    del model
                if 'trainer' in locals():
                    del trainer
                if 'dataset' in locals():
                    del dataset
                if 'callbacks' in locals():
                    del callbacks
                if 'metrics_callback' in locals():
                    del metrics_callback
                if 'checkpoint' in locals():
                    del checkpoint
                if 'early_stopping' in locals():
                    del early_stopping
                if 'csv_logger' in locals():
                    del csv_logger
                    
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
            except:
                pass  # Don't fail on cleanup
                
            raise e
    
    # Calculate aggregated metrics
    metrics = calculate_trial_metrics(fold_results)
    
    # Save trial details
    trial_details = {
        'trial_number': trial_number,
        'gpu_id': gpu_id,
        'parameters': hyperparams,
        'metrics': metrics,
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat()
    }
    
    trial_file = sweep_dir / "trials" / f"trial_{trial_number}.json"
    with open(trial_file, 'w') as f:
        json.dump(trial_details, f, indent=2)
    
    logger.info(f"Trial {trial_number} completed with val_loss: {metrics['mean_val_loss']:.4f}")
    
    # Final aggressive cleanup after trial completion
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        
        # Force reset GPU memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    logger.info(f"Trial {trial_number} GPU memory cleaned")
    
    return trial_number, metrics['mean_val_loss'], hyperparams, metrics


def main():
    parser = argparse.ArgumentParser(description='Parallel GPU Hyperparameter Sweep')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Number of trials to run')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data path from config')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_sweep_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    # Override data path if provided
    if args.data_path:
        config.data_path = args.data_path
    
    # Create sweep directory
    sweep_dir = create_sweep_directory(config)
    
    # Setup logging
    logger = setup_logging(sweep_dir, verbose=True)
    logger.info(f"Sweep directory created: {sweep_dir}")
    
    # Create fold splits
    fold_splits = create_fold_indices(config.data_path, config.k_folds, config.seed)
    logger.info(f"Created {len(fold_splits)} fold splits")
    
    # Determine number of trials
    if args.n_trials is None:
        n_trials = config.trials_per_gpu * config.n_gpus
    else:
        n_trials = args.n_trials
    
    # Get available GPUs
    n_gpus = min(config.n_gpus, torch.cuda.device_count())
    available_gpus = list(range(n_gpus))
    logger.info(f"Using {n_gpus} GPUs: {available_gpus}")
    
    # Create Optuna study
    study = create_optuna_study(config, sweep_dir)
    
    # Enqueue baseline trial
    if config.baseline:
        enqueue_baseline_trial(study, config.baseline)
    
    logger.info(f"Starting sweep with {n_trials} trials on {n_gpus} GPUs")
    
    # Create trial parameters
    trial_params_list = []
    for i in range(n_trials):
        trial = study.ask()
        suggested_params = suggest_hyperparameters(trial, config.hyperparameters)
        gpu_id = available_gpus[i % n_gpus]  # Round-robin GPU assignment
        
        trial_params = (
            trial.number,
            gpu_id,
            config,
            sweep_dir,
            fold_splits,
            suggested_params
        )
        trial_params_list.append(trial_params)
        
        # Don't tell the study yet - we'll do it after completion
    
    # Run trials in parallel using ProcessPoolExecutor
    start_time = time.time()
    completed_trials = 0
    
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        # Submit all trials
        future_to_trial = {
            executor.submit(run_single_trial, params): params[0] 
            for params in trial_params_list
        }
        
        # Process completed trials
        for future in as_completed(future_to_trial):
            trial_number = future_to_trial[future]
            try:
                trial_num, val_loss, params, metrics = future.result()
                
                # Update the study with the result using trial number
                study.tell(trial_num, val_loss)
                
                completed_trials += 1
                logger.info(f"Progress: {completed_trials}/{n_trials} trials completed")
                
                # Log to console
                print(f"Trial {trial_num} completed:")
                print(f"  Mean validation loss: {metrics['mean_val_loss']:.4f} ± {metrics['std_val_loss']:.4f}")
                print(f"  Mean validation accuracy: {metrics['mean_val_accu']:.4f} ± {metrics['std_val_accu']:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {trial_number} failed: {str(e)}")
                # Mark trial as failed in study using trial number
                study.tell(trial_number, float('inf'))
    
    total_time = time.time() - start_time
    
    # Generate final report
    logger.info("Generating final report...")
    stats = get_study_statistics(study)
    
    # Save statistics
    stats_file = sweep_dir / "reports" / "sweep_statistics.json"
    stats_file.parent.mkdir(exist_ok=True)
    
    stats['total_time_hours'] = total_time / 3600
    stats['trials_per_hour'] = len(study.trials) / (total_time / 3600) if total_time > 0 else 0
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
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
        print(f"Best parameters: {stats['best_params']}")
    print(f"Results saved to: {sweep_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())