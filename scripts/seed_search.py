#!/usr/bin/env python3
"""
Seed Search Runner - Based on working sweep_runner.py
=====================================================

This script searches for optimal seeds using the best hyperparameters from sweep,
using the exact same training setup that worked in the sweep.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# No longer need sweep utilities since we're handling logging directly

# Import model components  
from src.data.volume_dataloader_kfold import CTScanDataModuleKFold
from src.models.unet import UNet
from scripts.train import MetricsCallback, create_fold_indices

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger


def run_single_seed(seed_params):
    """Run a single seed evaluation on a specific GPU - runs in separate process"""
    
    seed, gpu_id, config, output_dir, fold_splits, best_hyperparams = seed_params
    
    # Set this process to use only the assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Setup logging for this seed
    log_file = output_dir / "logs" / f"seed_{seed}_gpu_{gpu_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger for this seed
    logger = logging.getLogger(f"seed_{seed}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    logger.info(f"Starting seed {seed} on GPU {gpu_id}")
    logger.info(f"Using hyperparameters: {json.dumps(best_hyperparams, indent=2)}")
    
    # Track metrics for all folds
    fold_metrics = []
    all_dice_scores = []
    
    # Train each fold (following exact sweep_runner structure)
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        logger.info(f"Seed {seed} - Fold {fold_idx + 1}/{len(fold_splits)}")
        
        try:
            # Create data module (exact same as sweep)
            dataset = CTScanDataModuleKFold(
                data_dir=config['data_path'],
                batch_size=best_hyperparams['batch_size'],
                train_indices=train_indices,
                val_indices=val_indices
            )
            
            # Initialize model with best hyperparameters (exact same as sweep)
            model = UNet(
                in_channels=1,
                out_channels=4,
                n_blocks=best_hyperparams['n_blocks'],
                start_filters=best_hyperparams['start_filters'],
                activation=best_hyperparams['activation'],
                normalization=best_hyperparams['normalization'],
                attention=best_hyperparams['attention'],
                conv_mode='same',
                dim=3,
                loss_alpha=best_hyperparams['loss_alpha'],
                loss_gamma=best_hyperparams['loss_gamma'],
                learning_rate=best_hyperparams['learning_rate']
            )
            
            # Setup callbacks
            seed_dir = output_dir / "seeds" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks = []
            
            # Model checkpoint
            checkpoint_dir = seed_dir / "checkpoints" / f"fold_{fold_idx}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                monitor='val_dice',
                dirpath=checkpoint_dir,
                filename='best-{epoch:02d}-{val_dice:.4f}',
                save_top_k=1,
                mode='max',
                save_last=False
            )
            callbacks.append(checkpoint)
            
            # Early stopping with patience=20 as requested
            early_stopping = EarlyStopping(
                monitor='val_dice',
                patience=20,
                mode='max',
                verbose=False
            )
            callbacks.append(early_stopping)
            
            # Metrics callback
            metrics_callback = MetricsCallback()
            callbacks.append(metrics_callback)
            
            # CSV Logger - ensure directory exists
            csv_log_dir = seed_dir / f"fold_{fold_idx}"
            csv_log_dir.mkdir(parents=True, exist_ok=True)
            
            csv_logger = CSVLogger(
                save_dir=csv_log_dir,
                name="",
                version=""
            )
            
            # Trainer configuration (matching sweep settings)
            trainer = Trainer(
                max_epochs=75,  # As requested
                accelerator='gpu',
                devices=[0],  # Use GPU 0 within this process
                callbacks=callbacks,
                logger=csv_logger,
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=10,
                deterministic='warn'  # Use 'warn' instead of True to handle max_pool3d issue
            )
            
            # Set seed for this fold
            torch.manual_seed(seed + fold_idx)  # Vary seed per fold
            np.random.seed(seed + fold_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed + fold_idx)
            
            # Train the model
            trainer.fit(model, dataset)
            
            # Record metrics
            best_dice = metrics_callback.best_val_dice
            fold_metrics.append({
                'fold': fold_idx,
                'val_dice': best_dice,
                'val_loss': metrics_callback.best_val_loss,
                'epoch': metrics_callback.best_epoch
            })
            all_dice_scores.append(best_dice)
            
            logger.info(f"Fold {fold_idx} completed - Best Dice: {best_dice:.4f}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold_idx}: {str(e)}")
            fold_metrics.append({
                'fold': fold_idx,
                'val_dice': 0.0,
                'val_loss': float('inf'),
                'error': str(e)
            })
    
    # Calculate aggregate metrics
    valid_scores = [m['val_dice'] for m in fold_metrics if 'error' not in m]
    
    if valid_scores:
        result = {
            'seed': seed,
            'mean_dice': np.mean(valid_scores),
            'std_dice': np.std(valid_scores),
            'min_dice': np.min(valid_scores),
            'max_dice': np.max(valid_scores),
            'n_valid_folds': len(valid_scores),
            'fold_metrics': fold_metrics
        }
    else:
        result = {
            'seed': seed,
            'mean_dice': 0.0,
            'std_dice': 0.0,
            'error': 'All folds failed'
        }
    
    # Save seed results
    results_file = seed_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Seed {seed} completed - Mean Dice: {result.get('mean_dice', 0):.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Seed search using best hyperparameters from sweep")
    parser.add_argument('--n_seeds', type=int, default=180,
                       help='Number of seeds to evaluate')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting seed value')
    parser.add_argument('--n_gpus', type=int, default=3,
                       help='Number of GPUs to use')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of folds for cross-validation')
    parser.add_argument('--data_path', type=str, default='/home/user/utooth/DATA',
                       help='Path to data directory')
    parser.add_argument('--sweep_results', type=str,
                       default='outputs/sweeps/utooth_default_sweep_20250813_115210/reports/sweep_statistics.json',
                       help='Path to sweep results JSON with best hyperparameters')
    parser.add_argument('--output_base', type=str, default='outputs/seed_search',
                       help='Base directory for output')
    
    args = parser.parse_args()
    
    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_base) / f"seed_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "seeds").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    
    # Setup main logger
    main_log = output_dir / "logs" / "seed_search.log"
    main_log.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging manually since setup_logging expects different structure
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("seed_search")
    
    logger.info("="*60)
    logger.info("Starting Seed Search with Cross-Validation")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Number of seeds: {args.n_seeds}")
    logger.info(f"  Starting seed: {args.start_seed}")
    logger.info(f"  Number of GPUs: {args.n_gpus}")
    logger.info(f"  Number of folds: {args.n_folds}")
    logger.info(f"  Data path: {args.data_path}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Load best hyperparameters from sweep
    if Path(args.sweep_results).exists():
        with open(args.sweep_results, 'r') as f:
            sweep_data = json.load(f)
            best_hyperparams = sweep_data.get('best_params', {})
    else:
        # Fallback to known best parameters
        best_hyperparams = {
            "learning_rate": 0.0023378111371697686,
            "loss_alpha": 0.55,
            "loss_gamma": 2.0,
            "batch_size": 6,
            "start_filters": 32,
            "n_blocks": 5,
            "normalization": "instance",
            "activation": "leaky",
            "attention": False
        }
    
    logger.info(f"Using hyperparameters: {json.dumps(best_hyperparams, indent=2)}")
    
    # Create fold indices (same for all seeds for fair comparison)
    fold_splits = create_fold_indices(args.data_path, args.n_folds, random_seed=42)
    
    # Save configuration
    config = {
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'n_gpus': args.n_gpus,
        'n_folds': args.n_folds,
        'data_path': args.data_path,
        'hyperparameters': best_hyperparams,
        'timestamp': timestamp
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare seed parameters for parallel execution
    seed_params = []
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    
    for i, seed in enumerate(seeds):
        gpu_id = i % args.n_gpus  # Distribute seeds across GPUs
        seed_params.append((seed, gpu_id, config, output_dir, fold_splits, best_hyperparams))
    
    # Run seeds in parallel
    all_results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.n_gpus) as executor:
        futures = {executor.submit(run_single_seed, params): params[0] 
                  for params in seed_params}
        
        completed = 0
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                
                if 'mean_dice' in result:
                    logger.info(f"[{completed}/{args.n_seeds}] Seed {seed} completed: "
                              f"Mean Dice = {result['mean_dice']:.4f} ± {result['std_dice']:.4f}")
                else:
                    logger.error(f"[{completed}/{args.n_seeds}] Seed {seed} failed")
                    
            except Exception as e:
                logger.error(f"Seed {seed} crashed: {str(e)}")
                all_results.append({
                    'seed': seed,
                    'error': str(e),
                    'mean_dice': 0.0
                })
    
    # Sort results by performance
    valid_results = [r for r in all_results if 'mean_dice' in r and r['mean_dice'] > 0]
    valid_results.sort(key=lambda x: x['mean_dice'], reverse=True)
    
    # Save final results
    total_time = (time.time() - start_time) / 3600
    
    final_results = {
        'config': config,
        'all_results': all_results,
        'sorted_results': valid_results,
        'total_time_hours': total_time,
        'seeds_per_hour': len(all_results) / total_time if total_time > 0 else 0
    }
    
    with open(output_dir / "reports" / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Create summary
    if valid_results:
        best_seed = valid_results[0]
        
        summary = {
            'best_seed': best_seed['seed'],
            'best_mean_dice': best_seed['mean_dice'],
            'best_std_dice': best_seed['std_dice'],
            'n_evaluated': len(all_results),
            'n_successful': len(valid_results),
            'top_5_seeds': [
                {'seed': r['seed'], 'mean_dice': r['mean_dice'], 'std_dice': r['std_dice']}
                for r in valid_results[:5]
            ],
            'total_time_hours': total_time
        }
        
        with open(output_dir / "reports" / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("="*60)
        logger.info("SEED SEARCH COMPLETED")
        logger.info(f"Best Seed: {best_seed['seed']}")
        logger.info(f"Best Mean Dice: {best_seed['mean_dice']:.4f} ± {best_seed['std_dice']:.4f}")
        logger.info(f"Total Time: {total_time:.2f} hours")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        # Print top 10 seeds
        print("\nTop 10 Seeds:")
        print("-" * 40)
        for i, result in enumerate(valid_results[:10], 1):
            print(f"{i:2d}. Seed {result['seed']:3d}: {result['mean_dice']:.4f} ± {result['std_dice']:.4f}")
    else:
        logger.error("No successful seed evaluations!")
    
    return output_dir


if __name__ == "__main__":
    output_dir = main()
    print(f"\nResults saved to: {output_dir}")