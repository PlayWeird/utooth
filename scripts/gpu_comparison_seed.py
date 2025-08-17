#!/usr/bin/env python3
"""
GPU Comparison Training for Seed 2113
======================================

This script runs seed 2113 on different GPUs sequentially (like the original setup)
to isolate GPU-specific effects and compare with the original results.

Each GPU will run all 10 folds of seed 2113 independently and sequentially.
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


def run_seed_on_gpu(gpu_params):
    """Run seed 2113 on a specific GPU - all 10 folds sequentially (like original)"""
    gpu_id, seed, output_dir, data_path, max_epochs = gpu_params
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create GPU-specific output directory
    gpu_dir = output_dir / f"gpu_{gpu_id}"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"STARTING SEED {seed} ON GPU {gpu_id}")
    print(f"Running all 10 folds sequentially (like original seed search)")
    print(f"Output: {gpu_dir}")
    print(f"{'='*70}\n")
    
    # Create fold indices - CRITICAL: Must match original seed search exactly
    fold_indices = create_fold_indices(data_path, n_folds=10, random_seed=seed)
    
    # Track metrics for all folds
    all_results = []
    
    # Train each fold sequentially (exactly like original seed search)
    for fold_idx, (train_indices, val_indices) in enumerate(fold_indices):
        print(f"\n{'='*60}")
        print(f"GPU {gpu_id} - Training Fold {fold_idx} (Seed {seed})")
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        print(f"{'='*60}")
        
        # Create fold-specific output directory
        fold_dir = gpu_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for this fold - EXACTLY like original
        torch.manual_seed(seed + fold_idx)
        np.random.seed(seed + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + fold_idx)
        
        try:
            # Create data module with exact same parameters as seed search
            data_module = CTScanDataModuleKFold(
                data_dir=str(data_path),
                batch_size=6,  # Same as original seed search
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
            
            # Callbacks - NO early stopping for extended training
            checkpoint = ModelCheckpoint(
                dirpath=fold_dir / "checkpoints",
                filename=f"fold_{fold_idx}_{{epoch:03d}}_{{val_dice:.4f}}",
                monitor='val_dice',
                mode='max',
                save_top_k=3,
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
            
            # Trainer - NO early stopping, exactly like original but extended epochs
            trainer = Trainer(
                max_epochs=max_epochs,
                accelerator='gpu',
                devices=[0],  # Use GPU 0 within this process (after CUDA_VISIBLE_DEVICES)
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
                
            all_results.append(fold_result)
            
            print(f"GPU {gpu_id} - Fold {fold_idx} completed: val_dice={best_val_dice:.4f} at epoch {best_epoch}")
            
        except Exception as e:
            print(f"ERROR in GPU {gpu_id} - Fold {fold_idx}: {e}")
            # Continue with other folds
            continue
    
    # Calculate and save summary statistics for this GPU
    if all_results:
        dice_scores = [r['val_dice'] for r in all_results]
        
        gpu_summary = {
            'gpu_id': gpu_id,
            'seed': seed,
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'min_dice': np.min(dice_scores),
            'max_dice': np.max(dice_scores),
            'n_valid_folds': len(all_results),
            'fold_metrics': all_results,
            'max_epochs': max_epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save GPU summary
        with open(gpu_dir / 'gpu_results.json', 'w') as f:
            json.dump(gpu_summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"GPU {gpu_id} COMPLETED - Seed {seed}")
        print(f"Mean Dice: {gpu_summary['mean_dice']:.4f} ± {gpu_summary['std_dice']:.4f}")
        print(f"Valid folds: {gpu_summary['n_valid_folds']}/10")
        print(f"{'='*70}\n")
        
        return gpu_summary
    else:
        print(f"GPU {gpu_id} - No successful folds!")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare seed 2113 performance across different GPUs')
    parser.add_argument('--seed', type=int, default=2113, help='Seed to test')
    parser.add_argument('--max_epochs', type=int, default=150, help='Maximum epochs')
    parser.add_argument('--data_path', type=str, default='/home/user/utooth/DATA/', 
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/gpu_comparison_seed_XXXX_TIMESTAMP)')
    parser.add_argument('--n_gpus', type=int, default=3,
                        help='Number of GPUs to test (default: 3)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"outputs/gpu_comparison_seed_{args.seed}_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'seed': args.seed,
        'max_epochs': args.max_epochs,
        'data_path': args.data_path,
        'n_folds': 10,
        'n_gpus': args.n_gpus,
        'timestamp': datetime.now().isoformat(),
        'description': 'Run same seed on different GPUs to isolate GPU-specific effects',
        'training_mode': 'sequential_folds_per_gpu'
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get available GPUs
    n_gpus = min(torch.cuda.device_count(), args.n_gpus)
    print(f"Using {n_gpus} GPUs for comparison")
    
    # Prepare GPU tasks
    gpu_tasks = []
    for gpu_id in range(n_gpus):
        gpu_tasks.append((
            gpu_id,
            args.seed,
            output_dir,
            args.data_path,
            args.max_epochs
        ))
    
    print(f"\nStarting GPU comparison for seed {args.seed}")
    print(f"Each of {n_gpus} GPUs will run all 10 folds sequentially")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Output directory: {output_dir}\n")
    
    # Run on different GPUs in parallel (each GPU runs all folds sequentially)
    all_gpu_results = []
    
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        future_to_gpu = {executor.submit(run_seed_on_gpu, task): task[0] for task in gpu_tasks}
        
        for future in as_completed(future_to_gpu):
            gpu_id = future_to_gpu[future]
            try:
                result = future.result()
                if result:
                    all_gpu_results.append(result)
                    print(f"\nGPU {gpu_id} completed successfully!")
                else:
                    print(f"\nGPU {gpu_id} failed!")
            except Exception as e:
                print(f"\nError in GPU {gpu_id}: {e}")
    
    # Analyze and compare results across GPUs
    if len(all_gpu_results) >= 2:
        print(f"\n{'='*80}")
        print(f"GPU COMPARISON RESULTS - Seed {args.seed}")
        print(f"{'='*80}")
        
        # Compare results
        comparison = {
            'seed': args.seed,
            'max_epochs': args.max_epochs,
            'n_gpus_completed': len(all_gpu_results),
            'gpu_results': all_gpu_results,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"{'GPU':<5} {'Mean Dice':<12} {'Std Dice':<12} {'Min Dice':<12} {'Max Dice':<12}")
        print("-" * 60)
        
        original_mean = 0.8310940742492676  # From original seed 2113
        
        for gpu_result in all_gpu_results:
            gpu_id = gpu_result['gpu_id']
            mean_dice = gpu_result['mean_dice']
            std_dice = gpu_result['std_dice']
            min_dice = gpu_result['min_dice']
            max_dice = gpu_result['max_dice']
            
            print(f"{gpu_id:<5} {mean_dice:<12.4f} {std_dice:<12.4f} {min_dice:<12.4f} {max_dice:<12.4f}")
            
            # Compare with original
            improvement = mean_dice - original_mean
            print(f"      Original: {original_mean:.4f}, Improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")
        
        # Check for consistency
        mean_dices = [r['mean_dice'] for r in all_gpu_results]
        variance_across_gpus = np.var(mean_dices)
        
        print(f"\nVariance across GPUs: {variance_across_gpus:.6f}")
        if variance_across_gpus < 1e-6:
            print("→ Results are essentially identical across GPUs")
        else:
            print("→ Results vary across GPUs - GPU assignment matters!")
        
        # Save comparison results
        with open(output_dir / 'comparison_results.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
    else:
        print("Not enough successful GPU runs for comparison!")


if __name__ == "__main__":
    main()