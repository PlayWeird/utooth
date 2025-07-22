# uTooth Training Guide

## Overview
The uTooth training system provides a production-ready pipeline for automated tooth segmentation model training with comprehensive statistics tracking and cross-validation support.

## Features
- ✅ **Production Ready**: 50 epochs per fold by default
- ✅ **K-Fold Cross Validation**: 5-fold validation (configurable)
- ✅ **Comprehensive Statistics**: Detailed metrics, timing, and performance tracking
- ✅ **Early Stopping**: Prevents overfitting with patience-based stopping
- ✅ **Multiple Loggers**: CSV metrics + optional Weights & Biases integration
- ✅ **Automated Reporting**: JSON, CSV, and Markdown reports generated
- ✅ **Checkpoint Management**: Top-3 models saved per fold with last checkpoint
- ✅ **Experiment Tracking**: Named experiments with timestamped results
- ✅ **Progress Monitoring**: Real-time monitoring tools included
- ✅ **Reproducible Results**: Fixed random seeds and deterministic training

## Usage

### Quick Start - Test Mode (2 epochs)
```bash
# Test with 2 epochs
python scripts/train.py --test_run

# Test with custom experiment name
python scripts/train.py --test_run --experiment_name my_test

# Test with W&B logging
python scripts/train.py --test_run --use_wandb
```

### Full Training (50 epochs per fold)
```bash
# Full training with default settings
python scripts/train.py

# Full training with W&B logging
python scripts/train.py --use_wandb

# Full training with custom experiment name
python scripts/train.py --experiment_name production_run_v1

# Full training with different number of folds
python scripts/train.py --n_folds 3 --experiment_name 3fold_experiment
```

### Using the Shell Script
```bash
# Test mode
./scripts/run_training.sh --test

# Full training
./scripts/run_training.sh

# Full training with W&B and custom name
./scripts/run_training.sh --wandb --experiment-name production_v1
```

## Hyperparameters (from notebook)
- **Batch size**: 5
- **Learning rate**: 2e-3 (0.002)
- **Loss alpha**: 0.5236
- **Loss gamma**: 1.0
- **U-Net blocks**: 4
- **Start filters**: 32
- **Max epochs**: 50 (2 for test mode)
- **Early stopping patience**: 10 epochs

## Output Structure
All outputs are organized under `outputs/runs/EXPERIMENT_NAME/`:

```
outputs/runs/EXPERIMENT_NAME/
├── config.json                 # Training configuration
├── results_summary.json        # Complete results with statistics
├── results_summary.csv         # Results in tabular format
├── training_report.md          # Human-readable report
├── checkpoints/                # Model checkpoints
│   ├── fold_0/                # Fold 0 checkpoints
│   │   ├── utooth-epoch=XX-val_loss=X.XXXX.ckpt
│   │   └── last.ckpt          # Last epoch checkpoint
│   └── fold_N/                # Additional folds...
├── fold_statistics/           # Per-fold detailed statistics
│   ├── fold_0_stats.json
│   └── fold_N_stats.json
└── metrics/                   # CSV metrics logs
    ├── fold_0/               # PyTorch Lightning CSV logs
    │   └── metrics.csv
    └── fold_N/
```

## Monitoring Progress

### Real-time Monitoring
```bash
# List all experiments
python scripts/monitor_training.py list

# Monitor specific experiment (refreshes every 10 seconds)
python scripts/monitor_training.py monitor --experiment my_experiment

# Show detailed experiment information
python scripts/monitor_training.py details --experiment my_experiment
```

### Generated Reports
After training completes, check the automatically generated:
- **Markdown Report**: `training_report.md` - Human-readable summary
- **JSON Results**: `results_summary.json` - Machine-readable complete results
- **CSV Results**: `results_summary.csv` - Tabular data for analysis

## Expected Results
Based on previous experiments:
- **Validation Loss**: ~0.098 (Focal Tversky Loss)
- **IoU Accuracy**: ~91.3%
- **Training Time**: ~20-30 minutes per fold (50 epochs on RTX 3080 Ti)
- **Total Training Time**: 2-3 hours for 5-fold cross-validation

## Key Features Explained

### Early Stopping
- Monitors validation loss with patience of 10 epochs
- Prevents overfitting and saves training time
- Can be disabled with `--no_early_stopping`

### Statistics Tracking
- **Per-fold metrics**: Best epoch, training time, early stopping status
- **Overall statistics**: Mean ± std validation loss, total time
- **Model checkpoints**: Paths to best models for each fold
- **Timing information**: Precise training duration tracking

### Experiment Management
- **Unique naming**: Auto-generated or custom experiment names
- **Configuration saving**: All parameters saved for reproducibility
- **Result organization**: Clean directory structure for easy analysis
- **Multiple formats**: JSON, CSV, and Markdown outputs for different use cases

## Advanced Usage

### Custom Configuration
```bash
# Custom hyperparameters
python scripts/train.py \
  --max_epochs 100 \
  --batch_size 3 \
  --n_folds 10 \
  --random_seed 123 \
  --experiment_name custom_config
```

### Configuration Files
Default training configuration is available in `configs/training_config.yaml`.

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce `--batch_size` from 5 to 3 or 2
- **Slow Training**: Verify GPU usage with `nvidia-smi`
- **Inconsistent Results**: Random seed is set automatically for reproducibility
- **Missing Dependencies**: Install with `pip install -r requirements.txt`

### Validation
```bash
# Quick validation that everything works
python scripts/train.py --test_run --n_folds 2 --experiment_name validation_test
```

### Performance Optimization
- Use smaller batch sizes if running out of GPU memory
- Consider mixed precision training for faster training (future feature)
- Monitor GPU utilization to ensure efficient resource usage

## Next Steps
After training:
1. Review the generated `training_report.md`
2. Analyze results in `results_summary.csv`
3. Use best model checkpoints for inference
4. Compare experiments using the monitoring tools

## File Reference
- **Training Script**: `scripts/train.py`
- **Shell Script**: `scripts/run_training.sh`
- **Monitoring Tool**: `scripts/monitor_training.py`
- **Configuration**: `configs/training_config.yaml`
- **Data Directory**: `DATA/` (CT scan cases)
- **Output Directory**: `outputs/runs/EXPERIMENT_NAME/`