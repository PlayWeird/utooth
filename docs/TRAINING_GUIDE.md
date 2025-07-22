# uTooth Training Script Guide

## Overview
This training script (`train.py`) converts the Jupyter notebook workflow into a production-ready Python script with k-fold cross validation.

## Features
- ✅ All hyperparameters preserved from the original notebook
- ✅ 5-fold cross validation (recommended for 48 samples)
- ✅ Automatic checkpoint saving for each fold
- ✅ Weights & Biases integration (optional)
- ✅ Progress tracking and validation metrics
- ✅ Reproducible with random seed

## Hyperparameters (from notebook)
- **Batch size**: 5
- **Learning rate**: 2e-3 (0.002)
- **Loss alpha**: 0.5236
- **Loss gamma**: 1.0
- **U-Net blocks**: 4
- **Start filters**: 32
- **Activation**: ReLU
- **Normalization**: Batch normalization
- **Attention**: False

## Usage

### Quick Test (2 epochs)
```bash
./run_training.sh
```

### Full Training (50 epochs)
```bash
source utooth_env/bin/activate
python train.py --max_epochs 50 --use_wandb
```

### Custom Configuration
```bash
python train.py \
    --data_path /home/gaetano/utooth/DATA/ \
    --max_epochs 50 \
    --batch_size 5 \
    --n_folds 5 \
    --random_seed 42 \
    --use_wandb
```

## Command Line Arguments
- `--data_path`: Path to data directory (default: /home/gaetano/utooth/DATA/)
- `--max_epochs`: Maximum training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 5)
- `--n_folds`: Number of cross-validation folds (default: 5)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--use_wandb`: Enable Weights & Biases logging
- `--test_run`: Quick test with only 2 epochs

## K-Fold Cross Validation
With 48 data samples, 5-fold cross validation provides:
- ~38-39 training samples per fold
- ~9-10 validation samples per fold
- Each sample is used for validation exactly once
- Final performance is averaged across all folds

## Output
- Checkpoints saved in `checkpoints/fold_{0-4}/`
- Best model for each fold based on validation loss
- Summary statistics printed at the end
- W&B logs (if enabled) under project "utooth_kfold"

## Files Created
1. `train.py` - Main training script
2. `volume_dataloader_kfold.py` - Extended data module with k-fold support
3. `run_training.sh` - Quick start script

## Next Steps
1. Run test mode to verify everything works
2. Launch full training with W&B logging
3. Best models will be in `checkpoints/fold_*/` directories
4. Use the model with lowest average validation loss