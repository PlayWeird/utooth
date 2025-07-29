# uTooth Scripts Directory

This directory contains all scripts for training, analysis, and visualization of the uTooth model.

## Directory Structure

```
scripts/
├── analysis/           # Analysis and evaluation scripts
├── visualization/      # Visualization and plotting scripts  
├── testing/           # Testing and debugging scripts
├── train.py           # Main training script
├── monitor_training.py # Training monitoring utilities
└── run_training.sh    # Training shell script
```

## Core Training Scripts

### `train.py`
Main K-fold cross-validation training script with corrected accuracy metrics.

**Usage:**
```bash
python scripts/train.py --experiment_name utooth_10f_new --n_folds 10 --max_epochs 50
```

**Key Features:**
- K-fold cross validation
- Resume/restart capabilities
- Multiple metric logging (IoU, Dice, Binary IoU)
- Automatic checkpoint management
- Comprehensive experiment tracking

### `monitor_training.py`
Utilities for monitoring ongoing training runs.

### `run_training.sh`
Shell script wrapper for training with common configurations.

## Analysis Scripts (`analysis/`)

### `analyze_existing_runs.py`
Re-evaluates existing training runs with corrected accuracy metrics.

**Usage:**
```bash
python scripts/analysis/analyze_existing_runs.py --experiments utooth_10f_v3 utooth_10f_v4
```

### `find_best_results.py`
Identifies the best performing models and folds from analysis results.

### `analyze_fold_performance.py`
Analyzes which fold indices perform best across experiments and creates visualizations.

### `analyze_fold_cases.py`
Identifies which specific cases are in the best/worst performing folds.

## Visualization Scripts (`visualization/`)

### `visualize_predictions.py`
Creates detailed visualizations comparing model predictions with ground truth.

**Usage:**
```bash
python scripts/visualization/visualize_predictions.py --experiment utooth_10f_v3 --fold 1 --num_samples 5
```

**Outputs:**
- Axial slice comparisons
- Maximum intensity projections
- Per-class segmentation masks
- Quantitative metrics per sample

### `visualize_run.py`
Creates summary visualizations for training runs.

## Testing Scripts (`testing/`)

### `test_corrected_accuracy.py`
Tests the corrected accuracy metrics on existing checkpoints.

### `test_accuracy_calculation.py`
Debugging script for understanding accuracy calculation issues.

## Usage Examples

### Train a New Model
```bash
python scripts/train.py \
    --experiment_name utooth_corrected_metrics \
    --n_folds 10 \
    --max_epochs 50 \
    --use_wandb
```

### Analyze Existing Results
```bash
# Get comprehensive analysis
python scripts/analysis/analyze_existing_runs.py

# Find the best model
python scripts/analysis/find_best_results.py

# Visualize best model predictions
python scripts/visualization/visualize_predictions.py \
    --experiment utooth_10f_v3 \
    --fold 7 \
    --num_samples 10
```

### Test Metrics on Checkpoint
```bash
python scripts/testing/test_corrected_accuracy.py
```

## New Features Added

All scripts now use the corrected accuracy metrics implemented in `src/models/accuracy_metrics.py`:

- **Corrected IoU**: Properly handles multi-class labels with shape (B, 1, C, D, H, W)
- **Dice Coefficient**: Medical imaging standard metric
- **Binary IoU**: Simplified tooth vs background segmentation

## Output Locations

- **Training outputs**: `outputs/runs/{experiment_name}/`
- **Analysis results**: `analysis/`
- **Visualizations**: `outputs/runs/{experiment_name}/visualizations/`

## Dependencies

Main dependencies are listed in `requirements.txt`. Key packages:
- PyTorch Lightning
- torchmetrics
- nibabel (NIfTI file handling)
- scikit-learn (K-fold splitting)
- matplotlib (visualizations)
- pandas (analysis)

For visualization scripts, ensure matplotlib backend is properly configured for your environment.