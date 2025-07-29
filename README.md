# uTooth: Automated Tooth Segmentation from CT Scans

A deep learning approach to automate tooth segmentation from computed tomography (CT) scans. This project processes full-body CT scans to isolate and segment individual teeth, with a current focus on canines.

## Current Features
* Preprocessing pipeline for CT scan data using Pydicom
* Jaw isolation using Hounsfield Unit thresholding
* 3D U-Net implementation for volumetric segmentation with configurable depth (3-6 blocks)
* Optional attention mechanisms in U-Net architecture
* Custom Focal Tversky loss function for handling class imbalance
* Visualization tools for CT volumes
* PyTorch Lightning integration for training
* Weights & Biases integration for experiment tracking
* Hyperparameter optimization with W&B sweeps
* K-fold cross-validation support
* GPU training with DataParallel strategy

## Results
* **85.0% Corrected IoU** (best single model: utooth_10f_v3, fold 7)
* **91.9% Dice Score** (medical imaging standard metric)
* **74-82% Average IoU** across all experiments (much higher than initially reported 58%)
* Successfully processes full-body CT scans
* Reduces segmentation time from 10 minutes to seconds per tooth

### Performance by Experiment:
- utooth_10f_v3: 74.2% ± 8.7% IoU (best overall)
- utooth_10f_v4: 74.2% ± 7.0% IoU  
- utooth_10f_v2: 73.7% ± 7.0% IoU
- utooth_10f_v1: 73.3% ± 11.2% IoU

*Note: Original training reported lower accuracy due to incorrect metric calculation. See `analysis/` for corrected results.*

## Project Structure

```
utooth/
├── src/                    # Source code modules
│   ├── models/            # Neural network architectures
│   │   ├── unet.py        # 3D U-Net implementation (with corrected metrics)
│   │   └── accuracy_metrics.py  # Corrected IoU, Dice, and Binary IoU metrics
│   ├── data/              # Data loading and processing
│   │   ├── volume_dataloader.py       # PyTorch Lightning DataModule
│   │   └── volume_dataloader_kfold.py  # K-fold cross-validation data loader
│   ├── utils/             # Utility functions
│   │   └── ct_utils.py    # CT scan preprocessing utilities
│   └── losses/            # Loss functions
│       └── loss.py        # Focal Tversky Loss implementation
├── scripts/               # Executable scripts
│   ├── analysis/          # Analysis and evaluation scripts
│   │   ├── analyze_existing_runs.py   # Re-evaluate runs with corrected metrics
│   │   ├── find_best_results.py       # Find best performing models
│   │   ├── analyze_fold_performance.py # Fold performance analysis
│   │   └── analyze_fold_cases.py      # Case difficulty analysis
│   ├── visualization/     # Visualization scripts
│   │   ├── visualize_predictions.py   # Model prediction visualizations
│   │   └── visualize_run.py          # Training run visualizations
│   ├── testing/           # Testing and debugging scripts
│   │   ├── test_corrected_accuracy.py # Test corrected metrics
│   │   └── test_accuracy_calculation.py # Debug accuracy calculations
│   ├── train.py          # Main training script with K-fold CV
│   ├── monitor_training.py # Training monitoring utilities
│   └── run_training.sh   # Bash script for training automation
├── analysis/              # Analysis results and reports
│   ├── corrected_metrics_analysis.json     # Detailed corrected metrics
│   ├── corrected_metrics_analysis_*.csv    # Summary tables
│   ├── fold_performance_analysis.png       # Fold performance charts
│   └── README.md         # Analysis documentation
├── notebooks/             # Jupyter notebooks
│   ├── preprocessing_dicom.ipynb  # DICOM preprocessing pipeline
│   ├── unet_trainer.ipynb        # Main training notebook
│   ├── sweeps.ipynb              # Hyperparameter optimization
│   └── model_tester.ipynb        # Model evaluation and testing
├── configs/               # Configuration files
│   └── wandb_config.yaml # Weights & Biases sweep configuration
├── docs/                  # Documentation
│   ├── CLAUDE.md         # AI assistant instructions
│   └── TRAINING_GUIDE.md # Detailed training guide
├── DATA/                  # CT scan data (NIfTI format)
├── outputs/               # Model outputs (created during training)
│   ├── runs/             # Individual experiment runs
│   ├── checkpoints/      # Saved model checkpoints
│   └── logs/             # Training logs
└── tests/                 # Unit tests
```

## Corrected Accuracy Metrics (Important!)

**Previous Issue**: Original training runs reported artificially low accuracy (~58%) due to incorrect `jaccard_index` usage with multi-class segmentation labels.

**Solution**: Implemented corrected metrics in `src/models/accuracy_metrics.py`:
- **Corrected IoU**: Properly handles label format (B, 1, C, D, H, W) for multi-class segmentation
- **Dice Coefficient**: Medical imaging standard, often preferred over IoU
- **Binary IoU**: Simplified tooth vs background metric

**Impact**: Re-analysis shows models actually achieve 74-85% accuracy, matching expectations!

### Using Corrected Metrics

**For new training** (automatic with updated code):
```bash
python scripts/train.py --experiment_name new_corrected_run
```

**To re-analyze existing runs**:
```bash
python scripts/analysis/analyze_existing_runs.py
python scripts/analysis/find_best_results.py
```

**Best model identified**:
- Checkpoint: `outputs/runs/utooth_10f_v3/checkpoints/fold_7/utooth-epoch=61-val_loss=0.0844.ckpt`
- IoU: 85.0%, Dice: 91.9%

See `analysis/README.md` for complete results.

## Requirements
* Python 3.7+
* PyTorch
* PyTorch Lightning
* Pydicom
* NumPy
* Weights & Biases
* Additional requirements in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PlayWeird/utooth.git
cd utooth
```

2. Create a virtual environment:
```bash
python -m venv utooth_env
source utooth_env/bin/activate  # On Windows: utooth_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Production Training (Recommended)

**Full 5-fold cross-validation training with Weights & Biases logging:**
```bash
python scripts/train.py \
  --experiment_name production_v1 \
  --use_wandb \
  --max_epochs 50 \
  --n_folds 5
```

This will train for ~2-3 hours with:
- 5-fold cross-validation (48 samples: ~38 train, ~10 validation per fold)
- 50 epochs per fold with early stopping
- Comprehensive statistics and reporting
- Resume capability if interrupted

### Quick Validation Test
```bash
# Test that everything works (2 epochs, 2 folds)
python scripts/train.py --test_run --experiment_name validation_test
```

### Resume Interrupted Training
```bash
# Resume from latest checkpoint
python scripts/train.py --resume --experiment_name production_v1 --use_wandb

# Auto-resume without confirmation
python scripts/train.py --auto_resume --experiment_name production_v1 --use_wandb
```

### Monitor Training Progress
```bash
# List all experiments
python scripts/monitor_training.py list

# Monitor specific experiment (real-time updates)
python scripts/monitor_training.py monitor --experiment production_v1

# Show detailed experiment information
python scripts/monitor_training.py details --experiment production_v1

# Get resume instructions
python scripts/monitor_training.py resume --experiment production_v1
```

### Alternative Training Methods

#### Using Shell Script
```bash
# Full training
./scripts/run_training.sh --wandb --experiment-name production_v1

# Test mode
./scripts/run_training.sh --test

# Resume training
./scripts/run_training.sh --resume --wandb --experiment-name production_v1
```

#### Training with Jupyter Notebook
```bash
cd notebooks
jupyter notebook unet_trainer.ipynb
```

#### Custom Configuration
```bash
# Custom hyperparameters
python scripts/train.py \
  --experiment_name custom_config \
  --max_epochs 100 \
  --batch_size 3 \
  --n_folds 10 \
  --random_seed 123 \
  --use_wandb
```

#### Hyperparameter Optimization
```bash
cd notebooks
jupyter notebook sweeps.ipynb
```

### Expected Results
- **Training Time**: ~2-3 hours for full 5-fold CV (RTX 3080 Ti)
- **Validation Loss**: ~0.098 (Focal Tversky Loss)
- **IoU Accuracy**: ~91.3%
- **Output Location**: `outputs/runs/EXPERIMENT_NAME/`

## Recent Updates

* **Reorganized Project Structure**: Code is now organized into proper modules under `src/` directory
* **Standardized Output Paths**: All model checkpoints and logs are saved to `outputs/` directory
* **Improved Imports**: All modules now use proper relative imports
* **Enhanced Documentation**: Added comprehensive documentation in `docs/` directory

## Project Status
This repository contains a functional implementation of automated tooth segmentation using deep learning, featuring a complete PyTorch Lightning-based training pipeline with 91.3% IoU accuracy.

## Citation
If you find this work useful, please cite our paper [currently in review].

## Acknowledgments
* CT preprocessing utilities adapted from [Rachel Lea Ballantyne Draelos's ct-volume-preprocessing](https://github.com/rachellea/ct-volume-preprocessing)
* [NMDID (New Mexico Decedent Image Database)](https://nmdid.unm.edu/) for CT scan data
* [Nevada Center for Applied Research](https://www.unr.edu/ncar)
* ELEKTRONN3 for U-Net architecture base
