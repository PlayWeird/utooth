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
* **Production Hyperparameter Sweep System** with Optuna optimization
* Multi-GPU parallel execution (3x RTX 3090 support)
* K-fold cross-validation support (5-fold default)
* GPU training with DataParallel strategy
* Comprehensive monitoring and visualization tools

## Results
* **84.3% IoU** (best single model: utooth_10f_v3, fold 4)
* **90.2% Dice Score** (medical imaging standard metric)
* **70-84% Average IoU** across all experiments
* Successfully processes full-body CT scans
* Reduces segmentation time from 10 minutes to seconds per canine

### Performance by Experiment (Corrected Metrics):
- utooth_10f_v3: 84.26% ± 5.55% IoU (best overall)
- utooth_10f_v2: 82.13% ± 5.77% IoU
- utooth_10f_v4: 82.13% ± 6.34% IoU  
- utooth_10f_v1: 73.31% ± 11.54% IoU
- utooth_5f_v1: 70.51% ± 5.99% IoU

## Project Structure

```
utooth/
├── src/                    # Source code modules
│   ├── models/            # Neural network architectures
│   │   ├── unet.py        # 3D U-Net implementation
│   │   └── accuracy_metrics.py  # IoU, Dice, and Binary IoU metrics
│   ├── data/              # Data loading and processing
│   │   ├── volume_dataloader.py       # PyTorch Lightning DataModule
│   │   └── volume_dataloader_kfold.py  # K-fold cross-validation data loader
│   ├── utils/             # Utility functions
│   │   └── ct_utils.py    # CT scan preprocessing utilities
│   └── losses/            # Loss functions
│       └── loss.py        # Focal Tversky Loss implementation
├── scripts/               # Executable scripts
│   ├── train.py          # Main training script with K-fold CV
│   ├── run_training.sh   # Bash script for training automation
│   ├── sweep_runner.py   # Production hyperparameter sweep runner
│   ├── monitor_sweep.py  # Real-time sweep monitoring
│   ├── sweep/            # Hyperparameter sweep system
│   │   ├── configs/      # Sweep configuration files
│   │   ├── core/         # Core sweep functionality
│   │   ├── monitoring/   # Monitoring and visualization
│   │   └── utils/        # Sweep utilities
│   └── visualization/     # Visualization scripts
│       ├── visualize_predictions.py   # Model prediction visualizations
│       ├── visualize_run.py          # Training run visualization
│       └── run_visualizations.sh     # Bash script for generating visualizations
├── notebooks/             # Jupyter notebooks
│   ├── unet_trainer.ipynb        # Original training notebook
│   ├── preprocessing_dicom.ipynb  # DICOM preprocessing pipeline
│   ├── sweeps.ipynb              # Hyperparameter optimization
│   └── model_tester.ipynb        # Model evaluation and testing
├── configs/               # Configuration files
│   ├── training_config.yaml # Training configuration
│   └── wandb_config.yaml    # Weights & Biases configuration
├── DATA/                  # CT scan data (NIfTI format)
├── outputs/               # Model outputs and results
│   └── runs/              # Training runs with checkpoints and metrics
│       ├── utooth_10f_v1/
│       ├── utooth_10f_v2/
│       ├── utooth_10f_v3/ # Best performing experiment
│       ├── utooth_10f_v4/
│       ├── utooth_10f_v5/
│       └── utooth_5f_v1/
└── requirements.txt       # Python dependencies
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PlayWeird/utooth.git
cd utooth
```

2. Create and activate virtual environment:
```bash
python -m venv utooth_env
source utooth_env/bin/activate  # On Windows: utooth_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Hyperparameter Sweep (Recommended for New Datasets)

**Production sweep** (90 trials, 3 GPUs, 5-fold CV):
```bash
python scripts/sweep_runner.py
```

**Monitor progress**:
```bash
python scripts/monitor_sweep.py --sweep_dir outputs/sweeps/latest --auto-detect
```

### Training with Optimized Parameters

**Quick test run** (2 epochs, 2 folds):
```bash
python scripts/train.py --test_run --experiment_name test_run
```

**Full training** (30 epochs, 5 folds with optimal hyperparameters):
```bash
python scripts/train.py --experiment_name production_v1 --max_epochs 30 --n_folds 5
```

**Using shell script**:
```bash
./scripts/run_training.sh --experiment-name production_v1
```

### Generate Visualizations

After training, generate performance visualizations:
```bash
# For specific experiment
python scripts/visualization/visualize_run.py utooth_10f_v3

# For all completed runs
bash scripts/visualization/run_visualizations.sh
```

## Requirements
* Python 3.8+
* PyTorch 2.0+
* PyTorch Lightning 2.0+
* NVIDIA RTX 3090 (or equivalent) for optimal performance
* 24GB+ VRAM recommended for hyperparameter sweeps
* All dependencies listed in requirements.txt

## Model Architecture & Hyperparameters

### Optimal Hyperparameters

Based on extensive cross-validation, the following hyperparameters are now used by default:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2e-3 | Initial learning rate with Adam optimizer |
| Loss Alpha | 0.55 | Alpha parameter for Focal Tversky Loss |
| Loss Beta | 0.45 | Beta parameter (1 - alpha) |
| Loss Gamma | 1.0 | Focal parameter |
| Network Blocks | 4 | Number of encoder/decoder blocks |
| Start Filters | 32 | Initial number of convolutional filters |
| Batch Size | 5 | Samples per batch |

### Learning Rate Schedule

The model uses a ReduceLROnPlateau scheduler:
- **Initial LR**: 2e-3
- **Reduction Factor**: 0.5 (halves the learning rate)
- **Patience**: 10 epochs (waits before reducing)
- **Min LR**: 1e-6 (lower bound)
- **Monitor**: Validation loss

This allows the model to:
1. Train quickly with high learning rate initially
2. Fine-tune with lower rates when validation loss plateaus
3. Achieve better convergence and final performance

## Hyperparameter Optimization

### Production Sweep System

The project includes a comprehensive hyperparameter sweep system powered by Optuna:

```bash
# Launch production sweep (90 trials across 3 GPUs)
python scripts/sweep_runner.py

# Monitor real-time progress 
python scripts/monitor_sweep.py --sweep_dir outputs/sweeps/latest --auto-detect
```

**Sweep Configuration:**
- **90 total trials** (30 per GPU) across 3x RTX 3090s
- **5-fold cross-validation** (450 total training runs)
- **Tree-structured Parzen Estimator** (TPE) optimization
- **Multi-GPU parallel execution** with queue management
- **Early stopping** with median pruning (patience=10)

**Resource Requirements:**
- **Runtime:** ~11-12 hours for complete optimization
- **VRAM:** ~12-16GB per GPU (within 24GB RTX 3090 capacity)
- **Storage:** ~50-100GB for results and checkpoints

**Search Space:**
```yaml
learning_rate: 1e-4 to 5e-3 (log scale)    # Learning dynamics
loss_alpha: 0.3 to 0.7 (step 0.05)         # Tversky loss balance  
loss_gamma: 0.75 to 2.0 (step 0.25)        # Focal loss focus
batch_size: [4, 5, 6, 8]                   # Memory vs convergence
start_filters: [16, 32, 64]                # Model capacity
n_blocks: [3, 4, 5]                        # Network depth
normalization: [batch, instance, group]    # Regularization type
activation: [relu, leaky, silu]             # Non-linearity choice
attention: [true, false]                   # Attention mechanism
```

**Expected Results:** 5-15% improvement over default parameters with high statistical confidence

### Sweep Outputs

Sweeps generate organized results in `outputs/sweeps/{experiment_name}/`:
```
├── results_summary.json        # Best hyperparameters & metrics
├── plots/                     # Optimization visualizations
│   ├── optimization_history.html
│   ├── param_importances.html
│   └── convergence_analysis.png
├── reports/                   # Analysis reports
└── trials/                    # Detailed trial data (90 files)
```

### Resume Capability

Sweeps automatically resume from interruption points:
- Loads existing Optuna study from database
- Skips completed trials and continues optimization
- Maintains all previous results and progress

## Advanced Usage

### Resume Interrupted Training
```bash
# Resume from latest checkpoint
python scripts/train.py --resume --experiment_name production_v1 --use_wandb

# Auto-resume without confirmation
python scripts/train.py --auto_resume --experiment_name production_v1 --use_wandb
```

### Custom Configuration
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

### Training with Jupyter Notebook
```bash
cd notebooks
jupyter notebook unet_trainer.ipynb
```

### Expected Results
- **Training Time**: ~2-3 hours for full 5-fold CV (RTX 3080 Ti)
- **Validation Loss**: ~0.098 (Focal Tversky Loss)
- **IoU Accuracy**: ~74-85%
- **Output Location**: `outputs/runs/EXPERIMENT_NAME/`

## Key Scripts

### Training
- `scripts/train.py` - Main training script with K-fold cross-validation
- `scripts/run_training.sh` - Shell wrapper for training
- `notebooks/unet_trainer.ipynb` - Original notebook implementation

### Evaluation
- `scripts/evaluate_trained_models.py` - Re-evaluate models with corrected metrics
- `scripts/analyze_cross_validation_results.py` - Analyze training logs

### Visualization
- `scripts/visualization/visualize_run.py` - Generate training metrics and performance charts
- `scripts/visualization/visualize_predictions.py` - Visualize model predictions
- `scripts/visualization/run_visualizations.sh` - Batch visualization generation

## Best Performing Model
- **Model**: utooth_10f_v3, fold 4
- **Checkpoint**: `outputs/runs/utooth_10f_v3/checkpoints/fold_4/utooth-epoch=43-val_loss=0.2227.ckpt`
- **Performance**: 88.7% IoU, 94.0% Dice Score

## Citation
If you find this work useful, please cite our paper [currently in review].

## Acknowledgments
* CT preprocessing utilities adapted from [Rachel Lea Ballantyne Draelos's ct-volume-preprocessing](https://github.com/rachellea/ct-volume-preprocessing)
* [NMDID (New Mexico Decedent Image Database)](https://nmdid.unm.edu/) for CT scan data
* [Nevada Center for Applied Research](https://www.unr.edu/ncar)
* ELEKTRONN3 for U-Net architecture base
