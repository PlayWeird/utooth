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
* **74-82% Average IoU** across all experiments
* Successfully processes full-body CT scans
* Reduces segmentation time from 10 minutes to seconds per tooth

### Performance by Experiment:
- utooth_10f_v3: 74.2% ± 8.7% IoU (best overall)
- utooth_10f_v4: 74.2% ± 7.0% IoU  
- utooth_10f_v2: 73.7% ± 7.0% IoU
- utooth_10f_v1: 73.3% ± 11.2% IoU

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

2. Create a virtual environment:
```bash
python -m venv utooth_env
source utooth_env/bin/activate  # On Windows: utooth_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

**Quick test run** (2 epochs, 2 folds):
```bash
python scripts/train.py --test_run --experiment_name test_run
```

**Full training** (50 epochs, 5 folds):
```bash
python scripts/train.py --experiment_name production_v1 --use_wandb --max_epochs 50 --n_folds 5
```

**Using shell script**:
```bash
./scripts/run_training.sh --wandb --experiment-name production_v1
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
* Python 3.7+
* PyTorch
* PyTorch Lightning
* Pydicom
* NumPy
* Weights & Biases
* Additional requirements in requirements.txt

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

### Visualization
- `scripts/visualization/visualize_run.py` - Generate training metrics and performance charts
- `scripts/visualization/visualize_predictions.py` - Visualize model predictions
- `scripts/visualization/run_visualizations.sh` - Batch visualization generation

## Best Performing Model
- **Model**: utooth_10f_v3, fold 7
- **Checkpoint**: `outputs/runs/utooth_10f_v3/checkpoints/fold_7/utooth-epoch=61-val_loss=0.0844.ckpt`
- **Performance**: 85.0% IoU, 91.9% Dice Score

## Citation
If you find this work useful, please cite our paper [currently in review].

## Acknowledgments
* CT preprocessing utilities adapted from [Rachel Lea Ballantyne Draelos's ct-volume-preprocessing](https://github.com/rachellea/ct-volume-preprocessing)
* [NMDID (New Mexico Decedent Image Database)](https://nmdid.unm.edu/) for CT scan data
* [Nevada Center for Applied Research](https://www.unr.edu/ncar)
* ELEKTRONN3 for U-Net architecture base
