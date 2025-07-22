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
* 91.3% Intersection-Over-Union (IOU) accuracy
* 0.098 Focal Tversky Loss
* Successfully processes full-body CT scans
* Reduces segmentation time from 10 minutes to seconds per tooth

## Project Structure

```
utooth/
├── src/                    # Source code modules
│   ├── models/            # Neural network architectures
│   │   └── unet.py        # 3D U-Net implementation (modified from ELEKTRONN3)
│   ├── data/              # Data loading and processing
│   │   ├── volume_dataloader.py       # PyTorch Lightning DataModule
│   │   └── volume_dataloader_kfold.py  # K-fold cross-validation data loader
│   ├── utils/             # Utility functions
│   │   └── ct_utils.py    # CT scan preprocessing utilities
│   └── losses/            # Loss functions
│       └── loss.py        # Focal Tversky Loss implementation
├── notebooks/             # Jupyter notebooks
│   ├── preprocessing_dicom.ipynb  # DICOM preprocessing pipeline
│   ├── unet_trainer.ipynb        # Main training notebook
│   ├── sweeps.ipynb              # Hyperparameter optimization
│   └── model_tester.ipynb        # Model evaluation and testing
├── scripts/               # Executable scripts
│   ├── train.py          # Standalone training script with K-fold CV
│   └── run_training.sh   # Bash script for training automation
├── configs/               # Configuration files
│   └── wandb_config.yaml # Weights & Biases sweep configuration
├── docs/                  # Documentation
│   ├── CLAUDE.md         # AI assistant instructions
│   └── TRAINING_GUIDE.md # Detailed training guide
├── DATA/                  # CT scan data (NIfTI format)
├── outputs/               # Model outputs (created during training)
│   ├── checkpoints/      # Saved model checkpoints
│   └── logs/             # Training logs
└── tests/                 # Unit tests (to be implemented)
```

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

### Training with Jupyter Notebook
```bash
cd notebooks
jupyter notebook unet_trainer.ipynb
```

### Training with Script
```bash
python scripts/train.py --data_path DATA/ --batch_size 5 --max_epochs 50
```

### Hyperparameter Optimization
```bash
cd notebooks
jupyter notebook sweeps.ipynb
```

## Recent Updates

* **Reorganized Project Structure**: Code is now organized into proper modules under `src/` directory
* **Standardized Output Paths**: All model checkpoints and logs are saved to `outputs/` directory
* **Improved Imports**: All modules now use proper relative imports
* **Enhanced Documentation**: Added comprehensive documentation in `docs/` directory

## Project Status
This repository contains a functional implementation of automated tooth segmentation using deep learning. The code was originally forked from [rachellea/ct-volume-preprocessing](https://github.com/rachellea/ct-volume-preprocessing) and has been extensively adapted for dental segmentation with PyTorch Lightning integration.

## Citation
If you find this work useful, please cite our paper [currently in review].

## Acknowledgments
* Original CT preprocessing code from Rachel Lea Ballantyne Draelos
* NMDID database for CT scan data
* Nevada Center for Applied Research
