# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

uTooth is a deep learning project for automated tooth segmentation from CT scans. It uses a 3D U-Net architecture implemented in PyTorch to process full-body CT scans and segment individual teeth, currently focusing on canine teeth.

## Common Development Commands

### Running Jupyter Notebooks
The project uses Jupyter notebooks as the primary interface for different stages of the pipeline:
```bash
jupyter notebook preprocessing_dicom.ipynb  # DICOM preprocessing
jupyter notebook unet_trainer.ipynb        # Model training
jupyter notebook sweeps.ipynb              # Hyperparameter optimization
jupyter notebook model_tester.ipynb        # Model evaluation
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Training the Model
Training is performed through the `unet_trainer.ipynb` notebook which uses PyTorch Lightning and Weights & Biases for logging. The training configuration includes:
- GPU training with DataParallel strategy (`strategy='dp'`)
- Weights & Biases integration for experiment tracking
- Model checkpointing based on validation loss

### Hyperparameter Optimization
Use `sweeps.ipynb` to run hyperparameter searches with Weights & Biases. Configuration is defined in `wandb_config.yaml`.

## Architecture and Key Components

### Core Modules
- **unet.py**: 3D U-Net implementation (modified from ELEKTRONN3)
  - Configurable depth (3-6 blocks)
  - Optional attention mechanisms
  - Adjustable starting filters (8, 16, or 32)

- **ct_utils.py**: CT scan preprocessing utilities
  - DICOM file loading and conversion to Hounsfield units
  - Jaw isolation using HU thresholding
  - Volume resampling and normalization

- **loss.py**: Focal Tversky Loss implementation
  - Handles class imbalance in medical imaging
  - Configurable alpha and gamma parameters

- **volume_dataloader.py**: PyTorch Lightning DataModule
  - Handles loading and batching of CT scan volumes
  - Train/validation/test split management

### Data Flow
1. Raw DICOM files → `preprocessing_dicom.ipynb` → Preprocessed volumes
2. Preprocessed volumes → `volume_dataloader.py` → Batched tensors
3. Batched tensors → `unet.py` → Segmentation predictions
4. Predictions → `loss.py` → Training optimization

### Data Organization
CT scan data is stored in `/home/gaetano/utooth/DATA/` with individual cases (e.g., case-185030). The project expects DICOM format input from the NMDID database.

## Important Notes
- The project achieves 91.3% IOU accuracy with 0.098 Focal Tversky Loss
- No formal test suite exists - evaluation is done through notebooks
- No linting configuration is present
- GPU memory management: `torch.cuda.empty_cache()` is used in training loops to prevent CUDA out of memory errors