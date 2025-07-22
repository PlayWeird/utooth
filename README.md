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
* `preprocessing_dicom.ipynb` - DICOM preprocessing pipeline
* `unet_trainer.ipynb` - Main training notebook with PyTorch Lightning
* `sweeps.ipynb` - Hyperparameter optimization experiments
* `model_tester.ipynb` - Model evaluation and testing
* `unet.py` - 3D U-Net implementation (modified from ELEKTRONN3)
* `ct_utils.py` - CT scan preprocessing utilities
* `loss.py` - Focal Tversky Loss implementation
* `volume_dataloader.py` - PyTorch Lightning DataModule for data handling
* `volume_dataloader_kfold.py` - K-fold cross-validation data loader
* `train.py` - Standalone training script

## Requirements
* Python 3.7+
* PyTorch
* PyTorch Lightning
* Pydicom
* NumPy
* Weights & Biases
* Additional requirements in requirements.txt

## Project Status
This repository is a work in progress. The code was originally forked from [rachellea/ct-volume-preprocessing](https://github.com/rachellea/ct-volume-preprocessing) and is being adapted for dental segmentation. Full technical documentation and usage instructions will be added as the project develops.

## Citation
If you find this work useful, please cite our paper [currently in review].

## Acknowledgments
* Original CT preprocessing code from Rachel Lea Ballantyne Draelos
* NMDID database for CT scan data
* Nevada Center for Applied Research
