# uTooth: Automated Tooth Segmentation from CT Scans

A deep learning approach to automate tooth segmentation from computed tomography (CT) scans. This project processes full-body CT scans to isolate and segment individual teeth, with a current focus on canines.

## Current Features
* Preprocessing pipeline for CT scan data using Pydicom
* Jaw isolation using Hounsfield Unit thresholding
* 3D U-Net implementation for volumetric segmentation
* Custom Focal Tversky loss function
* Visualization tools for CT volumes

## Results
* 91.3% Intersection-Over-Union (IOU) accuracy
* 0.098 Focal Tversky Loss
* Successfully processes full-body CT scans
* Reduces segmentation time from 10 minutes to seconds per tooth

## Requirements
* Python 3.7+
* PyTorch
* Pydicom
* NumPy
* Additional requirements in requirements.txt

## Project Status
This repository is a work in progress. The code was originally forked from [rachellea/ct-volume-preprocessing](https://github.com/rachellea/ct-volume-preprocessing) and is being adapted for dental segmentation. Full technical documentation and usage instructions will be added as the project develops.

## Citation
If you find this work useful, please cite our paper [currently in review].

## Acknowledgments
* Original CT preprocessing code from Rachel Lea Ballantyne Draelos
* NMDID database for CT scan data
* Nevada Center for Applied Research
