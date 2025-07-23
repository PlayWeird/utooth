# uTooth Training Report

**Experiment**: utooth_5f_v1

**Date**: 2025-07-22 17:11:40

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 50
- **Batch Size**: 5
- **Number of Folds**: 5
- **Random Seed**: 42
- **Early Stopping**: Enabled (patience=10)

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1817 ± 0.0585 |
| Min Validation Loss | 0.1222 |
| Max Validation Loss | 0.2931 |
| Total Training Time | 1.01 hours |

## Fold Details

| Fold | Val Loss | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- |
| 1 | 0.2931 | 0 | 7.7 min | No |
| 2 | 0.1513 | 0 | 12.3 min | Yes |
| 3 | 0.1729 | 0 | 14.6 min | No |
| 4 | 0.1222 | 0 | 15.1 min | No |
| 5 | 0.1689 | 0 | 10.8 min | Yes |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_5f_v1/checkpoints/fold_0/utooth-epoch=42-val_loss=0.2931.ckpt`
- Fold 2: `outputs/runs/utooth_5f_v1/checkpoints/fold_1/utooth-epoch=33-val_loss=0.1513.ckpt`
- Fold 3: `outputs/runs/utooth_5f_v1/checkpoints/fold_2/utooth-epoch=41-val_loss=0.1729.ckpt`
- Fold 4: `outputs/runs/utooth_5f_v1/checkpoints/fold_3/utooth-epoch=45-val_loss=0.1222.ckpt`
- Fold 5: `outputs/runs/utooth_5f_v1/checkpoints/fold_4/utooth-epoch=23-val_loss=0.1689.ckpt`
