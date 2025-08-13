# uTooth Training Report

**Experiment**: utooth_10f_v5

**Date**: 2025-08-12 22:33:01

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 80
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 2026
- **Early Stopping**: Disabled

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1603 ± 0.0924 |
| Min Validation Loss | 0.0847 |
| Max Validation Loss | 0.3792 |
| Mean Validation Accuracy | 0.7427 ± 0.1131 |
| Min Validation Accuracy | 0.5301 |
| Max Validation Accuracy | 0.8467 |
| Total Training Time | 4.85 hours |

## Fold Details

| Fold | Val Loss | Val Accuracy | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0848 | 0.8453 | 74 | 38.6 min | No |
| 2 | 0.2565 | 0.6088 | 60 | 27.6 min | No |
| 3 | 0.0988 | 0.8449 | 51 | 27.4 min | No |
| 4 | 0.1236 | 0.7858 | 57 | 27.4 min | No |
| 5 | 0.0914 | 0.8343 | 69 | 29.8 min | No |
| 6 | 0.0847 | 0.8467 | 63 | 30.8 min | No |
| 7 | 0.1465 | 0.7204 | 62 | 27.8 min | No |
| 8 | 0.3792 | 0.5301 | 22 | 26.8 min | No |
| 9 | 0.1118 | 0.8039 | 75 | 27.5 min | No |
| 10 | 0.2262 | 0.6068 | 72 | 27.2 min | No |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v5/checkpoints/fold_0/utooth-epoch=74-val_loss=0.0848.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v5/checkpoints/fold_1/utooth-epoch=60-val_loss=0.2565.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v5/checkpoints/fold_2/utooth-epoch=51-val_loss=0.0988.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v5/checkpoints/fold_3/utooth-epoch=57-val_loss=0.1236.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v5/checkpoints/fold_4/utooth-epoch=69-val_loss=0.0914.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v5/checkpoints/fold_5/utooth-epoch=63-val_loss=0.0847.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v5/checkpoints/fold_6/utooth-epoch=62-val_loss=0.1465.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v5/checkpoints/fold_7/utooth-epoch=22-val_loss=0.3792.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v5/checkpoints/fold_8/utooth-epoch=75-val_loss=0.1118.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v5/checkpoints/fold_9/utooth-epoch=72-val_loss=0.2262.ckpt`
