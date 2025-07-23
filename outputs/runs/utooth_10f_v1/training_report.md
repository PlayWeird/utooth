# uTooth Training Report

**Experiment**: utooth_10f_v1

**Date**: 2025-07-22 20:34:55

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 50
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 42
- **Early Stopping**: Enabled (patience=10)

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1634 ± 0.1084 |
| Min Validation Loss | 0.1030 |
| Max Validation Loss | 0.4822 |
| Total Training Time | 2.74 hours |

## Fold Details

| Fold | Val Loss | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- |
| 1 | 0.4822 | 0 | 8.8 min | Yes |
| 2 | 0.1030 | 0 | 16.1 min | No |
| 3 | 0.1092 | 0 | 19.2 min | No |
| 4 | 0.1471 | 0 | 19.4 min | No |
| 5 | 0.1133 | 0 | 17.8 min | No |
| 6 | 0.1694 | 0 | 17.7 min | No |
| 7 | 0.1304 | 0 | 16.7 min | No |
| 8 | 0.1095 | 0 | 16.5 min | No |
| 9 | 0.1147 | 0 | 13.3 min | Yes |
| 10 | 0.1551 | 0 | 18.9 min | Yes |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v1/checkpoints/fold_0/utooth-epoch=27-val_loss=0.4822.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v1/checkpoints/fold_1/utooth-epoch=38-val_loss=0.1030.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v1/checkpoints/fold_2/utooth-epoch=47-val_loss=0.1092.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v1/checkpoints/fold_3/utooth-epoch=42-val_loss=0.1471.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v1/checkpoints/fold_4/utooth-epoch=40-val_loss=0.1133.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v1/checkpoints/fold_5/utooth-epoch=45-val_loss=0.1694.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v1/checkpoints/fold_6/utooth-epoch=42-val_loss=0.1304.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v1/checkpoints/fold_7/utooth-epoch=46-val_loss=0.1095.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v1/checkpoints/fold_8/utooth-epoch=28-val_loss=0.1147.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v1/checkpoints/fold_9/utooth-epoch=32-val_loss=0.1551.ckpt`
