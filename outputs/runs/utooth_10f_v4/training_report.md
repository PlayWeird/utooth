# uTooth Training Report

**Experiment**: utooth_10f_v4

**Date**: 2025-07-23 19:50:46

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 50
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 1
- **Early Stopping**: Disabled

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1574 ± 0.0677 |
| Min Validation Loss | 0.0933 |
| Max Validation Loss | 0.3069 |
| Mean Validation Accuracy | 0.6097 ± 0.0287 |
| Min Validation Accuracy | 0.5527 |
| Max Validation Accuracy | 0.6629 |
| Total Training Time | 2.59 hours |

## Fold Details

| Fold | Val Loss | Val Accuracy | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1435 | 0.6248 | 0 | 12.5 min | No |
| 2 | 0.3069 | 0.5878 | 0 | 16.4 min | No |
| 3 | 0.1418 | 0.6271 | 0 | 16.5 min | No |
| 4 | 0.2657 | 0.5527 | 0 | 16.3 min | No |
| 5 | 0.1085 | 0.5924 | 0 | 15.7 min | No |
| 6 | 0.1470 | 0.5964 | 0 | 14.5 min | No |
| 7 | 0.0975 | 0.6358 | 0 | 14.7 min | No |
| 8 | 0.1285 | 0.6077 | 0 | 16.5 min | No |
| 9 | 0.0933 | 0.6629 | 0 | 16.8 min | No |
| 10 | 0.1415 | 0.6094 | 0 | 15.8 min | No |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v4/checkpoints/fold_0/utooth-epoch=45-val_loss=0.1435.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v4/checkpoints/fold_1/utooth-epoch=46-val_loss=0.3069.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v4/checkpoints/fold_2/utooth-epoch=42-val_loss=0.1418.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v4/checkpoints/fold_3/utooth-epoch=43-val_loss=0.2657.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v4/checkpoints/fold_4/utooth-epoch=43-val_loss=0.1085.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v4/checkpoints/fold_5/utooth-epoch=47-val_loss=0.1470.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v4/checkpoints/fold_6/utooth-epoch=41-val_loss=0.0975.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v4/checkpoints/fold_7/utooth-epoch=46-val_loss=0.1285.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v4/checkpoints/fold_8/utooth-epoch=48-val_loss=0.0933.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v4/checkpoints/fold_9/utooth-epoch=38-val_loss=0.1415.ckpt`
