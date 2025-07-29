# uTooth Training Report

**Experiment**: utooth_10f_v2

**Date**: 2025-07-23 12:44:33

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 50
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 123
- **Early Stopping**: Enabled (patience=10)

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1624 ± 0.0616 |
| Min Validation Loss | 0.1060 |
| Max Validation Loss | 0.3253 |
| Total Training Time | 2.88 hours |

## Fold Details

| Fold | Val Loss | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- |
| 1 | 0.1719 | 0 | 14.0 min | No |
| 2 | 0.1176 | 0 | 11.9 min | Yes |
| 3 | 0.3253 | 0 | 11.0 min | Yes |
| 4 | 0.1542 | 0 | 17.0 min | No |
| 5 | 0.1106 | 0 | 17.6 min | No |
| 6 | 0.1226 | 0 | 18.0 min | No |
| 7 | 0.1060 | 0 | 20.1 min | No |
| 8 | 0.2015 | 0 | 18.2 min | Yes |
| 9 | 0.1454 | 0 | 24.7 min | No |
| 10 | 0.1688 | 0 | 20.6 min | Yes |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v2/checkpoints/fold_0/utooth-epoch=44-val_loss=0.1719.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v2/checkpoints/fold_1/utooth-epoch=25-val_loss=0.1176.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v2/checkpoints/fold_2/utooth-epoch=21-val_loss=0.3253.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v2/checkpoints/fold_3/utooth-epoch=44-val_loss=0.1542.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v2/checkpoints/fold_4/utooth-epoch=49-val_loss=0.1106.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v2/checkpoints/fold_5/utooth-epoch=38-val_loss=0.1226.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v2/checkpoints/fold_6/utooth-epoch=49-val_loss=0.1060.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v2/checkpoints/fold_7/utooth-epoch=30-val_loss=0.2015.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v2/checkpoints/fold_8/utooth-epoch=49-val_loss=0.1454.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v2/checkpoints/fold_9/utooth-epoch=31-val_loss=0.1688.ckpt`
