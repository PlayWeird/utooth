# uTooth Training Report

**Experiment**: utooth_10f_v3_corrected

**Date**: 2025-08-12 16:29:18

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 80
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 42
- **Early Stopping**: Disabled

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1619 ± 0.0886 |
| Min Validation Loss | 0.0962 |
| Max Validation Loss | 0.3843 |
| Mean Validation Accuracy | 0.7452 ± 0.1056 |
| Min Validation Accuracy | 0.5285 |
| Max Validation Accuracy | 0.8479 |
| Total Training Time | 0.00 hours |

## Fold Details

| Fold | Val Loss | Val Accuracy | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1282 | 0.7755 | 0 | 14.1 min | No |
| 2 | 0.0962 | 0.8479 | 0 | 15.9 min | No |
| 3 | 0.3843 | 0.5285 | 0 | 13.5 min | No |
| 4 | 0.1432 | 0.7445 | 0 | 15.8 min | No |
| 5 | 0.2719 | 0.5567 | 0 | 20.5 min | No |
| 6 | 0.1357 | 0.7622 | 0 | 16.5 min | No |
| 7 | 0.1004 | 0.8214 | 0 | 16.8 min | No |
| 8 | 0.1046 | 0.8157 | 0 | 16.3 min | No |
| 9 | 0.1486 | 0.7838 | 0 | 15.8 min | No |
| 10 | 0.1059 | 0.8160 | 0 | 15.9 min | No |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_0/utooth-epoch=36-val_loss=0.1282.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_1/utooth-epoch=43-val_loss=0.0962.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_2/utooth-epoch=43-val_loss=0.3843.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_3/utooth-epoch=33-val_loss=0.1432.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_4/utooth-epoch=31-val_loss=0.2719.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_5/utooth-epoch=39-val_loss=0.1357.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_6/utooth-epoch=36-val_loss=0.1004.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_7/utooth-epoch=38-val_loss=0.1046.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_8/utooth-epoch=36-val_loss=0.1486.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v3_corrected/checkpoints/fold_9/utooth-epoch=48-val_loss=0.1059.ckpt`
