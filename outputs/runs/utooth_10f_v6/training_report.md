# uTooth Training Report

**Experiment**: utooth_10f_v6

**Date**: 2025-08-13 02:17:38

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 80
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 2025
- **Early Stopping**: Disabled

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.2023 ± 0.1108 |
| Min Validation Loss | 0.0830 |
| Max Validation Loss | 0.4231 |
| Mean Validation Accuracy | 0.6609 ± 0.1560 |
| Min Validation Accuracy | 0.4088 |
| Max Validation Accuracy | 0.8463 |
| Total Training Time | 3.70 hours |

## Fold Details

| Fold | Val Loss | Val Accuracy | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1032 | 0.8107 | 62 | 21.3 min | No |
| 2 | 0.1800 | 0.7031 | 21 | 22.1 min | No |
| 3 | 0.4231 | 0.4197 | 70 | 21.7 min | No |
| 4 | 0.1364 | 0.7502 | 43 | 21.8 min | No |
| 5 | 0.3564 | 0.4088 | 48 | 22.2 min | No |
| 6 | 0.1454 | 0.7276 | 42 | 23.1 min | No |
| 7 | 0.2173 | 0.6006 | 69 | 21.6 min | No |
| 8 | 0.0830 | 0.8463 | 69 | 22.9 min | No |
| 9 | 0.2824 | 0.5180 | 38 | 22.8 min | No |
| 10 | 0.0955 | 0.8239 | 76 | 22.6 min | No |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v6/checkpoints/fold_0/utooth-epoch=62-val_loss=0.1032.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v6/checkpoints/fold_1/utooth-epoch=21-val_loss=0.1800.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v6/checkpoints/fold_2/utooth-epoch=70-val_loss=0.4231.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v6/checkpoints/fold_3/utooth-epoch=43-val_loss=0.1364.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v6/checkpoints/fold_4/utooth-epoch=48-val_loss=0.3564.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v6/checkpoints/fold_5/utooth-epoch=42-val_loss=0.1454.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v6/checkpoints/fold_6/utooth-epoch=69-val_loss=0.2173.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v6/checkpoints/fold_7/utooth-epoch=69-val_loss=0.0830.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v6/checkpoints/fold_8/utooth-epoch=38-val_loss=0.2824.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v6/checkpoints/fold_9/utooth-epoch=76-val_loss=0.0955.ckpt`
