# uTooth Training Report

**Experiment**: utooth_10f_v3

**Date**: 2025-07-23 17:08:19

## Configuration

- **Data Path**: /home/gaetano/utooth/DATA/
- **Max Epochs**: 65
- **Batch Size**: 5
- **Number of Folds**: 10
- **Random Seed**: 2025
- **Early Stopping**: Disabled

## Results Summary

| Metric | Value |
| --- | --- |
| Mean Validation Loss | 0.1482 ± 0.0734 |
| Min Validation Loss | 0.0844 |
| Max Validation Loss | 0.3371 |
| Mean Validation Accuracy | 0.6299 ± 0.0466 |
| Min Validation Accuracy | 0.5734 |
| Max Validation Accuracy | 0.7142 |
| Total Training Time | 3.87 hours |

## Fold Details

| Fold | Val Loss | Val Accuracy | Best Epoch | Training Time | Early Stopped |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1157 | 0.5734 | 0 | 14.7 min | No |
| 2 | 0.1038 | 0.7142 | 0 | 28.4 min | No |
| 3 | 0.3371 | 0.6029 | 0 | 30.1 min | No |
| 4 | 0.1457 | 0.5776 | 0 | 23.3 min | No |
| 5 | 0.2227 | 0.5787 | 0 | 21.6 min | No |
| 6 | 0.1542 | 0.6061 | 0 | 21.7 min | No |
| 7 | 0.1025 | 0.6597 | 0 | 21.8 min | No |
| 8 | 0.0844 | 0.6769 | 0 | 21.7 min | No |
| 9 | 0.1158 | 0.6403 | 0 | 21.9 min | No |
| 10 | 0.1000 | 0.6691 | 0 | 26.8 min | No |

## Model Checkpoints

- Fold 1: `outputs/runs/utooth_10f_v3/checkpoints/fold_0/utooth-epoch=56-val_loss=0.1157.ckpt`
- Fold 2: `outputs/runs/utooth_10f_v3/checkpoints/fold_1/utooth-epoch=48-val_loss=0.1038.ckpt`
- Fold 3: `outputs/runs/utooth_10f_v3/checkpoints/fold_2/utooth-epoch=47-val_loss=0.3371.ckpt`
- Fold 4: `outputs/runs/utooth_10f_v3/checkpoints/fold_3/utooth-epoch=60-val_loss=0.1457.ckpt`
- Fold 5: `outputs/runs/utooth_10f_v3/checkpoints/fold_4/utooth-epoch=43-val_loss=0.2227.ckpt`
- Fold 6: `outputs/runs/utooth_10f_v3/checkpoints/fold_5/utooth-epoch=49-val_loss=0.1542.ckpt`
- Fold 7: `outputs/runs/utooth_10f_v3/checkpoints/fold_6/utooth-epoch=57-val_loss=0.1025.ckpt`
- Fold 8: `outputs/runs/utooth_10f_v3/checkpoints/fold_7/utooth-epoch=61-val_loss=0.0844.ckpt`
- Fold 9: `outputs/runs/utooth_10f_v3/checkpoints/fold_8/utooth-epoch=49-val_loss=0.1158.ckpt`
- Fold 10: `outputs/runs/utooth_10f_v3/checkpoints/fold_9/utooth-epoch=42-val_loss=0.1000.ckpt`
