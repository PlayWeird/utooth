# Cross-Validation Results Summary

## Overview

This report presents the re-evaluated performance metrics for the uTooth segmentation models using the corrected IoU and Dice coefficient calculations.

## Key Findings

### Performance Metrics Comparison

| Metric Source | IoU | Dice Score |
|--------------|-----|------------|
| **README Claims** | 85.0% | 91.9% |
| **Training Logs (val_accu)** | 59.59% | N/A |
| **Re-evaluated (Corrected)** | **84.26% ± 5.55%** | **90.15% ± 5.90%** |

The re-evaluation confirms that the README values are accurate. The discrepancy with training logs was due to:
1. Models were trained before the corrected metric calculations were implemented
2. The `val_accu` in training logs used a different calculation method

## Detailed Results: utooth_10f_v3

### Individual Fold Performance

| Fold | Checkpoint | IoU | Dice | Binary IoU | Val Loss |
|------|------------|-----|------|------------|----------|
| 0 | epoch=56 | 69.83% | 74.56% | 69.84% | 0.2571 |
| 1 | epoch=48 | 87.71% | 93.44% | 87.74% | 0.0698 |
| 2 | epoch=47 | 88.28% | 93.72% | 87.00% | 0.0785 |
| 3 | epoch=60 | 78.47% | 83.81% | 81.21% | 0.1117 |
| 4 | epoch=43 | **88.70%** | **93.96%** | 87.32% | 0.0742 |
| 5 | epoch=49 | 87.04% | 93.04% | 87.19% | 0.0739 |
| 6 | epoch=57 | 86.32% | 92.65% | 86.33% | 0.0775 |
| 7 | epoch=61 | 86.15% | 92.53% | 86.19% | 0.0765 |
| 8 | epoch=49 | 86.00% | 92.46% | 86.03% | 0.0774 |
| 9 | epoch=42 | 84.12% | 91.32% | 84.04% | 0.0919 |

### Summary Statistics

- **Mean IoU**: 84.26% ± 5.55%
- **Mean Dice**: 90.15% ± 5.90%
- **Best Fold**: Fold 4
  - IoU: 88.70%
  - Dice: 93.96%
- **Most Consistent Folds**: Folds 1, 2, 4, 5 (all above 87% IoU)

### Observations

1. **High Performance**: 8 out of 10 folds achieved IoU > 84%, demonstrating robust performance
2. **Outliers**: Fold 0 (69.83%) and Fold 3 (78.47%) performed below average
3. **Correlation**: Lower validation loss generally correlates with higher IoU/Dice scores
4. **Binary vs Multi-class**: Binary IoU scores are very close to multi-class IoU, indicating good overall segmentation

## Recommendations

1. **Model Selection**: Use Fold 4 checkpoint (epoch 43) for best performance
2. **Ensemble Approach**: Consider averaging predictions from top 5 folds for more robust results
3. **Investigation**: Analyze why Folds 0 and 3 underperformed - possible data distribution issues
4. **Future Training**: Implement the corrected metrics during training for better model selection

## Technical Notes

- **Evaluation Method**: Models re-evaluated on validation sets using corrected metric implementations
- **Metrics Used**:
  - Multi-class IoU (4 tooth classes)
  - Dice Coefficient
  - Binary IoU (any tooth vs background)
- **Consistency**: Results align with README claims when using corrected calculations