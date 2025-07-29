# uTooth Analysis Results

This directory contains the analysis results for corrected accuracy metrics applied to existing training runs.

## Overview

The original training runs reported artificially low accuracy (~58%) due to incorrect metric calculation. This analysis re-evaluates all experiments with corrected metrics, revealing the models actually achieved 74-85% IoU scores.

## Files

### Main Analysis Results
- `corrected_metrics_analysis.json` - Detailed corrected metrics for all experiments and folds
- `corrected_metrics_analysis_summary.csv` - Summary table of average performance by experiment
- `corrected_metrics_analysis_folds.csv` - Per-fold detailed results across all experiments

### Visualizations
- `fold_performance_analysis.png` - Charts showing fold performance patterns

## Key Findings

### Best Experiments (by average corrected IoU):
1. **utooth_10f_v3**: 74.2% ± 8.7%
2. **utooth_10f_v4**: 74.2% ± 7.0%
3. **utooth_10f_v2**: 73.7% ± 7.0%
4. **utooth_10f_v1**: 73.3% ± 11.2%

### Single Best Model:
- **Experiment**: utooth_10f_v3, Fold 7
- **Checkpoint**: `outputs/runs/utooth_10f_v3/checkpoints/fold_7/utooth-epoch=61-val_loss=0.0844.ckpt`
- **Corrected IoU**: 85.0%
- **Dice Score**: 91.9%
- **Binary IoU**: 85.3%

### Best Performing Folds (by average across experiments):
1. **Fold 6**: 80.4% IoU (most consistent)
2. **Fold 8**: 79.3% IoU
3. **Fold 1**: 77.2% IoU
4. **Fold 7**: 75.6% IoU (contains single best model)

### Worst Performing Folds:
1. **Fold 0**: 65.9% IoU (most variable)
2. **Fold 3**: 69.4% IoU
3. **Fold 2**: 70.0% IoU

## Difficult Cases Identified

Cases that consistently appear in worst-performing folds:
- case-109619, case-113059, case-146242, case-163545
- case-109833, case-138712, case-144615, case-145435
- case-173494, case-174235, case-178321, case-179082
- case-180398, case-182948, case-118091

## Metrics Explanation

- **Corrected IoU**: Properly calculated Intersection over Union for multi-class segmentation
- **Dice Score**: Sørensen-Dice coefficient, often preferred in medical imaging
- **Binary IoU**: IoU for tooth vs background (simplified binary task)
- **Old Accuracy**: The incorrectly calculated metric from original training

## Usage

To reproduce these results or analyze new experiments:

```bash
# Analyze specific experiments
python scripts/analysis/analyze_existing_runs.py --experiments utooth_10f_v3 utooth_10f_v4

# Find best performing models
python scripts/analysis/find_best_results.py

# Analyze fold performance patterns  
python scripts/analysis/analyze_fold_performance.py

# Analyze which cases are in best/worst folds
python scripts/analysis/analyze_fold_cases.py
```

## Recommendations

1. **Use Fold 6 or 8** for most reliable performance
2. **Use the best checkpoint** from utooth_10f_v3 fold 7 for peak performance
3. **Consider ensembling** the best folds (6, 8, 1, 7) for production
4. **Investigate difficult cases** for potential data quality issues
5. **Use corrected metrics** for future training runs

The model performance is actually excellent - the issue was just with metric calculation!