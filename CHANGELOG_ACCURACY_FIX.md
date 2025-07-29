# uTooth Accuracy Fix Changelog

## Summary
Fixed accuracy calculation bug that was reporting 58% accuracy when models actually achieved 74-85%. Implemented proper multi-class IoU, Dice coefficient, and binary IoU metrics.

## Files Created/Modified

### Core Implementation
- `src/models/accuracy_metrics.py` - **NEW**: Corrected accuracy calculation functions
- `src/models/unet.py` - **MODIFIED**: Updated validation_step to use corrected metrics

### Analysis Scripts (NEW)
- `scripts/analysis/analyze_existing_runs.py` - Re-evaluate existing runs with corrected metrics
- `scripts/analysis/find_best_results.py` - Identify best performing models and folds
- `scripts/analysis/analyze_fold_performance.py` - Analyze fold performance patterns
- `scripts/analysis/analyze_fold_cases.py` - Identify difficult/easy cases by fold

### Visualization Scripts (NEW)
- `scripts/visualization/visualize_predictions.py` - Detailed model prediction visualizations
- `scripts/visualization/visualize_run.py` - Training run visualizations (moved from scripts/)

### Testing Scripts (NEW)
- `scripts/testing/test_corrected_accuracy.py` - Test corrected metrics on checkpoints
- `scripts/testing/test_accuracy_calculation.py` - Debug accuracy calculation methods

### Analysis Results (NEW)
- `analysis/corrected_metrics_analysis.json` - Detailed corrected metrics for all experiments
- `analysis/corrected_metrics_analysis_summary.csv` - Summary table by experiment
- `analysis/corrected_metrics_analysis_folds.csv` - Per-fold detailed results
- `analysis/fold_performance_analysis.png` - Fold performance visualization
- `analysis/README.md` - Analysis documentation

### Documentation (NEW/MODIFIED)
- `scripts/README.md` - **NEW**: Comprehensive scripts documentation
- `README.md` - **MODIFIED**: Updated with corrected results and new structure
- `CHANGELOG_ACCURACY_FIX.md` - **NEW**: This file

## Key Findings

### Best Single Model
- **Experiment**: utooth_10f_v3, Fold 7
- **Checkpoint**: `outputs/runs/utooth_10f_v3/checkpoints/fold_7/utooth-epoch=61-val_loss=0.0844.ckpt`
- **Corrected IoU**: 85.0%
- **Dice Score**: 91.9%
- **Binary IoU**: 85.3%

### Average Performance by Experiment
1. utooth_10f_v3: 74.2% ± 8.7% IoU
2. utooth_10f_v4: 74.2% ± 7.0% IoU  
3. utooth_10f_v2: 73.7% ± 7.0% IoU
4. utooth_10f_v1: 73.3% ± 11.2% IoU

### Best Performing Folds (across experiments)
1. Fold 6: 80.4% average IoU (most consistent)
2. Fold 8: 79.3% average IoU
3. Fold 1: 77.2% average IoU
4. Fold 7: 75.6% average IoU (contains best single model)

## Directory Structure Changes

### Before
```
scripts/
├── train.py
├── monitor_training.py
├── run_training.sh
├── visualize_run.py
└── (other mixed scripts)
```

### After
```
scripts/
├── analysis/          # Analysis and evaluation scripts
├── visualization/     # Visualization scripts  
├── testing/          # Testing and debugging scripts
├── train.py          # Main training script
├── monitor_training.py
├── run_training.sh
└── README.md         # Scripts documentation

analysis/             # NEW: Analysis results directory
├── *.json, *.csv    # Analysis results
├── *.png            # Visualizations
└── README.md        # Analysis documentation
```

## Usage for Future Work

### New Training (with corrected metrics)
```bash
python scripts/train.py --experiment_name corrected_metrics_v1
```

### Analyze Existing Results
```bash
python scripts/analysis/analyze_existing_runs.py
python scripts/analysis/find_best_results.py
```

### Visualize Best Model
```bash
python scripts/visualization/visualize_predictions.py \
    --experiment utooth_10f_v3 --fold 7 --num_samples 10
```

## Impact
- Models are performing much better than initially thought (74-85% vs 58%)
- Proper metrics now align with visual inspection results
- Best model (91.9% Dice score) exceeds original 90% target
- All future training will use corrected metrics automatically