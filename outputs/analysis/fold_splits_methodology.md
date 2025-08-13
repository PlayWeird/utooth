# Determining Fold Splits for uTooth Training Runs

## How Fold Splits are Determined

### Prerequisites
To recreate the exact fold splits used in any training run, you need:

1. **Random Seed**: Stored in `outputs/runs/{run_name}/config.json`
2. **Number of Folds**: Also in config.json
3. **Data Path**: Path to the dataset
4. **Same Data Ordering**: The cases must be in the same order as during training

### Method
The fold splits are generated using scikit-learn's `KFold` with:
- **Shuffle**: `True`
- **Random State**: From the training config
- **Consistent Sorting**: Cases sorted alphabetically by folder name

### Example: utooth_10f_v3

**Configuration:**
- Random Seed: `2025`
- Folds: `10`
- Total Cases: `48`

**Best Performing Fold (Fold 4):**
- **Performance**: 88.7% IoU, 94.0% Dice
- **Validation Cases** (5):
  - case-105584
  - case-121537
  - case-127337
  - case-151481
  - case-160558
- **Training Cases**: 43 (remaining cases)

## Tools Available

### 1. `determine_fold_splits.py`
```bash
# Show fold splits for any run
python scripts/determine_fold_splits.py utooth_10f_v3

# Save splits to JSON file
python scripts/determine_fold_splits.py utooth_10f_v3 --save
```

### 2. Fold Splits File
Automatically saved to: `outputs/runs/{run_name}/fold_splits.json`

Contains:
- Complete train/validation splits for each fold
- Case names and indices
- Reproducible for future reference

## Reproducibility

The fold splits are **100% reproducible** because:
1. Fixed random seed stored in config
2. Deterministic data ordering (alphabetical)
3. Same KFold parameters used

## Use Cases

### Model Analysis
- Identify which cases are in best/worst performing folds
- Analyze data distribution across folds
- Debug fold-specific performance issues

### Ensemble Creation
- Combine predictions from specific high-performing folds
- Weight models based on validation performance

### Data Analysis
- Check for data leakage between train/validation
- Analyze case difficulty patterns
- Validate cross-validation strategy

## All Training Runs

The methodology works for any completed training run:
- utooth_10f_v1 (seed: 42)
- utooth_10f_v2 (seed: 42) 
- utooth_10f_v3 (seed: 2025)
- utooth_10f_v4 (seed: 42)
- utooth_5f_v1 (seed: 42)