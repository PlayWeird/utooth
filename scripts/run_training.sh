#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
source utooth_env/bin/activate

# Run training with test mode (2 epochs)
echo "Running uTooth training with 5-fold cross validation (test mode)..."
python scripts/train.py --test_run --use_wandb

# For full training, use:
# python scripts/train.py --max_epochs 50 --use_wandb