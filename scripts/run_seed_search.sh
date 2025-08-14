#!/bin/bash

# Seed search runner script - based on working sweep configuration

echo "Starting seed search with 10-fold cross validation..."
echo "=================================================="

# Default settings
N_SEEDS=180
N_GPUS=3
START_SEED=42
N_FOLDS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --start)
            START_SEED="$2"
            shift 2
            ;;
        --folds)
            N_FOLDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--seeds N] [--gpus N] [--start SEED] [--folds N]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Number of seeds: $N_SEEDS"
echo "  Number of GPUs: $N_GPUS"
echo "  Starting seed: $START_SEED"
echo "  Number of folds: $N_FOLDS"
echo "  Max epochs: 75 with patience=20"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    echo ""
fi

# Run the seed search
python scripts/seed_search.py \
    --n_seeds $N_SEEDS \
    --n_gpus $N_GPUS \
    --start_seed $START_SEED \
    --n_folds $N_FOLDS \
    --data_path /home/user/utooth/DATA \
    --sweep_results outputs/sweeps/utooth_default_sweep_20250813_115210/reports/sweep_statistics.json

echo ""
echo "Seed search started. Monitor progress in the logs directory."
echo "Results will be saved to outputs/seed_search/seed_search_[timestamp]/"