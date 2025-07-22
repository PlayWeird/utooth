#!/bin/bash

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d "utooth_env" ]; then
    echo "Virtual environment not found. Creating one..."
    python -m venv utooth_env
    source utooth_env/bin/activate
    pip install -r requirements.txt
else
    # Activate virtual environment
    source utooth_env/bin/activate
fi

echo "======================================================"
echo "uTooth Training Script"
echo "======================================================"
echo "Virtual Environment: $(which python)"
echo "GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "======================================================"

# Parse command line arguments
TEST_RUN=false
EXPERIMENT_NAME=""
USE_WANDB=false
RESUME=false
AUTO_RESUME=false
FORCE_RESTART=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_RUN=true
            shift
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --auto-resume)
            AUTO_RESUME=true
            RESUME=true
            shift
            ;;
        --force-restart)
            FORCE_RESTART=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--experiment-name NAME] [--wandb] [--resume] [--auto-resume] [--force-restart]"
            echo "  --test: Run with 2 epochs for testing"
            echo "  --experiment-name: Name for this experiment"
            echo "  --wandb: Enable Weights & Biases logging"
            echo "  --resume: Resume from latest checkpoints"
            echo "  --auto-resume: Resume automatically without confirmation"
            echo "  --force-restart: Force restart even if checkpoints exist"
            exit 1
            ;;
    esac
done

# Build command
CMD="python scripts/train.py"

if [ "$TEST_RUN" = true ]; then
    CMD="$CMD --test_run"
    echo "Running in TEST MODE (2 epochs per fold)"
else
    echo "Running FULL TRAINING (50 epochs per fold)"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
    echo "Weights & Biases logging: ENABLED"
else
    echo "Weights & Biases logging: DISABLED"
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
    echo "Resume mode: ENABLED"
fi

if [ "$AUTO_RESUME" = true ]; then
    CMD="$CMD --auto_resume"
    echo "Auto-resume mode: ENABLED"
fi

if [ "$FORCE_RESTART" = true ]; then
    CMD="$CMD --force_restart"
    echo "Force restart mode: ENABLED"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment_name $EXPERIMENT_NAME"
    echo "Experiment name: $EXPERIMENT_NAME"
else
    echo "Experiment name: Auto-generated"
fi

echo "======================================================"
echo "Starting training..."
echo "Command: $CMD"
echo "======================================================"

# Run the training
eval $CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "======================================================"
    echo "Training completed successfully!"
    echo "Check the outputs/runs/ directory for results."
    echo "======================================================"
else
    echo "======================================================"
    echo "Training failed with exit code $?"
    echo "======================================================"
    exit 1
fi