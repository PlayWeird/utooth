#!/bin/bash
# Activation script for uTooth environment
# Usage: source activate_env.sh

# Activate the virtual environment
source utooth_env/bin/activate

# Verify installation
echo "🚀 uTooth Environment Activated!"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Environment setup complete. You can now run:"
echo "  python scripts/sweep_runner.py --help"
echo "  python scripts/train.py --help"