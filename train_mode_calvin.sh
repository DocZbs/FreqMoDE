#!/bin/bash
# MoDE Diffusion Policy - CALVIN Training Script
# Original training script for MoDE model

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "Training MoDE Diffusion Policy on CALVIN"
echo "========================================"
echo "Working directory: $(pwd)"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Check if dataset exists
if [ ! -d "dataset/task_D_D" ]; then
    echo "ERROR: Dataset not found at dataset/task_D_D"
    echo "Please download CALVIN dataset first:"
    echo "  cd dataset && sh download_data.sh D"
    exit 1
fi

# Default training with configuration from config_calvin.yaml
# You can override any parameter by adding it as command line argument
# Example: bash train_mode_calvin.sh trainer.devices=4 batch_size=128

echo "Starting training with Hydra config..."
echo "Config file: conf/config_calvin.yaml"
echo ""

python mode/training_calvin.py "$@"

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
