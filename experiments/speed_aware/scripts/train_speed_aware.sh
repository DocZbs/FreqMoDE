#!/bin/bash
# Training script for Speed-Aware Flow Matching on CALVIN
# Independent from freq_flow experiments

set -e


# Project root
PROJECT_ROOT="/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy"
cd "$PROJECT_ROOT"

# Experiment directory
EXP_DIR="experiments/speed_aware"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

echo "================================================"
echo "Speed-Aware Flow Matching Training"
echo "================================================"
echo "Project root: $PROJECT_ROOT"
echo "Experiment dir: $EXP_DIR"
echo "GPUs: $NUM_GPUS (devices: $CUDA_VISIBLE_DEVICES)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "================================================"

# Run training
python experiments/flow_matching/training/train_flow_calvin.py \
    --config-path="../../speed_aware/configs" \
    --config-name="config_speed_aware_calvin" \
    trainer.devices=$NUM_GPUS \
    trainer.strategy=ddp \
    trainer.precision=32

echo "================================================"
echo "Training complete!"
echo "Logs saved to: $EXP_DIR/logs"
echo "================================================"
