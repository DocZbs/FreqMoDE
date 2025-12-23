#!/bin/bash

# Vision-Aware Flow Matching Training Script for CALVIN
# This script trains the vision-aware model with dual loss (action + vision cls token)

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/../.."

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1

# Training parameters
FLOW_TYPE="rectified"  # or "rectified"
SAMPLING_METHOD="euler"  # or "heun", "rk4", "midpoint"
SEED=42
DEVICES=2

echo "Starting Vision-Aware Flow Matching Training"
echo "=============================================="
echo "Flow Type: $FLOW_TYPE"
echo "Sampling Method: $SAMPLING_METHOD"
echo "Seed: $SEED"
echo "Devices: $DEVICES"
echo "Note: Vision token used as conditioning (not denoised)"
echo "=============================================="

# Run training
python training/train_vis_aware_calvin.py \
    flow_type=$FLOW_TYPE \
    sampling_method=$SAMPLING_METHOD \
    seed=$SEED \
    trainer.devices=$DEVICES

echo "Training completed!"
