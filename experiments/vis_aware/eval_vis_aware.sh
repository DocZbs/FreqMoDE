#!/bin/bash

# Vision-Aware Flow Matching Evaluation Script for CALVIN
# Multi-GPU parallel evaluation

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/../.."

# Checkpoint path - modify this to your trained model checkpoint
CHECKPOINT_PATH="/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/vis_aware/logs/20251222/105514_visaware_rectified_euler/seed_42/checkpoints/vis_aware_epoch_epoch20_step_step042000.ckpt"
NUM_SEQUENCES=1000
NUM_GPUS=2

echo "Starting Vision-Aware Flow Matching Evaluation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Number of sequences: $NUM_SEQUENCES"
echo "Number of GPUs: $NUM_GPUS"
echo "=============================================="

# Run evaluation
cd eval
python evaluate_calvin_parallel.py \
    checkpoint_path=$CHECKPOINT_PATH \
    ++num_sequences=$NUM_SEQUENCES \
    +num_gpus=$NUM_GPUS

echo "Evaluation completed!"
