#!/bin/bash
# Parallel evaluation script for Speed-Aware Flow Matching on CALVIN

set -e

PROJECT_ROOT="/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy"
cd "$PROJECT_ROOT"

# Default values
CHECKPOINT_PATH="${1:-/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/speed_aware/logs/20251219/222645_speed_aware_rectified_euler/seed_42/checkpoints/speed_aware_epoch_epoch17_step_step036238.ckpt}"
NUM_SEQUENCES="${2:-1000}"
NUM_GPUS="${3:-2}"

echo "================================================"
echo "Speed-Aware Flow Matching Parallel Evaluation"
echo "================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Sequences: $NUM_SEQUENCES"
echo "GPUs: $NUM_GPUS"
echo "================================================"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Run evaluation
python experiments/speed_aware/eval/evaluate_calvin_parallel.py \
    --config-path="../configs" \
    --config-name="config_speed_aware_calvin" \
    checkpoint_path="$CHECKPOINT_PATH" \
    num_sequences=$NUM_SEQUENCES \
    num_gpus=$NUM_GPUS

echo "================================================"
echo "Evaluation complete!"
echo "================================================"
