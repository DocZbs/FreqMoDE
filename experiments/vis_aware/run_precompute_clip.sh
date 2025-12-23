#!/bin/bash

# Precompute CLIP features for CALVIN dataset
# This should be run ONCE before training with the fast agent

set -e

# Configuration
DATA_ROOT="/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/dataset/task_D_D"
OUTPUT_DIR="/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/dataset/task_D_D/clip_features"
CLIP_MODEL="ViT-B/32"
BATCH_SIZE=64
DEVICE="cuda"

echo "=========================================="
echo "Precomputing CLIP features for CALVIN"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "CLIP model: $CLIP_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="
echo ""

cd /mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/vis_aware

python precompute_clip_features.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --clip_model_name "$CLIP_MODEL" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "Precomputation completed successfully!"
echo "You can now train with the fast agent using:"
echo "  ./train_vis_aware_fast.sh"
echo "=========================================="
