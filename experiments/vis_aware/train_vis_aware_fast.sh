#!/bin/bash

# Train VisAware Flow Matching Agent with PRECOMPUTED CLIP features
# This is 3-5x faster than the standard version!
#
# Prerequisites:
#   1. Run ./run_precompute_clip.sh first to generate CLIP features
#   2. Make sure clip_features directory exists

set -e

cd /mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/vis_aware

# Check if CLIP features exist
CLIP_FEATURES_DIR="/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/dataset/task_D_D/clip_features"
if [ ! -d "$CLIP_FEATURES_DIR/training" ]; then
    echo "ERROR: CLIP features not found at $CLIP_FEATURES_DIR"
    echo "Please run ./run_precompute_clip.sh first!"
    exit 1
fi

echo "=========================================="
echo "Training VisAware Agent (FAST MODE)"
echo "Using precomputed CLIP features"
echo "=========================================="

python training/train_vis_aware_calvin.py \
    --config-name config_vis_aware_calvin_fast \
    trainer.devices=1 \
    trainer.precision=32 \
    datamodule.batch_size.train=16 \
    datamodule.batch_size.val=16 \
    model.use_precomputed_clip=true \
    model.flow_type=conditional \
    model.sampling_method=euler \
    task_name=vis_aware_calvin_fast
