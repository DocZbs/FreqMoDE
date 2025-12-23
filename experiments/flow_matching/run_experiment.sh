#!/bin/bash
# Flow Matching 快速启动脚本

set -e

echo "========================================"
echo "Flow Matching Training"
echo "========================================"

# 默认参数
FLOW_TYPE=${FLOW_TYPE:-"conditional"}
SAMPLING_METHOD=${SAMPLING_METHOD:-"euler"}
NUM_GPUS=${NUM_GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-42}

MODE=${1:-"train"}

if [ "$MODE" == "test" ]; then
    echo "Running tests..."
    python experiments/flow_matching/test_flow_matching.py
elif [ "$MODE" == "train" ]; then
    echo "Starting training..."
    echo "  Flow Type: $FLOW_TYPE"
    echo "  Sampling Method: $SAMPLING_METHOD"
    echo "  GPUs: $NUM_GPUS"
    echo "  Batch Size: $BATCH_SIZE"
    echo ""

    python experiments/flow_matching/training/train_flow_calvin.py \
        flow_type=$FLOW_TYPE \
        sampling_method=$SAMPLING_METHOD \
        trainer.devices=$NUM_GPUS \
        batch_size=$BATCH_SIZE \
        seed=$SEED
else
    echo "Usage: $0 [train|test]"
    exit 1
fi
