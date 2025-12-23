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
NOISE_SCHEDULE=${NOISE_SCHEDULE:-"exponential"}

MODE=${1:-"train"}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "Project root: $PROJECT_ROOT"

if [ "$MODE" == "test" ]; then
    echo "Running tests..."
    cd "$PROJECT_ROOT"
    python experiments/flow_matching/test_flow_matching.py

elif [ "$MODE" == "train" ]; then
    echo "Starting training..."
    echo "  Flow Type: $FLOW_TYPE"
    echo "  Noise Schedule: $NOISE_SCHEDULE"
    echo "  Sampling Method: $SAMPLING_METHOD"
    echo "  GPUs: $NUM_GPUS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Seed: $SEED"
    echo ""

    cd "$PROJECT_ROOT"
    python experiments/flow_matching/training/train_flow_calvin.py \
        flow_type=$FLOW_TYPE \
        sampling_method=$SAMPLING_METHOD \
        trainer.devices=$NUM_GPUS \
        batch_size=$BATCH_SIZE \
        seed=$SEED \
        model.noise_schedule=$NOISE_SCHEDULE \
        model.sigma_min=0.001 \
        model.sigma_max=80.0

elif [ "$MODE" == "eval" ]; then
    echo "Running evaluation..."
    CHECKPOINT=${CHECKPOINT:-""}
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: Please set CHECKPOINT environment variable"
        echo "Example: CHECKPOINT=path/to/checkpoint.ckpt $0 eval"
        exit 1
    fi

    cd "$PROJECT_ROOT"
    python experiments/flow_matching/evaluate_calvin.py \
        checkpoint_path=$CHECKPOINT \
        num_sequences=1000

else
    echo "Usage: $0 [train|test|eval]"
    echo ""
    echo "Examples:"
    echo "  $0 train                           # Train with defaults"
    echo "  FLOW_TYPE=rectified $0 train       # Train with Rectified Flow"
    echo "  NUM_GPUS=4 BATCH_SIZE=128 $0 train # Train with 4 GPUs"
    echo "  CHECKPOINT=path/to/ckpt $0 eval    # Evaluate checkpoint"
    exit 1
fi

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
