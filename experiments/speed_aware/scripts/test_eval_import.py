#!/usr/bin/env python
"""
Quick test to verify evaluation scripts can be imported successfully
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

print("=" * 60)
print("Testing Speed-Aware Evaluation Import")
print("=" * 60)

# Test 1: Import required modules
print("\n1. Testing imports...")
try:
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    import experiments.speed_aware.models.speed_aware_agent as speed_aware_models
    from mode.evaluation.utils import LangEmbeddings
    from mode.evaluation.multistep_sequences import get_sequences
    print("   All imports successful")
except Exception as e:
    print(f"   Import failed: {e}")
    sys.exit(1)

# Test 2: Check config exists
print("\n2. Checking config file...")
try:
    config_path = Path(__file__).parent.parent / "configs" / "config_speed_aware_calvin.yaml"
    if config_path.exists():
        config = OmegaConf.load(config_path)
        print(f"   Config loaded successfully")
        print(f"   - Flow type: {config.flow_type}")
        print(f"   - Model: {config.model._target_}")
    else:
        print(f"   Config not found at {config_path}")
        sys.exit(1)
except Exception as e:
    print(f"   Config loading failed: {e}")
    sys.exit(1)

# Test 3: Check sequence generation
print("\n3. Testing sequence generation...")
try:
    sequences = get_sequences(10)
    print(f"   Generated {len(sequences)} sequences")
    print(f"   Sample sequence: {sequences[0][:3]}...")
except Exception as e:
    print(f"   Sequence generation failed: {e}")
    sys.exit(1)

# Test 4: Check CUDA availability
print("\n4. Checking GPU availability...")
num_gpus = torch.cuda.device_count()
print(f"   Available GPUs: {num_gpus}")
if num_gpus > 0:
    print(f"   GPU 0: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 60)
print("All checks passed!")
print("=" * 60)
print("\nYou can now run evaluation with:")
print("  Single GPU: ./run_eval_single.sh <checkpoint> <num_seq> <gpu_id>")
print("  Multi GPU:  ./run_eval_parallel.sh <checkpoint> <num_seq> <num_gpus>")
print("=" * 60)
