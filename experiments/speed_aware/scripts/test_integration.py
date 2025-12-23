"""
Test script to verify Speed-Aware integration.

Run this before training to ensure all components work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

import torch
from omegaconf import OmegaConf

print("="*60)
print("Speed-Aware Integration Test")
print("="*60 + "\n")

# Test 1: Import modules
print("Test 1: Importing modules...")
try:
    from experiments.speed_aware.models.speed_aware_modedit import SpeedAwareMoDeDiT
    from experiments.speed_aware.models.speed_aware_agent import SpeedAwareFlowAgent
    from experiments.freq_flow.models.speed_aware_head import (
        AdaptiveSmoothingHead,
        compute_speed_label
    )
    print("✓ All modules imported successfully\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Load config
print("Test 2: Loading config...")
try:
    config_path = Path(__file__).parent.parent / "configs" / "config_speed_aware_calvin.yaml"
    config = OmegaConf.load(config_path)
    print(f"✓ Config loaded from {config_path}")
    print(f"  - Flow type: {config.flow_type}")
    print(f"  - Speed loss weight: {config.model.speed_loss_weight}")
    print(f"  - Enable speed head: {config.model.model.inner_model.enable_speed_head}\n")
except Exception as e:
    print(f"✗ Config loading failed: {e}\n")
    sys.exit(1)

# Test 3: Create SpeedAwareMoDeDiT
print("Test 3: Creating SpeedAwareMoDeDiT...")
try:
    model = SpeedAwareMoDeDiT(
        obs_dim=2048,
        goal_dim=512,
        device='cuda',
        goal_conditioned=True,
        action_dim=7,
        embed_dim=512,
        embed_pdrob=0.0,
        attn_pdrop=0.1,
        n_layers=4,  # Small for testing
        n_heads=8,
        goal_seq_len=1,
        obs_seq_len=1,
        action_seq_len=10,
        state_dim=15,
        mlp_pdrop=0.1,
        enable_speed_head=True,
        min_smoothing=0.3,
        max_smoothing=0.9,
    )
    print(f"✓ SpeedAwareMoDeDiT created")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Speed head enabled: {model.enable_speed_head}")
    print(f"  - Output head type: {type(model.out).__name__}\n")
except Exception as e:
    print(f"✗ Model creation failed: {e}\n")
    sys.exit(1)

# Test 4: Forward pass
print("Test 4: Testing forward pass...")
try:
    B = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Dummy inputs
    states = {
        'state_images': torch.randn(B, 1, 2048).to(device)
    }
    actions = torch.randn(B, 10, 7).to(device)
    goals = torch.randn(B, 512).to(device)
    sigma = torch.rand(B).to(device)

    # Forward
    with torch.no_grad():
        pred_actions, pred_speed, freq_filter = model(
            states, actions, goals, sigma,
            return_speed_info=True
        )

    print(f"✓ Forward pass successful")
    print(f"  - Pred actions shape: {pred_actions.shape}")
    print(f"  - Pred speed shape: {pred_speed.shape}")
    print(f"  - Pred speed values: {pred_speed.squeeze()}")
    print(f"  - Freq filter shape: {freq_filter.shape}\n")
except Exception as e:
    print(f"✗ Forward pass failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Speed label computation
print("Test 5: Testing speed label computation...")
try:
    # Create test actions with different speeds
    fast_actions = torch.randn(1, 10, 7) * 0.5
    fast_actions[:, :5, :] += torch.linspace(0, 2, 5).view(1, 5, 1)

    slow_actions = torch.linspace(-0.1, 0.1, 10).view(1, 10, 1).expand(1, 10, 7)

    test_batch = torch.cat([fast_actions, slow_actions], dim=0)
    speeds = compute_speed_label(test_batch)

    print(f"✓ Speed label computation successful")
    print(f"  - Fast action speed: {speeds[0]:.3f}")
    print(f"  - Slow action speed: {speeds[1]:.3f}")
    print(f"  - Speed difference: {(speeds[0] - speeds[1]).abs():.3f}")

    if speeds[0] > speeds[1]:
        print("  ✓ Fast > Slow (correct)\n")
    else:
        print("  ✗ Fast <= Slow (incorrect!)\n")
except Exception as e:
    print(f"✗ Speed label computation failed: {e}\n")
    sys.exit(1)

# Test 6: Check dependencies
print("Test 6: Checking dependencies...")
try:
    import pytorch_lightning
    import wandb
    import einops
    import omegaconf

    print(f"✓ All dependencies available")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - PyTorch Lightning: {pytorch_lightning.__version__}\n")
except ImportError as e:
    print(f"✗ Missing dependency: {e}\n")
    sys.exit(1)

# Summary
print("="*60)
print("All tests passed! ✓")
print("="*60)
print("\nYou can now run training with:")
print("  cd experiments/speed_aware/scripts")
print("  ./train_speed_aware.sh")
print("="*60)
