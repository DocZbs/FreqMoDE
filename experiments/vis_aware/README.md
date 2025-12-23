# Vision-Aware Flow Matching Experiment

## Overview

This experiment implements a vision-aware flow matching approach where the model processes N+1 tokens during denoising:
- **N action tokens** (e.g., 10 action tokens for CALVIN tasks)
- **1 vision cls token** (extracted from vision foundation model)

## Key Idea

During the denoising process, we concat a vision cls token alongside action tokens. The ground truth for this vision token comes from the vision encoder's cls token (feature before global pooling).

This design allows the action denoising process to constantly "see" and attend to visual features, potentially learning better action representations through implicit visual guidance.

## Loss Function

We compute a dual loss:
```
total_loss = action_fm_loss + λ * vis_fm_loss
```

Where:
- `action_fm_loss`: Flow matching loss for action tokens (standard)
- `vis_fm_loss`: Flow matching loss for vision cls token
- `λ`: Vision loss weight (default: 0.5)

## Architecture

### Vision Encoder with Cls Token
- Modified FiLMResNet50/34 that returns both:
  - Pooled features (for state representation)
  - Cls token (mean of features before global pooling)
- Cls token dimension: 2048 for ResNet50, 512 for ResNet34
- Combined cls token from both cameras: 4096 (or 1024 for ResNet34)

### MoDeDiT Extension
- `MoDeDiTVisAware`: Extended MoDeDiT that handles N+1 tokens
- Additional vision token embedding layer
- Additional vision token output head
- Proper position embedding handling for the extra token

### Agent
- `VisAwareFlowMatchingAgent`: Extended FlowMatchingAgent
- Dual loss computation in training
- Vision-aware generation (vision token denoised alongside actions during inference)

## File Structure

```
vis_aware/
├── models/
│   ├── __init__.py
│   ├── vis_aware_agent.py          # Main agent with dual loss
│   ├── vision_encoder_with_cls.py  # ResNet with cls token extraction
│   ├── network/
│   │   ├── __init__.py
│   │   └── modedit_visaware.py     # Extended MoDeDiT for N+1 tokens
│   ├── flow_core/                  # Reuses flow_matching implementation
│   └── flow_wrappers/              # Reuses flow_matching wrapper
├── configs/
│   └── config_vis_aware_calvin.yaml
├── training/
│   └── train_vis_aware_calvin.py
├── train_vis_aware.sh
├── idea.md
└── README.md
```

## Usage

### Training

```bash
cd experiments/vis_aware
./train_vis_aware.sh
```

Or directly:

```bash
python training/train_vis_aware_calvin.py \
    flow_type=conditional \
    sampling_method=euler \
    vis_loss_weight=0.5 \
    seed=42 \
    trainer.devices=2
```

### Configuration Options

Key parameters in `config_vis_aware_calvin.yaml`:

- `vis_loss_weight`: Weight for vision cls token loss (default: 0.5)
- `cls_token_dim`: Dimension of cls token (2048 for ResNet50, 512 for ResNet34)
- `flow_type`: "conditional" or "rectified"
- `sampling_method`: "euler", "heun", "rk4", or "midpoint"
- `sigma_min`, `sigma_max`: Noise level range
- `noise_schedule`: "exponential", "linear", or "constant"

### Model Architecture

```python
# During training:
# 1. Extract cls token from vision encoder
rgb_static, rgb_gripper → ResNet50 → pooled_features, cls_token

# 2. Denoise N+1 tokens
actions_noisy (N tokens) + vis_token_noisy (1 token) → MoDeDiTVisAware →
    action_pred (N tokens) + vis_token_pred (1 token)

# 3. Compute dual loss
action_fm_loss = MSE(action_pred, action_gt)
vis_fm_loss = MSE(vis_token_pred, cls_token_gt)
total_loss = action_fm_loss + λ * vis_fm_loss
```

## Expected Benefits

1. **Implicit Visual Guidance**: Actions can attend to visual features during denoising
2. **Richer Representations**: The model learns to denoise both action and visual information
3. **Better Generalization**: Visual grounding may help with novel scenes/objects
4. **N+1 Paradigm**: Extensible to other auxiliary tokens beyond vision

## Experiments

### Baseline Comparison
Compare with standard flow matching (without vision token):
- Same architecture, hyperparameters
- Only difference: presence/absence of vision cls token

### Ablation Studies
1. Vision loss weight: λ ∈ {0.1, 0.5, 1.0, 2.0}
2. Cls token dimension: {512, 1024, 2048, 4096}
3. With/without cls token from different cameras

## Implementation Details

### Vision Cls Token Extraction
```python
# In FiLMResNet50WithCls
cls_token = x.mean(dim=[2, 3])  # Average pool before global pool
cls_token = self.cls_token_proj(cls_token)  # Project to desired dim
```

### Dual Loss Computation
```python
# Sample noise for both action and vision
x_t_action, u_t_action = flow_matcher(x0_action, x1_action, t)
x_t_vis, u_t_vis = flow_matcher(x0_vis, x1_vis, t)

# Predict velocities
velocity_action, velocity_vis = model(state, x_t_action, goal, t, vis_token=x_t_vis)

# Compute losses
action_loss = MSE(velocity_action, u_t_action)
vis_loss = MSE(velocity_vis, u_t_vis)
total_loss = action_loss + λ * vis_loss
```

## Notes

- The N+1 paradigm is general: N action tokens + 1 auxiliary token
- Current implementation: auxiliary token = vision cls token
- Can be extended to other auxiliary signals (e.g., proprioception, language embeddings)
- During inference, vision token is denoised but only actions are used

## Citation

If you use this code, please cite the original MoDE and Flow Matching papers.

## License

Same as parent MoDE Diffusion Policy project.
