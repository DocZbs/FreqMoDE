"""
Test to check if frequency losses have reasonable scale.
"""
import torch
import sys
sys.path.insert(0, '/mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy')

from experiments.freq_flow.models.freq_utils import dct_differentiable, create_band_masks
from experiments.freq_flow.models.freq_losses import compute_freq_losses

# Simulate some data
B, T, da = 4, 10, 7
device = 'cuda'

# Random actions (typical range for normalized actions)
target_actions = torch.randn(B, T, da, device=device) * 0.5
pred_actions = target_actions + torch.randn(B, T, da, device=device) * 0.1

# Simulate expert outputs
num_experts = 4
expert_freq_outputs = []
for i in range(num_experts):
    expert_out = target_actions + torch.randn(B, T, da, device=device) * 0.15
    expert_freq = dct_differentiable(expert_out)
    expert_freq_outputs.append(expert_freq)

expert_freq_outputs = torch.stack(expert_freq_outputs, dim=0)  # (4, B, T, da)

# Random expert weights (properly normalized)
expert_weights = torch.rand(B, num_experts, device=device)
expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

# Create band masks
band_masks = create_band_masks(
    num_experts=num_experts,
    seq_len=T,
    action_dim=da,
    band_division=[0.25, 0.5, 0.75, 1.0],
    device=device
)

# Test with current weights
loss_weights_current = {
    'subband': 0.1,
    'whole': 0.1,
    'balance': 0.01,
    'smoothness': 0.0
}

losses_current = compute_freq_losses(
    pred_actions=pred_actions,
    target_actions=target_actions,
    expert_freq_outputs=expert_freq_outputs,
    expert_weights=expert_weights,
    band_masks=band_masks,
    loss_weights=loss_weights_current
)

print("=" * 60)
print("Loss Scale Test with Current Weights (0.1, 0.1, 0.01)")
print("=" * 60)
print(f"Subband loss (weighted):   {losses_current['subband'].item():.6f}")
print(f"Whole loss (weighted):     {losses_current['whole'].item():.6f}")
print(f"Balance loss (weighted):   {losses_current['balance'].item():.6f}")
print(f"Smoothness loss (weighted): {losses_current['smoothness'].item():.6f}")
print(f"Total freq losses:          {(losses_current['subband'] + losses_current['whole'] + losses_current['balance'] + losses_current['smoothness']).item():.6f}")

# Simulate typical flow matching loss
fm_loss_typical = torch.nn.functional.mse_loss(
    torch.randn(B, T, da, device=device) * 0.5,
    torch.randn(B, T, da, device=device) * 0.5
)
print(f"\nTypical FM loss (for comparison): {fm_loss_typical.item():.6f}")
print(f"Total loss would be: {(fm_loss_typical + losses_current['subband'] + losses_current['whole'] + losses_current['balance']).item():.6f}")

# Check unweighted losses
loss_weights_unweighted = {
    'subband': 1.0,
    'whole': 1.0,
    'balance': 1.0,
    'smoothness': 0.0
}

losses_unweighted = compute_freq_losses(
    pred_actions=pred_actions,
    target_actions=target_actions,
    expert_freq_outputs=expert_freq_outputs,
    expert_weights=expert_weights,
    band_masks=band_masks,
    loss_weights=loss_weights_unweighted
)

print("\n" + "=" * 60)
print("Raw Loss Scale (unweighted, for reference)")
print("=" * 60)
print(f"Raw subband loss:   {losses_unweighted['subband'].item():.6f}")
print(f"Raw whole loss:     {losses_unweighted['whole'].item():.6f}")
print(f"Raw balance loss:   {losses_unweighted['balance'].item():.6f}")

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)
ratio = losses_current['whole'].item() / (fm_loss_typical.item() + 1e-8)
print(f"Frequency loss / FM loss ratio: {ratio:.4f}")
if ratio > 0.5:
    print("WARNING: Frequency losses are too large relative to FM loss!")
    print("This could slow down or prevent convergence.")
    print("Recommendation: Reduce frequency loss weights further.")
elif ratio < 0.05:
    print("Frequency losses are very small relative to FM loss.")
    print("They may not provide enough guidance.")
else:
    print("Frequency loss scale looks reasonable.")
