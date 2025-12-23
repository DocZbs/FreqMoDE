"""
Speed-Aware MoDeDiT Model.

A completely independent implementation that extends baseline MoDeDiT
with adaptive frequency smoothing based on motion speed prediction.

This is separate from freq_flow experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

import torch
import torch.nn as nn
from typing import Optional, Tuple
from omegaconf import DictConfig

from experiments.flow_matching.models.network.modedit import MoDeDiT
from experiments.freq_flow.models.speed_aware_head import (
    AdaptiveSmoothingHead,
    SpeedPredictor
)


class SpeedAwareMoDeDiT(MoDeDiT):
    """
    MoDeDiT with Speed-Aware Adaptive Smoothing.

    Key differences from baseline MoDeDiT:
    1. Replaces linear output with AdaptiveSmoothingHead
    2. Learns to predict motion speed from context
    3. Applies adaptive frequency filtering based on speed

    This does NOT use frequency-domain expert decomposition like FreqMoDeDiT.
    """

    def __init__(
        self,
        # All base MoDeDiT parameters
        obs_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        state_dim,
        mlp_pdrop: float = 0.1,
        goal_drop: float = 0.1,
        linear_output: bool = True,
        use_proprio: bool = False,
        cond_router: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        router_normalize: bool = True,
        use_goal_in_routing: bool = False,
        use_argmax: bool = False,
        causal: bool = True,
        use_shared_expert: bool = False,
        use_noise_token_as_input: bool = True,
        use_custom_attn_mask: bool = False,
        init_style: str = 'default',
        # Speed-aware specific parameters
        enable_speed_head: bool = True,
        min_smoothing: float = 0.3,
        max_smoothing: float = 0.9,
        smoothing_sharpness: float = 3.0,
    ):
        """
        Initialize SpeedAwareMoDeDiT.

        Args:
            All standard MoDeDiT args, plus:
            enable_speed_head: If True, use AdaptiveSmoothingHead; else standard linear
            min_smoothing: Minimum smoothing for fast motions (preserve 30% of spectrum)
            max_smoothing: Maximum smoothing for slow motions (preserve 10% of spectrum)
            smoothing_sharpness: Steepness of frequency filter cutoff
        """
        # Initialize base MoDeDiT
        super().__init__(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            device=device,
            goal_conditioned=goal_conditioned,
            action_dim=action_dim,
            embed_dim=embed_dim,
            embed_pdrob=embed_pdrob,
            attn_pdrop=attn_pdrop,
            n_layers=n_layers,
            n_heads=n_heads,
            goal_seq_len=goal_seq_len,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
            state_dim=state_dim,
            mlp_pdrop=mlp_pdrop,
            goal_drop=goal_drop,
            linear_output=linear_output,
            use_proprio=use_proprio,
            cond_router=cond_router,
            num_experts=num_experts,
            top_k=top_k,
            router_normalize=router_normalize,
            use_goal_in_routing=use_goal_in_routing,
            use_argmax=use_argmax,
            causal=causal,
            use_shared_expert=use_shared_expert,
            use_noise_token_as_input=use_noise_token_as_input,
            use_custom_attn_mask=use_custom_attn_mask,
            init_style=init_style,
        )

        self.enable_speed_head = enable_speed_head

        # Replace output head if speed-aware is enabled
        if enable_speed_head:
            # Remove original output layer
            if hasattr(self, 'out'):
                del self.out

            # Create speed-aware output head
            self.out = AdaptiveSmoothingHead(
                hidden_dim=embed_dim,
                action_horizon=action_seq_len,
                action_dim=action_dim,
                dropout=mlp_pdrop,
                min_smoothing=min_smoothing,
                max_smoothing=max_smoothing,
                smoothing_sharpness=smoothing_sharpness,
                device=device
            )

        # Storage for predicted speed (used in loss computation)
        self.pred_speed = None
        self.freq_filter = None

    def forward(
        self,
        states,
        actions,
        goals,
        sigma,
        uncond: Optional[bool] = False,
        return_speed_info: bool = False,
    ):
        """
        Forward pass through SpeedAwareMoDeDiT.

        Args:
            states: State observations dict
            actions: Action sequences
            goals: Goal conditions
            sigma: Noise levels
            uncond: Unconditional mode flag
            return_speed_info: If True, return speed and filter info

        Returns:
            pred_actions: Predicted actions (B, T, da)
            If return_speed_info: (pred_actions, speed, freq_filter)
        """
        t = 1

        # Process sigma embeddings
        emb_t = self.process_sigma_embeddings(sigma)

        # Preprocess goals
        goals = self.preprocess_goals(goals, 1, uncond=uncond)

        # Embed inputs
        if len(goals.shape) == 2:
            import einops
            goals = einops.rearrange(goals, 'b d -> b 1 d')

        state_embed = self.tok_emb(states['state_images'])
        if 'robot_obs' in states and self.use_proprio:
            proprio_embed = self.process_state_obs(states['robot_obs'].to(goals.dtype))
        else:
            proprio_embed = None
        goal_embed = self.goal_emb(goals)
        action_embed = self.action_emb(actions)

        # Position embeddings
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
                :, :(t + self.goal_seq_len + self.action_seq_len - 1), :
            ]
        else:
            position_embeddings = self.pos_emb[:, :t, :]

        # Add position embeddings
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len+t), :])
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len+t-1):, :])

        if 'robot_obs' in states and self.use_proprio:
            proprio_x = self.drop(proprio_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len+t)])
        else:
            proprio_x = None

        # Build input sequence
        input_seq = self.build_input_seq(state_x, action_x, goal_x, emb_t, proprio_x)

        # Custom attention mask if needed
        if self.use_custom_attn_mask:
            custom_mask = self.create_custom_mask(input_seq.shape[1])
        else:
            custom_mask = None

        # Prepare conditioning token
        cond_token = emb_t
        if self.use_goal_in_routing:
            cond_token = cond_token + goal_embed

        # Forward through MoDeDiT blocks
        x = self.forward_modedit(input_seq, cond_token, custom_attn_mask=custom_mask)

        # Get action outputs (last action_seq_len tokens)
        action_outputs = x[:, -self.action_seq_len:, :]  # (B, T, embed_dim)

        # Forward through output head
        if self.enable_speed_head:
            # Speed-aware head returns actions and speed
            pred_actions, pred_speed, freq_filter = self.out(
                action_outputs,
                return_speed=True,
                return_filter=True
            )

            # Store for loss computation
            self.pred_speed = pred_speed
            self.freq_filter = freq_filter

            if return_speed_info:
                return pred_actions, pred_speed, freq_filter
            else:
                return pred_actions
        else:
            # Standard linear output
            pred_actions = self.out(action_outputs)
            self.pred_speed = None
            self.freq_filter = None

            if return_speed_info:
                return pred_actions, None, None
            else:
                return pred_actions

    def get_speed_info(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get stored speed prediction and frequency filter.

        Returns:
            pred_speed: (B, 1) or None
            freq_filter: (B, T, 1) or None
        """
        return self.pred_speed, self.freq_filter
