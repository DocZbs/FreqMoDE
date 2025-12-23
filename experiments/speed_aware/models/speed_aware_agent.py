"""
Speed-Aware Flow Matching Agent.

Extends FlowMatchingAgent with speed prediction auxiliary loss.
Completely independent from FreqFlowMatchingAgent.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_info

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

from experiments.flow_matching.models.flow_agent import FlowMatchingAgent
from experiments.freq_flow.models.speed_aware_head import (
    SpeedAwareLoss,
    compute_speed_label
)

logger = logging.getLogger(__name__)


class SpeedAwareFlowAgent(FlowMatchingAgent):
    """
    Flow Matching Agent with Speed-Aware Smoothing.

    Key additions over baseline FlowMatchingAgent:
    1. Speed prediction auxiliary loss
    2. Optional speed visualization/logging
    3. No frequency-domain expert decomposition (unlike FreqFlowAgent)

    Training flow:
    - Main loss: Standard Flow Matching velocity prediction
    - Auxiliary loss: Predict motion speed from hidden state
    """

    def __init__(
        self,
        # All base FlowMatchingAgent parameters
        language_goal: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        latent_dim: int = 512,
        multistep: int = 10,
        flow_type: str = 'rectified',
        num_sampling_steps: int = 20,
        sampling_method: str = 'euler',
        sigma_min: float = 1e-4,
        sigma_max: float = 80.0,
        noise_schedule: str = 'linear',
        use_ot_flow: bool = True,
        use_perceiver: bool = False,
        obs_enc_dim: int = 512,
        cond_dim: int = 512,
        use_lr_scheduler: bool = True,
        ckpt_path=None,
        seed: int = 42,
        entropy_gamma: float = 0.0,
        router_z_delta: float = 0.001,
        start_from_pretrained: bool = False,
        use_text_not_embedding: bool = True,
        use_proprio: bool = False,
        act_window_size: int = 10,
        resnet_type: str = '18',
        # Speed-aware specific parameters
        speed_loss_weight: float = 0.1,
        use_global_speed_norm: bool = True,
        log_speed_every_n_steps: int = 100,
    ):
        """
        Initialize SpeedAwareFlowAgent.

        Args:
            All standard FlowMatchingAgent args, plus:
            speed_loss_weight: Weight for speed prediction auxiliary loss
            use_global_speed_norm: Use global statistics for speed normalization
            log_speed_every_n_steps: Log speed predictions every N steps
        """
        # Initialize parent FlowMatchingAgent
        super().__init__(
            language_goal=language_goal,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            latent_dim=latent_dim,
            multistep=multistep,
            flow_type=flow_type,
            num_sampling_steps=num_sampling_steps,
            sampling_method=sampling_method,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            noise_schedule=noise_schedule,
            use_ot_flow=use_ot_flow,
            use_perceiver=use_perceiver,
            obs_enc_dim=obs_enc_dim,
            cond_dim=cond_dim,
            use_lr_scheduler=use_lr_scheduler,
            ckpt_path=ckpt_path,
            seed=seed,
            entropy_gamma=entropy_gamma,
            router_z_delta=router_z_delta,
            start_from_pretrained=start_from_pretrained,
            use_text_not_embedding=use_text_not_embedding,
            use_proprio=use_proprio,
            act_window_size=act_window_size,
            resnet_type=resnet_type,
        )

        # Speed-aware loss module
        self.speed_loss_fn = SpeedAwareLoss(
            speed_loss_weight=speed_loss_weight,
            use_global_norm=use_global_speed_norm
        )

        self.log_speed_every_n_steps = log_speed_every_n_steps

        logger.info(
            f"Initialized SpeedAwareFlowAgent with speed_loss_weight={speed_loss_weight}"
        )

    def speed_aware_flow_matching_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flow matching loss with speed prediction auxiliary loss.

        Args:
            perceptual_emb: Perceptual embeddings
            latent_goal: Goal embeddings
            actions: Target action sequences

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components for logging
        """
        self.model.train()
        batch_size = actions.shape[0]

        # Sample time uniformly from [0, 1]
        t = self.flow_matcher.sample_time(batch_size, self.device)

        # Sample source distribution (standard Gaussian noise)
        x0 = torch.randn_like(actions).to(self.device)

        # Target distribution is the data
        x1 = actions

        # Compute conditional flow: interpolated state x_t and target velocity u_t
        x_t, u_t = self.flow_matcher.compute_conditional_flow(x0, x1, t)

        # Predict velocity using the model
        velocity_pred = self.model(perceptual_emb, x_t, latent_goal, t)

        # Standard flow matching loss
        fm_loss = self.flow_matcher.compute_loss(velocity_pred, u_t)

        # Get predicted speed from inner model (if available)
        inner_model = self.model.inner_model
        if hasattr(inner_model, 'get_speed_info'):
            pred_speed, freq_filter = inner_model.get_speed_info()

            if pred_speed is not None:
                # Compute speed-aware loss
                # Note: We use target actions (x1) for speed label, not noisy x_t
                speed_loss_dict = self.speed_loss_fn(
                    pred_actions=velocity_pred,
                    target_actions=actions,  # Use clean target for speed computation
                    pred_speed=pred_speed
                )

                total_loss = fm_loss + speed_loss_dict['speed_loss']

                loss_dict = {
                    'fm_loss': fm_loss,
                    'speed_loss': speed_loss_dict['speed_loss'],
                    'speed_label': speed_loss_dict['speed_label'],
                    'speed_pred': speed_loss_dict['speed_pred'],
                    'total': total_loss
                }
            else:
                # Model doesn't have speed prediction
                total_loss = fm_loss
                loss_dict = {
                    'fm_loss': fm_loss,
                    'total': total_loss
                }
        else:
            # Fallback: standard FM loss only
            logger.warning(
                "Model does not have speed info, using standard FM loss. "
                "Make sure you're using SpeedAwareMoDeDiT with enable_speed_head=True."
            )
            total_loss = fm_loss
            loss_dict = {
                'fm_loss': fm_loss,
                'total': total_loss
            }

        return total_loss, loss_dict

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Training step with speed-aware auxiliary loss.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Total loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)

        # Accumulate speed-related metrics
        speed_metrics = {
            'fm_loss': torch.tensor(0.0, device=self.device),
            'speed_loss': torch.tensor(0.0, device=self.device),
            'speed_label': torch.tensor(0.0, device=self.device),
            'speed_pred': torch.tensor(0.0, device=self.device),
        }

        total_bs = 0
        batch_sizes = []

        for self.modality_scope, dataset_batch in batch.items():
            # Compute embeddings
            perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)

            # Compute speed-aware flow matching loss
            act_loss, loss_dict = self.speed_aware_flow_matching_loss(
                perceptual_emb,
                latent_goal,
                dataset_batch["actions"],
            )

            # Accumulate metrics
            for key in speed_metrics.keys():
                if key in loss_dict:
                    speed_metrics[key] += loss_dict[key]

            # Add auxiliary losses (router losses from MoE backbone)
            if self.entropy_gamma > 0:
                entropy_loss = self.model.inner_model.load_balancing_loss()
                total_loss += entropy_loss * self.entropy_gamma

            if self.router_z_delta > 0:
                router_z_loss = self.model.inner_model.compute_router_z_loss()
                total_loss += self.router_z_delta * router_z_loss

            action_loss += act_loss
            total_loss += act_loss

            batch_sizes.append(dataset_batch["actions"].shape[0])
            total_bs += dataset_batch["actions"].shape[0]

        # Average losses
        batch_len = len(batch)
        total_loss = total_loss / batch_len
        action_loss = action_loss / batch_len

        for key in speed_metrics.keys():
            speed_metrics[key] = speed_metrics[key] / batch_len

        # Log metrics
        self.log("train/action_loss", action_loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=total_bs)

        # Log FM loss on every step
        self.log("train/fm_loss", speed_metrics['fm_loss'],
                 on_step=True, on_epoch=True, prog_bar=False,
                 sync_dist=True, batch_size=total_bs)

        # Log speed metrics
        if speed_metrics['speed_loss'] > 0:  # Only if speed head is enabled
            # Log speed loss every step for monitoring
            self.log("train/speed_loss", speed_metrics['speed_loss'],
                     on_step=True, on_epoch=True, prog_bar=False,
                     sync_dist=True, batch_size=total_bs)

            # Log speed values less frequently
            if batch_idx % self.log_speed_every_n_steps == 0:
                self.log("train/speed_label_avg", speed_metrics['speed_label'],
                         on_step=True, on_epoch=False, prog_bar=False,
                         sync_dist=True, batch_size=total_bs)
                self.log("train/speed_pred_avg", speed_metrics['speed_pred'],
                         on_step=True, on_epoch=False, prog_bar=False,
                         sync_dist=True, batch_size=total_bs)

        # Log auxiliary losses
        if self.entropy_gamma > 0:
            self.log("train/load_balancing_loss", entropy_loss,
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        if self.router_z_delta > 0:
            self.log("train/router_z_loss", router_z_loss,
                     on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return total_loss

    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Validation step - same as parent but with speed logging.
        """
        # Call parent validation
        val_loss = super().validation_step(batch, batch_idx)

        # Optionally log speed statistics on validation set
        if batch_idx == 0:  # Log only first batch to avoid overhead
            try:
                for modality_scope, dataset_batch in batch.items():
                    actions = dataset_batch["actions"]

                    # Compute ground truth speed distribution
                    with torch.no_grad():
                        speed_labels = compute_speed_label(actions)
                        self.log("val/speed_label_mean", speed_labels.mean(),
                                 on_step=False, on_epoch=True, sync_dist=True)
                        self.log("val/speed_label_std", speed_labels.std(),
                                 on_step=False, on_epoch=True, sync_dist=True)
            except Exception as e:
                # Skip speed logging if it fails (e.g., during sanity check)
                pass

        return val_loss
