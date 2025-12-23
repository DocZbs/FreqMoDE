import logging
import os
from typing import Any, Dict, Optional, Tuple
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import einops
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).absolute().parents[4]))
sys.path.insert(0, str(Path(__file__).absolute().parents[5]))

from mode.models.mode_agent import MoDEAgent
from experiments.flow_matching.models.flow_core import (
    ConditionalFlowMatcher,
    RectifiedFlowMatcher,
    FlowMatchingSampler,
)

logger = logging.getLogger(__name__)


class FlowMatchingAgent(MoDEAgent):
    """
    Flow Matching Agent that extends MoDEAgent.

    Replaces the diffusion training objective with flow matching.
    Key differences from diffusion:
    - Learns velocity field instead of denoising
    - Uses straight ODE paths instead of stochastic diffusion
    - Simpler and more efficient training
    """

    def __init__(
        self,
        language_goal: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        latent_dim: int = 512,
        multistep: int = 10,
        flow_type: str = 'conditional',
        num_sampling_steps: int = 10,
        sampling_method: str = 'euler',
        sigma_min: float = 1e-4,
        sigma_max: float = 80.0,
        noise_schedule: str = 'exponential',
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
    ):
        """
        Args:
            flow_type: Type of flow matching ('conditional', 'rectified')
            sampling_method: ODE solver ('euler', 'heun', 'rk4', 'midpoint')
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            noise_schedule: Type of noise schedule ('constant', 'exponential', 'linear')
            use_ot_flow: Use Optimal Transport flow paths
            Other args inherited from MoDEAgent
        """
        # Initialize parent class (MoDEAgent)
        super().__init__(
            language_goal=language_goal,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            latent_dim=latent_dim,
            multistep=multistep,
            sampler_type=sampling_method,
            num_sampling_steps=num_sampling_steps,
            sigma_data=0.5,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            noise_scheduler='exponential',
            sigma_sample_density_type='loglogistic',
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

        # Flow matching specific initialization
        self.flow_type = flow_type
        self.use_ot_flow = use_ot_flow
        self.noise_schedule = noise_schedule

        # Initialize flow matcher
        if flow_type == 'conditional':
            self.flow_matcher = ConditionalFlowMatcher(
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                use_ot_flow=use_ot_flow,
                noise_schedule=noise_schedule,
            )
        elif flow_type == 'rectified':
            self.flow_matcher = RectifiedFlowMatcher()
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        # Initialize sampler
        self.flow_sampler = FlowMatchingSampler(
            num_steps=num_sampling_steps,
            method=sampling_method,
        )

        logger.info(f"Initialized FlowMatchingAgent with {flow_type} flow, noise_schedule={noise_schedule}, sigma_min={sigma_min}, sigma_max={sigma_max}")

    def flow_matching_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow matching loss.

        Instead of adding noise and predicting it (diffusion),
        we interpolate between noise and data, and predict the velocity field.

        Args:
            perceptual_emb: Perceptual embeddings
            latent_goal: Goal embeddings
            actions: Target action sequences

        Returns:
            Flow matching loss
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
        # Pass time t directly to the wrapper, which will convert it to sigma
        velocity_pred = self.model(perceptual_emb, x_t, latent_goal, t)

        # Compute loss
        loss = self.flow_matcher.compute_loss(velocity_pred, u_t)

        return loss

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Training step using flow matching instead of diffusion.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Total loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0
        batch_sizes = []

        for self.modality_scope, dataset_batch in batch.items():
            # Compute embeddings
            perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)

            # Compute flow matching loss
            act_loss = self.flow_matching_loss(
                perceptual_emb,
                latent_goal,
                dataset_batch["actions"],
            )

            # Add auxiliary losses (same as diffusion version)
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

        # Log metrics
        self.log("train/action_loss", action_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)
        if self.entropy_gamma > 0:
            self.log("train/load_balancing_loss", entropy_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        if self.router_z_delta > 0:
            self.log("train/router_z_delta", router_z_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return total_loss

    def generate_actions(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        inference: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Generate actions using flow matching ODE sampling.

        Args:
            latent_plan: Latent plan (unused, kept for compatibility)
            perceptual_emb: Perceptual embeddings
            latent_goal: Goal embeddings
            inference: Whether in inference mode

        Returns:
            Generated action sequences
        """
        self.model.eval()

        if len(latent_goal.shape) < len(
            perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape
        ):
            latent_goal = latent_goal.unsqueeze(1)

        # Initial state: random noise
        x_init = torch.randn(
            (len(latent_goal), self.act_window_size, 7),
            device=self.device
        )

        # Sample using ODE integration
        actions = self.flow_sampler.sample(
            self.model,
            perceptual_emb,
            x_init,
            latent_goal,
            num_steps=self.num_sampling_steps if inference else 10,
        )

        return actions

    def forward(self, obs, goal):
        """
        Forward pass for inference using flow matching.

        Args:
            obs: Observations
            goal: Goal specification

        Returns:
            Predicted action sequence
        """
        if self.use_text_not_embedding:
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(goal["lang_text"])
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(obs["rgb_obs"]['rgb_static'].device)

        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper, latent_goal)

        act_seq = self.generate_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return act_seq

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> None:
        """
        Validation step using flow matching generation.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation outputs
        """
        output = {}
        dataset_batch = batch
        perceptual_emb, latent_goal = self.compute_input_embeddings(dataset_batch)

        action_pred = self.generate_actions(
            torch.zeros_like(latent_goal).to(self.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )

        actions = dataset_batch["actions"].to(self.device)
        pred_loss = torch.nn.functional.mse_loss(action_pred, actions)

        self._log_validation_metrics(pred_loss)

        output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
        output["validation_loss"] = pred_loss

        return output


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)
