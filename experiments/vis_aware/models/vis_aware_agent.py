import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import einops
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).absolute().parents[3]))
sys.path.insert(0, str(Path(__file__).absolute().parents[4]))

from experiments.flow_matching.models.flow_agent import FlowMatchingAgent
from mode.models.perceptual_encoders.pretrained_resnets import FiLMResNet50Policy, FiLMResNet34Policy

logger = logging.getLogger(__name__)


class VisAwareFlowMatchingAgent(FlowMatchingAgent):
    """
    Vision-Aware Flow Matching Agent.

    Key innovation: During denoising, concat a vision cls token alongside action tokens.
    The ground truth for this vision token comes from the vision encoder's cls token.

    Training process:
    1. Extract cls token from vision encoder (represents visual semantics)
    2. During flow matching, denoise both actions (10 tokens) and vision token (1 token)
    3. Compute dual loss: action_fm_loss + vis_fm_loss
    4. The vision token helps actions implicitly attend to visual features during denoising

    This allows the action denoising process to constantly "see" visual features,
    potentially learning better action representations.
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
        resnet_type: str = '50',
        cls_token_dim: int = 2048,
    ):
        """
        Args:
            cls_token_dim: Dimension of cls token from vision encoder (used as conditioning)
            Other args: Same as FlowMatchingAgent
        """
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

        self.cls_token_dim = cls_token_dim

        obs_dim = 2048 if resnet_type == '50' else 512
        if resnet_type == '50':
            ResNetClass = FiLMResNet50Policy
        elif resnet_type == '34':
            ResNetClass = FiLMResNet34Policy
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Use hybrid encoder: ResNet for features + shared CLIP for cls_token
        # Create single shared CLIP encoder to save memory
        from experiments.vis_aware.models.clip_vision_encoder import ClipVisionEncoder
        self.shared_clip_encoder = ClipVisionEncoder(
            model_name="ViT-B/32",
            freeze_backbone=True,
            device=self.device
        )

        # Create separate ResNet encoders for each camera
        self.static_resnet = ResNetClass(cond_dim)
        self.gripper_resnet = ResNetClass(cond_dim)

        logger.info(
            f"Initialized VisAwareFlowMatchingAgent with "
            f"cls_token_dim={cls_token_dim}, "
            f"using shared CLIP encoder (memory optimized), "
            f"vision token as conditioning (not denoised)"
        )

    def embed_visual_obs_with_cls(self, rgb_static, rgb_gripper, latent_goal):
        """
        Embed visual observations and extract cls tokens.

        Args:
            rgb_static: Static camera images (B, T, C, H, W)
            rgb_gripper: Gripper camera images (B, T, C, H, W)
            latent_goal: Language goal embedding (B, D) or (B, 1, D)

        Returns:
            perceptual_emb: Dict with 'state_images' key
            cls_tokens: Concatenated cls tokens from both cameras (B, cls_token_dim)
        """
        # Use the most recent frame so batch size matches the action batch (B).
        if rgb_static.dim() == 5:
            rgb_static = rgb_static[:, -1]
        if rgb_gripper.dim() == 5:
            rgb_gripper = rgb_gripper[:, -1]

        if len(latent_goal.shape) == 3:
            latent_goal = latent_goal.squeeze(1)

        # Extract ResNet features separately for each camera
        static_tokens = self.static_resnet(rgb_static, latent_goal)
        gripper_tokens = self.gripper_resnet(rgb_gripper, latent_goal)

        # Extract CLIP cls tokens using shared CLIP encoder
        static_cls = self.shared_clip_encoder(rgb_static)
        gripper_cls = self.shared_clip_encoder(rgb_gripper)

        perceptual_emb = {'state_images': torch.stack([static_tokens, gripper_tokens], dim=1)}

        # Concatenate cls tokens along feature dimension for the vision loss
        cls_tokens = torch.cat([static_cls, gripper_cls], dim=-1)

        return perceptual_emb, cls_tokens

    def flow_matching_loss_with_vision(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        cls_token_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss with vision token as conditioning.

        Key insight: Vision token is used as CONDITIONING, not denoised.
        This ensures train-test consistency - both use real cls_token.

        Args:
            perceptual_emb: Perceptual embeddings
            latent_goal: Goal embeddings
            actions: Target action sequences
            cls_token_gt: Ground truth cls token from vision encoder (used as conditioning)

        Returns:
            action_loss: Action flow matching loss
        """
        self.model.train()
        batch_size = actions.shape[0]

        t = self.flow_matcher.sample_time(batch_size, self.device)

        x0_action = torch.randn_like(actions).to(self.device)
        x1_action = actions

        x_t_action, u_t_action = self.flow_matcher.compute_conditional_flow(x0_action, x1_action, t)

        # Use real cls_token as conditioning (not denoised)
        result = self.model(
            perceptual_emb, x_t_action, latent_goal, t, vis_token=cls_token_gt
        )

        # Model may return tuple (action_velocity, vis_velocity) or just action_velocity
        if isinstance(result, tuple):
            velocity_pred_action, _ = result
        else:
            velocity_pred_action = result

        action_loss = self.flow_matcher.compute_loss(velocity_pred_action, u_t_action)

        return action_loss

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Training step with vision-conditioned flow matching.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Total loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        total_bs = 0

        for self.modality_scope, dataset_batch in batch.items():
            rgb_static = dataset_batch["rgb_obs"]['rgb_static']
            rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper']

            if self.use_text_not_embedding:
                latent_goal = self.lang_buffer.get_goal_instruction_embeddings(
                    dataset_batch["lang_text"]
                ).to(rgb_static.dtype)
            else:
                latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)

            perceptual_emb, cls_token_gt = self.embed_visual_obs_with_cls(
                rgb_static, rgb_gripper, latent_goal
            )

            if self.use_proprio:
                perceptual_emb['robot_obs'] = dataset_batch['robot_obs']

            batch_action_loss = self.flow_matching_loss_with_vision(
                perceptual_emb,
                latent_goal,
                dataset_batch["actions"],
                cls_token_gt,
            )

            batch_total_loss = batch_action_loss

            if self.entropy_gamma > 0:
                entropy_loss = self.model.inner_model.load_balancing_loss()
                batch_total_loss += entropy_loss * self.entropy_gamma

            if self.router_z_delta > 0:
                router_z_loss = self.model.inner_model.compute_router_z_loss()
                batch_total_loss += self.router_z_delta * router_z_loss

            total_loss += batch_total_loss
            action_loss += batch_action_loss

            total_bs += dataset_batch["actions"].shape[0]

        batch_len = len(batch)
        total_loss = total_loss / batch_len
        action_loss = action_loss / batch_len

        self.log("train/action_loss", action_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)

        if self.entropy_gamma > 0:
            self.log("train/load_balancing_loss", entropy_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        if self.router_z_delta > 0:
            self.log("train/router_z_delta", router_z_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return total_loss

    def forward(self, obs, goal):
        """
        Method for doing inference with the model.
        Overrides parent to use vision-aware embedding.
        """
        if self.use_text_not_embedding:
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(goal["lang_text"])
            latent_goal = latent_goal.to(torch.float32)
        else:
            latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(obs["rgb_obs"]['rgb_static'].device)

        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        perceptual_emb, cls_token_gt = self.embed_visual_obs_with_cls(rgb_static, rgb_gripper, latent_goal)

        act_seq = self.generate_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            cls_token_gt=cls_token_gt,
            inference=True,
        )
        return act_seq

    def generate_actions(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        cls_token_gt: Optional[torch.Tensor] = None,
        inference: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Generate actions using flow matching ODE sampling.

        Vision token is used as conditioning (not denoised).
        Train-test consistency: both use real cls_token as input.

        Args:
            latent_plan: Latent plan (unused)
            perceptual_emb: Perceptual embeddings
            latent_goal: Goal embeddings
            cls_token_gt: Ground truth cls token (used as conditioning)
            inference: Whether in inference mode

        Returns:
            Generated action sequences
        """
        self.model.eval()

        if len(latent_goal.shape) < len(
            perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape
        ):
            latent_goal = latent_goal.unsqueeze(1)

        x_init_action = torch.randn(
            (len(latent_goal), self.act_window_size, 7),
            device=self.device
        )

        num_steps = self.num_sampling_steps if inference else 10
        dt = 1.0 / num_steps

        x_action = x_init_action

        # Use real cls_token as conditioning (same for train and inference)
        for step in range(num_steps):
            t = torch.ones(x_action.shape[0], device=self.device) * (step / num_steps)

            result = self.model(
                perceptual_emb, x_action, latent_goal, t, vis_token=cls_token_gt
            )

            # Model may return tuple (action_velocity, vis_velocity) or just action_velocity
            if isinstance(result, tuple):
                velocity_action, _ = result
            else:
                velocity_action = result

            x_action = x_action + dt * velocity_action

        return x_action

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

        rgb_static = dataset_batch["rgb_obs"]['rgb_static']
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper']

        if self.use_text_not_embedding:
            latent_goal = self.lang_buffer.get_goal_instruction_embeddings(
                dataset_batch["lang_text"]
            ).to(rgb_static.dtype)
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)

        perceptual_emb, cls_token_gt = self.embed_visual_obs_with_cls(
            rgb_static, rgb_gripper, latent_goal
        )

        if self.use_proprio:
            perceptual_emb['robot_obs'] = dataset_batch['robot_obs']

        action_pred = self.generate_actions(
            torch.zeros_like(latent_goal).to(self.device),
            perceptual_emb,
            latent_goal,
            cls_token_gt=cls_token_gt,
            inference=True,
        )

        actions = dataset_batch["actions"].to(self.device)
        pred_loss = torch.nn.functional.mse_loss(action_pred, actions)

        # Also compute validation loss for monitoring
        batch_action_loss = self.flow_matching_loss_with_vision(
            perceptual_emb,
            latent_goal,
            actions,
            cls_token_gt,
        )

        batch_total_loss = batch_action_loss

        if self.entropy_gamma > 0:
            entropy_loss = self.model.inner_model.load_balancing_loss()
            batch_total_loss += entropy_loss * self.entropy_gamma

        if self.router_z_delta > 0:
            router_z_loss = self.model.inner_model.compute_router_z_loss()
            batch_total_loss += self.router_z_delta * router_z_loss

        self.log("val/action_mse", pred_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/action_loss", batch_action_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", batch_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self._log_validation_metrics(pred_loss)

        output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
        output["validation_loss"] = pred_loss

        return output
