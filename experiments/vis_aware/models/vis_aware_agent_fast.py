"""
Fast version of VisAwareFlowMatchingAgent that uses precomputed CLIP features.

Speed improvement: ~3-5x faster training!
"""

import logging
from pathlib import Path
import torch
from typing import Dict, Tuple

from experiments.vis_aware.models.vis_aware_agent import VisAwareFlowMatchingAgent


logger = logging.getLogger(__name__)


class VisAwareFlowMatchingAgentFast(VisAwareFlowMatchingAgent):
    """
    Fast version that loads precomputed CLIP cls_token from disk.

    Key changes:
    1. No CLIP encoder during training (use precomputed features)
    2. Dataset should provide 'clip_static' and 'clip_gripper' keys
    3. Significantly faster training
    """

    def __init__(self, *args, use_precomputed_clip: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_precomputed_clip = use_precomputed_clip

        if use_precomputed_clip:
            logger.info("Using precomputed CLIP features (fast mode)")
            # 不需要在训练时使用CLIP encoder，可以删除或设为None
            # 但保留用于inference
        else:
            logger.info("Computing CLIP features on-the-fly (slow mode)")

    def embed_visual_obs_with_cls(
        self,
        rgb_static: torch.Tensor,
        rgb_gripper: torch.Tensor,
        latent_goal: torch.Tensor,
        precomputed_clip_static: torch.Tensor = None,
        precomputed_clip_gripper: torch.Tensor = None,
    ):
        """
        Embed visual observations with optional precomputed CLIP features.

        Args:
            rgb_static: Static camera images (B, T, C, H, W)
            rgb_gripper: Gripper camera images (B, T, C, H, W)
            latent_goal: Language goal embedding (B, D)
            precomputed_clip_static: Precomputed CLIP features (B, T, 512) or None
            precomputed_clip_gripper: Precomputed CLIP features (B, T, 512) or None

        Returns:
            perceptual_emb: Dict with 'state_images' key
            cls_tokens: CLIP cls tokens (B*T, 1024)
        """
        import einops

        batch_size, seq_len = rgb_static.shape[0], rgb_static.shape[1]
        rgb_static = einops.rearrange(rgb_static, 'b t c h w -> (b t) c h w')
        rgb_gripper = einops.rearrange(rgb_gripper, 'b t c h w -> (b t) c h w')

        # Expand latent_goal to match (B*T) batch size
        if len(latent_goal.shape) == 3:
            latent_goal = latent_goal.squeeze(1)
        latent_goal_expanded = einops.repeat(latent_goal, 'b d -> (b t) d', t=seq_len)

        # ResNet features (always compute)
        static_tokens = self.static_resnet.resnet_encoder(rgb_static, latent_goal_expanded)
        gripper_tokens = self.gripper_resnet.resnet_encoder(rgb_gripper, latent_goal_expanded)

        batch_size = rgb_static.shape[0]
        static_tokens = static_tokens.view(batch_size, 1, -1)
        gripper_tokens = gripper_tokens.view(batch_size, 1, -1)

        perceptual_emb = {
            'state_images': torch.cat([static_tokens, gripper_tokens], dim=-1)
        }

        # CLIP cls tokens
        if self.use_precomputed_clip and precomputed_clip_static is not None:
            # Use precomputed features (fast!)
            # Reshape from (B, T, 512) to (B*T, 512)
            static_cls = einops.rearrange(precomputed_clip_static, 'b t d -> (b t) d')
            gripper_cls = einops.rearrange(precomputed_clip_gripper, 'b t d -> (b t) d')
        else:
            # Compute on-the-fly (slow)
            static_cls = self.static_resnet.clip_encoder(rgb_static)
            gripper_cls = self.gripper_resnet.clip_encoder(rgb_gripper)

        # Concatenate static + gripper cls tokens
        cls_tokens = torch.cat([static_cls, gripper_cls], dim=-1)  # (B*T, 1024)

        return perceptual_emb, cls_tokens

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        """
        Training step with precomputed CLIP features.

        Expected batch keys:
        - rgb_obs: RGB images (as before)
        - clip_static: Precomputed CLIP features for static cam (NEW)
        - clip_gripper: Precomputed CLIP features for gripper cam (NEW)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        action_loss = torch.tensor(0.0, device=self.device)
        vis_loss = torch.tensor(0.0, device=self.device)
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

            # Get precomputed CLIP features from batch (if available)
            precomputed_clip_static = dataset_batch.get("clip_static", None)
            precomputed_clip_gripper = dataset_batch.get("clip_gripper", None)

            if precomputed_clip_static is not None:
                precomputed_clip_static = precomputed_clip_static.to(self.device)
                precomputed_clip_gripper = precomputed_clip_gripper.to(self.device)

            # Embed with precomputed features
            perceptual_emb, cls_token_gt = self.embed_visual_obs_with_cls(
                rgb_static,
                rgb_gripper,
                latent_goal,
                precomputed_clip_static=precomputed_clip_static,
                precomputed_clip_gripper=precomputed_clip_gripper,
            )

            if self.use_proprio:
                perceptual_emb['robot_obs'] = dataset_batch['robot_obs']

            batch_total_loss, batch_action_loss, batch_vis_loss = self.flow_matching_loss_with_vision(
                perceptual_emb,
                latent_goal,
                dataset_batch["actions"],
                cls_token_gt,
            )

            if self.entropy_gamma > 0:
                entropy_loss = self.model.inner_model.load_balancing_loss()
                batch_total_loss += entropy_loss * self.entropy_gamma

            if self.router_z_delta > 0:
                router_z_loss = self.model.inner_model.compute_router_z_loss()
                batch_total_loss += self.router_z_delta * router_z_loss

            total_loss += batch_total_loss
            action_loss += batch_action_loss
            vis_loss += batch_vis_loss

            total_bs += dataset_batch["actions"].shape[0]

        batch_len = len(batch)
        total_loss = total_loss / batch_len
        action_loss = action_loss / batch_len
        vis_loss = vis_loss / batch_len

        self.log("train/action_loss", action_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)
        self.log("train/vis_loss", vis_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=total_bs)

        if self.entropy_gamma > 0:
            self.log("train/load_balancing_loss", entropy_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        if self.router_z_delta > 0:
            self.log("train/router_z_delta", router_z_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

        return total_loss
