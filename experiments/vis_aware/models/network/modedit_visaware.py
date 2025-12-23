import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[5]))

from experiments.flow_matching.models.network.modedit import MoDeDiT
import torch
import torch.nn as nn


class MoDeDiTVisAware(MoDeDiT):
    """
    Extended MoDeDiT that handles an additional vision cls token during denoising.

    Key changes from base MoDeDiT:
    1. Processes 11 tokens instead of 10 (10 action + 1 vision cls token)
    2. Vision cls token is concatenated with action tokens during forward pass
    3. Returns both action predictions and vision token predictions for dual loss computation
    """

    def __init__(
        self,
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
        vis_token_dim: int = 2048,
    ):
        """
        Args:
            vis_token_dim: Dimension of vision cls token from ResNet encoder
            Other args: Same as MoDeDiT
        """
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

        self.vis_token_dim = vis_token_dim
        self.vis_token_emb = nn.Linear(vis_token_dim, embed_dim, bias=False)

        if self.linear_output:
            self.vis_out = nn.Linear(embed_dim, vis_token_dim)
        else:
            from mode.models.networks.modedit import Mlp
            self.vis_out = Mlp(embed_dim, bias=False, dropout=mlp_pdrop, output_dim=vis_token_dim)

        # Create positional embedding that supports both with and without vision token
        # Max sequence: goal + obs + vis_token + actions
        seq_size_with_vis = goal_seq_len + obs_seq_len + 1 + action_seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size_with_vis, embed_dim))

    def forward(
        self,
        states,
        actions,
        goals,
        sigma,
        vis_token=None,
        uncond=False,
    ):
        """
        Forward pass with optional vision token.

        Args:
            states: State observations
            actions: Action tokens (noisy actions during training)
            goals: Goal embeddings
            sigma: Noise level
            vis_token: Vision cls token (B, vis_token_dim) or (B, 1, vis_token_dim)
                      During training, this should be the ground truth cls token
                      During inference, this will be predicted alongside actions
            uncond: Unconditional flag

        Returns:
            pred_actions: Predicted actions (B, action_seq_len, action_dim)
            pred_vis_token: Predicted vision token (B, 1, vis_token_dim) if vis_token is provided
        """
        t = 1

        emb_t = self.process_sigma_embeddings(sigma)
        goals = self.preprocess_goals(goals, 1, uncond=uncond)

        if len(goals.shape) == 2:
            import einops
            goals = einops.rearrange(goals, 'b d -> b 1 d')

        state_embed = self.tok_emb(states['state_images'])
        if 'robot_obs' in states and self.use_proprio:
            proprio_embed = self.process_state_obs(states['robot_obs'].to(goals.dtype))
        else:
            proprio_embed = None
        goal_embed = self.goal_emb(goals)

        if vis_token is not None:
            if len(vis_token.shape) == 2:
                import einops
                vis_token = einops.rearrange(vis_token, 'b d -> b 1 d')
            vis_token_embed = self.vis_token_emb(vis_token)
        else:
            vis_token_embed = None

        action_embed = self.action_emb(actions)

        if self.goal_conditioned:
            # Calculate total sequence length
            total_seq_len = self.goal_seq_len + t
            if vis_token_embed is not None:
                total_seq_len += 1
            total_seq_len += self.action_seq_len
            position_embeddings = self.pos_emb[:, :total_seq_len, :]
        else:
            position_embeddings = self.pos_emb[:, :t, :]

        # Assign position embeddings sequentially
        pos_idx = 0
        goal_x = self.drop(goal_embed + position_embeddings[:, pos_idx:pos_idx+self.goal_seq_len, :])
        pos_idx += self.goal_seq_len

        state_x = self.drop(state_embed + position_embeddings[:, pos_idx:pos_idx+t, :])
        pos_idx += t

        if vis_token_embed is not None:
            vis_x = self.drop(vis_token_embed + position_embeddings[:, pos_idx:pos_idx+1, :])
            pos_idx += 1
            action_x = self.drop(action_embed + position_embeddings[:, pos_idx:pos_idx+self.action_seq_len, :])
        else:
            action_x = self.drop(action_embed + position_embeddings[:, pos_idx:pos_idx+self.action_seq_len, :])

        if 'robot_obs' in states and self.use_proprio:
            proprio_x = self.drop(proprio_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len+t)])
        else:
            proprio_x = None

        if vis_token_embed is not None:
            input_seq = self.build_input_seq_with_vis(
                state_x, action_x, goal_x, emb_t, proprio_x, vis_x
            )
        else:
            input_seq = self.build_input_seq(state_x, action_x, goal_x, emb_t, proprio_x)

        if self.use_custom_attn_mask:
            custom_mask = self.create_custom_mask(input_seq.shape[1])
        else:
            custom_mask = None

        cond_token = emb_t

        if self.use_goal_in_routing:
            cond_token = cond_token + goal_embed

        x = self.forward_modedit(input_seq, cond_token, custom_attn_mask=custom_mask)

        if vis_token_embed is not None:
            vis_output = x[:, -self.action_seq_len-1:-self.action_seq_len, :]
            action_outputs = x[:, -self.action_seq_len:, :]

            pred_vis_token = self.vis_out(vis_output)
            pred_actions = self.out(action_outputs)

            return pred_actions, pred_vis_token
        else:
            action_outputs = x[:, -self.action_seq_len:, :]
            pred_actions = self.out(action_outputs)
            return pred_actions

    def build_input_seq_with_vis(self, state_x, action_x, goal_x=None, emb_t=None, proprio_embed=None, vis_x=None):
        """Build input sequence including vision token"""
        sequences = []
        if self.use_noise_token_as_input and emb_t is not None:
            sequences.append(emb_t)

        if self.goal_conditioned and goal_x is not None:
            sequences.append(goal_x)

        if proprio_embed is not None:
            sequences.append(proprio_embed)

        sequences.append(state_x)

        if vis_x is not None:
            sequences.append(vis_x)

        sequences.append(action_x)

        return torch.cat(sequences, dim=1)
