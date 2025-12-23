import hydra
import torch
import torch.nn as nn
from typing import Optional


class FlowMatchingWrapper(nn.Module):
    """
    Wrapper for Flow Matching models.
    Adapts the model interface to predict velocity fields instead of denoising.

    This wrapper makes the flow model compatible with the existing
    MoDE architecture (MoDeDiT backbone).
    """

    def __init__(
        self,
        inner_model,
        flow_type: str = 'conditional',
        sigma_min: float = 1e-3,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        use_time_scaling: bool = False,
        noise_schedule: str = 'exponential',
        rectified_use_linear_time: bool = True,
    ):
        """
        Args:
            inner_model: The backbone model (e.g., MoDeDiT)
            flow_type: Type of flow ('conditional', 'rectified')
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            sigma_data: Data scale for time scaling
            use_time_scaling: Whether to use time-dependent scaling
            noise_schedule: Type of noise schedule ('constant', 'exponential', 'linear')
            rectified_use_linear_time: For rectified flow, use t directly instead of noise schedule
                                       Set to False for backward compatibility with old checkpoints
        """
        super().__init__()
        self.inner_model = hydra.utils.instantiate(inner_model)
        self.flow_type = flow_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.use_time_scaling = use_time_scaling
        self.noise_schedule = noise_schedule
        # Backward compatibility: old checkpoints trained with rectified flow used noise_schedule
        # Set rectified_use_linear_time=False to maintain consistency with those checkpoints
        self.rectified_use_linear_time = rectified_use_linear_time

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma from time using the same schedule as ConditionalFlowMatcher.

        Args:
            t: Time in [0, 1], shape (B,)

        Returns:
            sigma: Noise levels, shape (B,)
        """
        if self.noise_schedule == 'constant':
            return torch.full_like(t, self.sigma_min)
        elif self.noise_schedule == 'exponential':
            return self.sigma_max ** (1 - t) * self.sigma_min ** t
        elif self.noise_schedule == 'linear':
            return (1 - t) * self.sigma_max + t * self.sigma_min
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def get_time_scalings(self, t: torch.Tensor) -> tuple:
        """
        Compute time-dependent scalings (similar to EDM but for flow matching).

        Args:
            t: Time in [0, 1], shape (B,)

        Returns:
            c_in: Input scaling
            c_out: Output scaling
        """
        if not self.use_time_scaling:
            return torch.ones_like(t), torch.ones_like(t)

        sigma_t = self.get_noise_schedule(t)

        # EDM-style scaling adapted for flow matching
        c_in = 1.0 / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
        c_out = sigma_t * self.sigma_data / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)

        return c_in, c_out

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        goal: torch.Tensor,
        t: torch.Tensor,    
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass that predicts velocity field.

        Args:
            state: State observations (dict with 'state_images')
            x_t: Current flow state at time t
            goal: Goal conditioning
            t: Time in [0, 1], shape (B,)
            **kwargs: Additional arguments

        Returns:
            Predicted velocity field v_θ(x_t, t, condition)
        """
        # For rectified flow, use t directly as sigma for linear time conditioning
        # For conditional flow, convert time to sigma using noise schedule
        # For backward compatibility, rectified can also use noise schedule if rectified_use_linear_time=False

        sigma = t

        if self.use_time_scaling:
            c_in, c_out = self.get_time_scalings(t)

            # Expand dimensions for broadcasting
            c_in = c_in.view(-1, 1, 1)
            c_out = c_out.view(-1, 1, 1)

            # Scale input
            scaled_x_t = x_t * c_in

            # Predict velocity - MoDeDiT expects (states, actions, goals, sigma)
            result = self.inner_model(
                states=state,      # MoDeDiT expects 'states' (dict)
                actions=scaled_x_t, # MoDeDiT expects 'actions'
                goals=goal,        # MoDeDiT expects 'goals'
                sigma=sigma,       # MoDeDiT expects 'sigma'
                **kwargs
            )

            # Handle both single and dual output (for vision-aware models)
            if isinstance(result, tuple):
                velocity_action, velocity_vis = result
                velocity_action = velocity_action * c_out
                velocity_vis = velocity_vis * c_out
                velocity = (velocity_action, velocity_vis)
            else:
                velocity = result * c_out
        else:
            # Direct velocity prediction without scaling
            # MoDeDiT expects (states, actions, goals, sigma)
            result = self.inner_model(
                states=state,
                actions=x_t,
                goals=goal,
                sigma=sigma,
                **kwargs
            )
            # Handle dual output for vision-aware models
            velocity = result

        return velocity

    def compute_loss(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        goal: torch.Tensor,
        t: torch.Tensor,
        target_velocity: torch.Tensor,
        **kwargs
    ) -> tuple:
        """
        Compute the flow matching loss.

        Args:
            state: State observations
            x_t: Flow state at time t
            goal: Goal conditioning
            t: Time steps
            target_velocity: Target velocity u_t
            **kwargs: Additional arguments

        Returns:
            loss: Flow matching loss
            velocity_pred: Predicted velocity (for logging)
        """
        # Predict velocity
        velocity_pred = self.forward(state, x_t, goal, t, **kwargs)

        # Compute MSE loss
        loss = torch.mean((velocity_pred - target_velocity) ** 2)

        return loss, velocity_pred


class TimeConditionedFlowWrapper(FlowMatchingWrapper):
    """
    Flow wrapper with explicit time conditioning.
    Embeds time and adds it to the model's conditioning.
    """

    def __init__(
        self,
        inner_model,
        flow_type: str = 'conditional',
        sigma_min: float = 1e-4,
        sigma_data: float = 0.5,
        use_time_scaling: bool = False,
        time_embed_dim: int = 512,
    ):
        super().__init__(inner_model, flow_type, sigma_min, sigma_data, use_time_scaling)

        # Time embedding (similar to transformer positional encoding)
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time into a higher dimensional space.

        Args:
            t: Time in [0, 1], shape (B,)

        Returns:
            Time embeddings, shape (B, time_embed_dim)
        """
        t_input = t.view(-1, 1)
        return self.time_mlp(t_input)

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        goal: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward with explicit time embedding added to goal conditioning.
        """
        # Embed time
        time_emb = self.embed_time(t)  # (B, time_embed_dim)

        # Add time embedding to goal (assuming goal has similar dimension)
        # If dimensions don't match, we concatenate instead
        if goal.shape[-1] == time_emb.shape[-1]:
            enhanced_goal = goal + time_emb.unsqueeze(1)
        else:
            # Concatenate along feature dimension
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, goal.shape[1], -1)
            enhanced_goal = torch.cat([goal, time_emb_expanded], dim=-1)

        # Call parent forward with enhanced goal
        return super().forward(state, x_t, enhanced_goal, t, **kwargs)
