import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class ConditionalFlowMatcher:
    """
    Conditional Flow Matching implementation based on:
    "Flow Matching for Generative Modeling" (Lipman et al., 2023)

    This implements Optimal Transport Conditional Flow Matching (OT-CFM)
    which uses optimal transport paths between data and noise distributions.
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 80.0,
        use_ot_flow: bool = True,
        noise_schedule: str = 'constant',
    ):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            use_ot_flow: If True, use Optimal Transport flow, else use independent flow
            noise_schedule: Type of noise schedule ('constant', 'exponential', 'linear')
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_ot_flow = use_ot_flow
        self.noise_schedule = noise_schedule

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent noise level sigma(t) based on schedule type.

        Args:
            t: Time steps in [0, 1], shape (B,)

        Returns:
            sigma_t: Noise levels, shape (B,)
        """
        if self.noise_schedule == 'constant':
            return torch.full_like(t, self.sigma_min)
        elif self.noise_schedule == 'exponential':
            # Exponential schedule: sigma(t) = sigma_max^(1-t) * sigma_min^t
            return self.sigma_max ** (1 - t) * self.sigma_min ** t
        elif self.noise_schedule == 'linear':
            # Linear schedule: sigma(t) = (1-t) * sigma_max + t * sigma_min
            return (1 - t) * self.sigma_max + t * self.sigma_min
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample time steps uniformly from [0, 1]

        Args:
            batch_size: Number of samples
            device: Device to create tensor on

        Returns:
            Time steps of shape (batch_size,)
        """
        return torch.rand(batch_size, device=device)

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        sigma_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the conditional flow path x_t and velocity u_t.

        For Optimal Transport path:
            x_t = t * x1 + (1 - t) * x0 + sigma_t * noise
            u_t = x1 - x0

        Args:
            x0: Source samples (noise), shape (B, T, D)
            x1: Target samples (data), shape (B, T, D)
            t: Time steps, shape (B,)
            sigma_t: Optional time-dependent noise level

        Returns:
            x_t: Interpolated samples at time t
            u_t: Target velocity field
        """
        # Expand time dimensions to match data
        t_expanded = t.view(-1, 1, 1)  # (B, 1, 1)

        # Compute noise level from schedule if not provided
        if sigma_t is None:
            sigma_t = self.get_noise_schedule(t)

        if self.use_ot_flow:
            # Optimal Transport path: straight line interpolation
            mu_t = t_expanded * x1 + (1 - t_expanded) * x0

            # Add Gaussian noise according to schedule
            if isinstance(sigma_t, float):
                noise = torch.randn_like(x1) * sigma_t
            else:
                sigma_t_expanded = sigma_t.view(-1, 1, 1)
                noise = torch.randn_like(x1) * sigma_t_expanded

            x_t = mu_t + noise

            # Target velocity is the direction from x0 to x1
            u_t = x1 - x0
        else:
            # Independent (variance preserving) flow
            mu_t = t_expanded * x1

            # Use schedule-based noise level
            if isinstance(sigma_t, float):
                sigma_t_val = sigma_t
            else:
                sigma_t_val = sigma_t.view(-1, 1, 1)

            noise = torch.randn_like(x1)
            x_t = mu_t + sigma_t_val * noise

            # Target velocity
            if isinstance(sigma_t_val, float):
                u_t = (x1 - sigma_t_val * noise) / (1 - sigma_t_val).clamp(min=1e-5)
            else:
                u_t = (x1 - sigma_t_val * noise) / (1 - sigma_t_val).clamp(min=1e-5)

        return x_t, u_t

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the flow matching loss.

        Loss = E_t,x0,x1 [||v_θ(x_t, t) - u_t||^2]

        Args:
            model_output: Predicted velocity from model, shape (B, T, D)
            target_velocity: Target velocity u_t, shape (B, T, D)

        Returns:
            Loss scalar
        """
        return torch.mean((model_output - target_velocity) ** 2)


class TimeScaledFlowMatcher(ConditionalFlowMatcher):
    """
    Flow Matching with time-dependent scaling for improved training stability.
    Uses a similar scaling strategy as EDM but adapted for flow matching.
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_data: float = 0.5,
        use_ot_flow: bool = True,
    ):
        super().__init__(sigma_min, use_ot_flow)
        self.sigma_data = sigma_data

    def get_time_scalings(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time-dependent scalings for input and output.

        Args:
            t: Time steps, shape (B,)

        Returns:
            c_in: Input scaling
            c_out: Output scaling
        """
        # Convert time [0,1] to sigma-like scale
        sigma_t = (1 - t) * 80.0 + t * self.sigma_min

        # EDM-style scaling
        c_in = 1.0 / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)
        c_out = sigma_t * self.sigma_data / torch.sqrt(sigma_t ** 2 + self.sigma_data ** 2)

        return c_in, c_out


class RectifiedFlowMatcher:
    """
    Rectified Flow implementation based on:
    "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (Liu et al., 2022)

    This is a simpler variant that learns straight paths directly.
    """

    def __init__(self, num_reflow_iterations: int = 1):
        """
        Args:
            num_reflow_iterations: Number of reflow iterations for straightening paths
        """
        self.num_reflow_iterations = num_reflow_iterations

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time uniformly from [0, 1]"""
        return torch.rand(batch_size, device=device)

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute straight-line interpolation between x0 and x1.

        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0

        Args:
            x0: Source samples, shape (B, T, D)
            x1: Target samples, shape (B, T, D)
            t: Time steps, shape (B,)

        Returns:
            x_t: Interpolated samples
            u_t: Constant velocity
        """
        t_expanded = t.view(-1, 1, 1)

        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        u_t = x1 - x0

        return x_t, u_t

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE between predicted and target velocity"""
        return torch.mean((model_output - target_velocity) ** 2)
