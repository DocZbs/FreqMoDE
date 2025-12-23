import torch
import torch.nn as nn
from typing import Optional


class FlowMatchingSampler:
    """
    Sampler for Flow Matching models.
    Implements ODE integration to generate samples from learned flow.
    """

    def __init__(
        self,
        num_steps: int = 50,
        method: str = 'euler',
    ):
        """
        Args:
            num_steps: Number of integration steps
            method: Integration method ('euler', 'rk4', 'dopri5')
        """
        self.num_steps = num_steps
        self.method = method

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        state: torch.Tensor,
        x_init: torch.Tensor,
        goal: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Euler method for ODE integration.

        dx/dt = v_θ(x_t, t, condition)
        x_{t+dt} = x_t + dt * v_θ(x_t, t, condition)

        Args:
            model: Flow model that predicts velocity
            state: State observations
            x_init: Initial noise, shape (B, T, D)
            goal: Goal conditioning
            num_steps: Number of steps (overrides default)

        Returns:
            Generated samples at t=1
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        x = x_init
        for step in range(steps):
            t = torch.ones(x.shape[0], device=x.device) * (step / steps)

            # Predict velocity
            v = model(state, x, goal, t)

            # Euler step
            x = x + dt * v

        return x

    @torch.no_grad()
    def sample_heun(
        self,
        model: nn.Module,
        state: torch.Tensor,
        x_init: torch.Tensor,
        goal: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Heun's method (2nd order Runge-Kutta) for ODE integration.
        More accurate than Euler with minimal overhead.

        Args:
            model: Flow model
            state: State observations
            x_init: Initial noise
            goal: Goal conditioning
            num_steps: Number of steps

        Returns:
            Generated samples
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        x = x_init
        for step in range(steps):
            t = torch.ones(x.shape[0], device=x.device) * (step / steps)
            t_next = torch.ones(x.shape[0], device=x.device) * ((step + 1) / steps)

            # First evaluation
            v1 = model(state, x, goal, t)

            # Predictor step
            x_pred = x + dt * v1

            # Second evaluation
            v2 = model(state, x_pred, goal, t_next)

            # Corrector step
            x = x + dt * (v1 + v2) / 2

        return x

    @torch.no_grad()
    def sample_rk4(
        self,
        model: nn.Module,
        state: torch.Tensor,
        x_init: torch.Tensor,
        goal: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        4th order Runge-Kutta method.
        Higher accuracy but requires 4 model evaluations per step.

        Args:
            model: Flow model
            state: State observations
            x_init: Initial noise
            goal: Goal conditioning
            num_steps: Number of steps

        Returns:
            Generated samples
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        x = x_init
        for step in range(steps):
            t = torch.ones(x.shape[0], device=x.device) * (step / steps)

            # RK4 stages
            k1 = model(state, x, goal, t)

            t_half = t + dt / 2
            k2 = model(state, x + dt * k1 / 2, goal, t_half)

            k3 = model(state, x + dt * k2 / 2, goal, t_half)

            t_next = t + dt
            k4 = model(state, x + dt * k3, goal, t_next)

            # Weighted combination
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x

    @torch.no_grad()
    def sample_midpoint(
        self,
        model: nn.Module,
        state: torch.Tensor,
        x_init: torch.Tensor,
        goal: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Midpoint method (2nd order explicit).
        Good balance between accuracy and computation.

        Args:
            model: Flow model
            state: State observations
            x_init: Initial noise
            goal: Goal conditioning
            num_steps: Number of steps

        Returns:
            Generated samples
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        x = x_init
        for step in range(steps):
            t = torch.ones(x.shape[0], device=x.device) * (step / steps)
            t_mid = t + dt / 2

            # Evaluate at current point
            v1 = model(state, x, goal, t)

            # Predict midpoint
            x_mid = x + (dt / 2) * v1

            # Evaluate at midpoint
            v_mid = model(state, x_mid, goal, t_mid)

            # Full step using midpoint velocity
            x = x + dt * v_mid

        return x

    def sample(
        self,
        model: nn.Module,
        state: torch.Tensor,
        x_init: torch.Tensor,
        goal: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Main sampling interface that dispatches to the appropriate method.

        Args:
            model: Flow model
            state: State observations
            x_init: Initial noise
            goal: Goal conditioning
            num_steps: Number of steps

        Returns:
            Generated samples
        """
        if self.method == 'euler':
            return self.sample_euler(model, state, x_init, goal, num_steps)
        elif self.method == 'heun':
            return self.sample_heun(model, state, x_init, goal, num_steps)
        elif self.method == 'rk4':
            return self.sample_rk4(model, state, x_init, goal, num_steps)
        elif self.method == 'midpoint':
            return self.sample_midpoint(model, state, x_init, goal, num_steps)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
