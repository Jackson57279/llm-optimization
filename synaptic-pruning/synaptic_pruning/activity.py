"""Activity tracking module for Synaptic Pruning.

This module implements the EMAActivity tracker that monitors per-weight activity
using exponential moving averages with configurable decay rates.
"""

from typing import Any

import torch


class EMAActivity:
    """Exponential Moving Average activity tracker for neural network weights.

    Tracks per-weight activity scores using EMA with configurable decay.
    Weights with consistent high activity have EMA approaching 1.0.
    Inactive weights decay toward 0.0 according to the decay rate.

    The activity is computed based on gradient magnitude. When gradients flow
    through a weight, its activity increases. When no gradients are present,
    the activity decays exponentially.

    Attributes:
        decay: EMA decay factor (0 < decay < 1). Higher = slower decay.
        hot_threshold: Threshold for "hot" (high activity) tier.
        warm_threshold: Threshold for "warm" (medium activity) tier.
        activity_scores: Dictionary mapping parameter names to EMA tensors.
    """

    def __init__(
        self,
        decay: float = 0.9,
        hot_threshold: float = 0.8,
        warm_threshold: float = 0.3,
    ) -> None:
        """Initialize EMA activity tracker.

        Args:
            decay: EMA decay factor. Default 0.9 means 90% history, 10% new.
                   Must be between 0 and 1.
            hot_threshold: Threshold for "hot" (high activity) tier.
                           Must be between warm_threshold and 1.
            warm_threshold: Threshold for "warm" (medium activity) tier.
                            Must be between 0 and hot_threshold.

        Raises:
            ValueError: If decay is not in (0, 1) or if thresholds are invalid.
        """
        if not 0 < decay < 1:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        if not 0 < warm_threshold < hot_threshold < 1:
            raise ValueError(
                f"thresholds must satisfy 0 < warm < hot < 1, "
                f"got warm={warm_threshold}, hot={hot_threshold}"
            )

        self.decay: float = decay
        self.hot_threshold: float = hot_threshold
        self.warm_threshold: float = warm_threshold
        self.activity_scores: dict[str, torch.Tensor] = {}

    def update(self, param_name: str, gradients: torch.Tensor) -> None:
        """Update activity scores based on gradient information.

        Activity is computed as the normalized gradient magnitude.
        The EMA formula is: new_score = decay * old_score + (1 - decay) * activity

        Args:
            param_name: Name of the parameter being updated.
            gradients: Gradient tensor for the parameter.

        Raises:
            ValueError: If gradients contain NaN or Inf values.
        """
        if not torch.isfinite(gradients).all():
            raise ValueError(f"Gradients for {param_name} contain NaN or Inf values")

        # Compute current activity from gradient magnitude
        # Normalize by the max gradient magnitude to get [0, 1] range
        grad_norm = gradients.abs()
        max_grad = grad_norm.max()

        if max_grad > 0:
            # Normalize to [0, 1] based on maximum gradient in this batch
            current_activity = grad_norm / max_grad
        else:
            # All gradients are zero
            current_activity = torch.zeros_like(grad_norm)

        # Initialize or update EMA
        if param_name not in self.activity_scores:
            # First update - initialize with current activity
            self.activity_scores[param_name] = current_activity.clone()
        else:
            # EMA update: new = decay * old + (1 - decay) * current
            old_score = self.activity_scores[param_name]
            # Ensure shapes match (in case parameter shape changed)
            if old_score.shape != current_activity.shape:
                # Reinitialize for new shape
                self.activity_scores[param_name] = current_activity.clone()
            else:
                # Move old_score to same device as current_activity if needed
                if old_score.device != current_activity.device:
                    old_score = old_score.to(current_activity.device)
                new_score = self.decay * old_score + (1 - self.decay) * current_activity
                self.activity_scores[param_name] = new_score

    def get_activity(self, param_name: str) -> torch.Tensor:
        """Get current activity scores for a parameter.

        Args:
            param_name: Name of the parameter.

        Returns:
            Tensor of activity scores matching parameter shape.
            Returns zeros if parameter has not been seen before.

        Raises:
            KeyError: If param_name has not been registered yet.
        """
        if param_name not in self.activity_scores:
            raise KeyError(
                f"Parameter '{param_name}' has not been registered. " f"Call update() first."
            )
        return self.activity_scores[param_name]

    def get_tier_mask(self, param_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get tier classification masks for a parameter.

        Tiers are classified based on activity thresholds:
        - Hot: activity > hot_threshold
        - Warm: warm_threshold < activity <= hot_threshold
        - Cold: activity <= warm_threshold

        Args:
            param_name: Name of the parameter.

        Returns:
            Tuple of (hot_mask, warm_mask, cold_mask) boolean tensors.

        Raises:
            KeyError: If param_name has not been registered yet.
        """
        activity = self.get_activity(param_name)

        hot_mask = activity > self.hot_threshold
        warm_mask = (activity > self.warm_threshold) & ~hot_mask
        cold_mask = ~hot_mask & ~warm_mask

        return hot_mask, warm_mask, cold_mask

    def get_tier_counts(self, param_name: str) -> tuple[int, int, int]:
        """Get count of weights in each tier.

        Args:
            param_name: Name of the parameter.

        Returns:
            Tuple of (hot_count, warm_count, cold_count).

        Raises:
            KeyError: If param_name has not been registered yet.
        """
        hot_mask, warm_mask, cold_mask = self.get_tier_mask(param_name)

        hot_count = int(hot_mask.sum().item())
        warm_count = int(warm_mask.sum().item())
        cold_count = int(cold_mask.sum().item())

        return hot_count, warm_count, cold_count

    def reset(self) -> None:
        """Reset all activity scores. Clears the internal state."""
        self.activity_scores.clear()

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing activity_scores and configuration.
        """
        return {
            "activity_scores": self.activity_scores,
            "decay": torch.tensor(self.decay),
            "hot_threshold": torch.tensor(self.hot_threshold),
            "warm_threshold": torch.tensor(self.warm_threshold),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.activity_scores = state_dict["activity_scores"]
        self.decay = float(state_dict["decay"].item())
        self.hot_threshold = float(state_dict["hot_threshold"].item())
        self.warm_threshold = float(state_dict["warm_threshold"].item())
