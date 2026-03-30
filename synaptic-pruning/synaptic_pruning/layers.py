"""SynapticLayer - unified layer combining all components.

This module implements the SynapticLayer class, a drop-in replacement for nn.Linear
that combines:
- Activity tracking (EMAActivity)
- Tiered quantization (TieredQuantizer)
- Recovery mechanism integration
"""

from typing import Any

import torch
import torch.nn as nn

from synaptic_pruning.activity import EMAActivity
from synaptic_pruning.quantization import TieredQuantizer


class SynapticLayer(nn.Module):
    """Drop-in replacement for nn.Linear with activity-driven quantization.

    Combines activity tracking, tiered quantization, and recovery mechanisms
    into a single layer that can replace nn.Linear in any PyTorch model.

    The forward pass applies tiered quantization to weights based on their
    activity levels:
    - Hot weights (activity > hot_threshold): FP16, no quantization
    - Warm weights (warm_threshold < activity <= hot_threshold): 4-bit
    - Cold weights (activity <= warm_threshold): 1-bit

    Straight-Through Estimator (STE) allows gradients to flow through
    quantization during training.

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        weight: The learnable weights of shape (out_features, in_features).
        bias: The learnable bias of shape (out_features).
        activity_tracker: EMAActivity instance for tracking weight activity.
        quantizer: TieredQuantizer for applying precision tiers.

    Example:
        >>> layer = SynapticLayer(768, 3072)
        >>> x = torch.randn(4, 768)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([4, 3072])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        decay: float = 0.9,
        hot_threshold: float = 0.8,
        warm_threshold: float = 0.3,
    ) -> None:
        """Initialize SynapticLayer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias. Default: True.
            decay: EMA decay factor for activity tracking. Default: 0.9.
            hot_threshold: Activity threshold for FP16 tier. Default: 0.8.
            warm_threshold: Activity threshold for 4-bit tier. Default: 0.3.

        Raises:
            ValueError: If in_features or out_features is not positive.
        """
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self._reset_parameters()

        # Initialize activity tracker and quantizer with matching thresholds
        self.activity_tracker = EMAActivity(
            decay=decay,
            hot_threshold=hot_threshold,
            warm_threshold=warm_threshold,
        )
        self.quantizer = TieredQuantizer(
            hot_threshold=hot_threshold,
            warm_threshold=warm_threshold,
        )

        # Register backward hook on the weight parameter
        self._register_backward_hook()

    def _reset_parameters(self) -> None:
        """Initialize weights and bias.

        Uses the same initialization as nn.Linear for compatibility.
        """
        # Xavier uniform initialization like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in", nonlinearity="linear")
        if self.bias is not None:
            # Bias initialized to zero
            nn.init.zeros_(self.bias)

    def _register_backward_hook(self) -> None:
        """Register backward hook on weight to track activity."""
        self.weight.register_hook(self._weight_backward_hook)

    def _weight_backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Hook called during backward pass to update activity.

        Args:
            grad: Gradient tensor for the weight.

        Returns:
            The gradient unchanged (pass-through).
        """
        self.activity_tracker.update("weight", grad)
        return grad

    def _get_quantized_weight(self) -> torch.Tensor:
        """Get weight tensor with tiered quantization applied.

        Returns:
            Quantized weight tensor. If activity scores exist, applies
            tiered quantization. Otherwise returns weights unchanged.
        """
        # Check if we have activity scores for this weight
        if "weight" not in self.activity_tracker.activity_scores:
            # No activity tracking yet, return weights unchanged
            return self.weight

        # Get activity scores
        activity = self.activity_tracker.get_activity("weight")

        # Apply tiered quantization
        quantized_weight, _ = self.quantizer.apply_tiered_quantization(
            self.weight, activity
        )

        return quantized_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tiered quantization.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        # Get quantized weights
        weight = self._get_quantized_weight()

        # Handle different input dimensions
        input_shape = x.shape

        # Flatten all dimensions except last (features) for matmul
        if input_shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input features {self.in_features}, got {input_shape[-1]}"
            )

        # Compute output: x @ W^T + b
        # PyTorch handles various input shapes automatically with matmul
        output = torch.matmul(x, weight.t())

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for printing."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def state_dict(self, *, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> dict[str, Any]:
        """Get state dictionary including activity tracker and quantizer state.

        Args:
            destination: Optional dict to store state.
            prefix: Prefix for state keys.
            keep_vars: Whether to keep variables as Variables.

        Returns:
            State dictionary with all components.
        """
        # Get base nn.Module state
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Add activity tracker state
        state[f"{prefix}activity_tracker"] = self.activity_tracker.state_dict()

        # Add quantizer state
        state[f"{prefix}quantizer"] = self.quantizer.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False) -> Any:
        """Load state dictionary including activity tracker and quantizer state.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce state matching.
            assign: Whether to assign the state directly (PyTorch 2.0+).

        Returns:
            NamedTuple with missing_keys and unexpected_keys if strict=True.
        """
        # Extract component states
        activity_tracker_key = "activity_tracker"
        quantizer_key = "quantizer"

        if activity_tracker_key in state_dict:
            self.activity_tracker.load_state_dict(state_dict[activity_tracker_key])
            del state_dict[activity_tracker_key]

        if quantizer_key in state_dict:
            self.quantizer.load_state_dict(state_dict[quantizer_key])
            del state_dict[quantizer_key]

        # Load base nn.Module state
        return super().load_state_dict(state_dict, strict, assign)

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics for this layer.

        Returns:
            Dictionary with compression statistics:
            - total_params: Total number of parameters
            - hot_count: Number of hot weights
            - warm_count: Number of warm weights
            - cold_count: Number of cold weights
            - compression_ratio: Effective compression ratio
        """
        stats = {
            "total_params": self.out_features * self.in_features,
            "hot_count": 0,
            "warm_count": 0,
            "cold_count": 0,
            "hot_ratio": 0.0,
            "warm_ratio": 0.0,
            "cold_ratio": 0.0,
        }

        if "weight" in self.activity_tracker.activity_scores:
            hot, warm, cold = self.activity_tracker.get_tier_counts("weight")
            total = hot + warm + cold

            stats["hot_count"] = hot
            stats["warm_count"] = warm
            stats["cold_count"] = cold

            if total > 0:
                stats["hot_ratio"] = hot / total
                stats["warm_ratio"] = warm / total
                stats["cold_ratio"] = cold / total

        return stats
