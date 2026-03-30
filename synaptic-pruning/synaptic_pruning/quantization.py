"""Quantization module for Synaptic Pruning.

This module implements tiered quantization with multiple precision levels:
- FP16: Full precision for hot (active) weights
- 4-bit: Reduced precision for warm weights
- 1-bit: Binary representation for cold weights
"""

from typing import Any

import torch


class _STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for quantization.
    
    Forward pass: apply quantization function
    Backward pass: pass gradients straight through (identity)
    
    This connects the quantized output back to the input weights
    for proper gradient flow during training.
    """
    
    @staticmethod
    def forward(ctx, x, quantized):
        """Forward pass returns quantized values.
        
        Args:
            ctx: Context for saving tensors for backward
            x: Original input tensor (for gradient flow)
            quantized: Quantized values (what we return)
        """
        ctx.save_for_backward(x)
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass passes gradients straight through to original input."""
        (x,) = ctx.saved_tensors
        # Pass gradients straight through
        return grad_output, None


class _STEDequantize(torch.autograd.Function):
    """Straight-Through Estimator for dequantization.
    
    Forward pass: apply dequantization
    Backward pass: pass gradients straight through
    """
    
    @staticmethod
    def forward(ctx, x):
        """Forward pass returns dequantized values."""
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass passes gradients straight through."""
        return grad_output


class TieredQuantizer:
    """Multi-tier quantizer with FP16, 4-bit, and 1-bit precision levels.

    Implements differentiable quantization using Straight-Through Estimator (STE)
    for training with quantized weights.

    Attributes:
        hot_threshold: Activity threshold for FP16 tier.
        warm_threshold: Activity threshold for 4-bit tier.
        scales: Dictionary of quantization scales per parameter.
    """

    def __init__(
        self,
        hot_threshold: float = 0.8,
        warm_threshold: float = 0.3,
    ) -> None:
        """Initialize tiered quantizer.

        Args:
            hot_threshold: Minimum activity for FP16 tier.
            warm_threshold: Minimum activity for 4-bit tier.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if not 0 < warm_threshold < hot_threshold < 1:
            raise ValueError(
                f"thresholds must satisfy 0 < warm < hot < 1, "
                f"got warm={warm_threshold}, hot={hot_threshold}"
            )

        self.hot_threshold: float = hot_threshold
        self.warm_threshold: float = warm_threshold
        self.scales: dict[str, torch.Tensor] = {}

    def quantize_fp16(self, weights: torch.Tensor) -> torch.Tensor:
        """Keep weights in FP16 (no quantization).

        Args:
            weights: Input weight tensor.

        Returns:
            FP16 weight tensor (unchanged).
        """
        return weights

    def quantize_4bit(
        self, 
        weights: torch.Tensor, 
        scale: torch.Tensor | None = None,
        param_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weights to 4-bit representation.

        Uses symmetric quantization with STE for differentiability.
        4-bit signed range: -7 to +7 (to ensure round-trip stability)

        Args:
            weights: Input weight tensor.
            scale: Optional quantization scale. Computed from max abs weight if not provided.
            param_name: Optional parameter name for storing scale.

        Returns:
            Tuple of (quantized_int8, scale, quantized_with_ste). 
            - quantized_int8: int8 storage for serialization
            - scale: quantization scale
            - quantized_with_ste: differentiable tensor with STE applied
        """
        # Compute scale if not provided
        if scale is None:
            max_abs = weights.abs().max()
            # Avoid division by zero
            if max_abs > 0:
                scale = max_abs / 7.0  # 4-bit signed max value is 7
            else:
                scale = torch.tensor(1.0, device=weights.device, dtype=weights.dtype)
        else:
            scale = scale.to(weights.device, weights.dtype)
        
        # Quantize: round to nearest 4-bit level
        # Map weights to [-7, 7] range
        normalized = weights / scale
        # Clamp to valid 4-bit signed range (-7 to +7 for stability)
        normalized = torch.clamp(normalized, -7, 7)
        # Round to nearest integer for discrete values
        quantized_float = torch.round(normalized)
        # Convert to int8 for storage
        quantized_int = quantized_float.to(torch.int8)
        
        # Apply STE to create differentiable version
        # This connects quantized values back to original weights
        quantized_with_ste = _STEQuantize.apply(weights, quantized_float * scale)
        
        # Store scale for this parameter
        if param_name is not None:
            self.scales[param_name] = scale.detach().clone()
        
        return quantized_int, scale, quantized_with_ste

    def dequantize_4bit(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize 4-bit weights back to FP16.

        Args:
            quantized: 4-bit quantized weights (int8 or float with STE).
            scale: Quantization scale.

        Returns:
            Dequantized FP16 weights.
        """
        # If quantized has STE attached (float tensor), return as-is
        if quantized.dtype != torch.int8:
            # Already dequantized with STE - just return
            return quantized
        
        # Convert from int8 to float and multiply by scale
        weights_float = quantized.to(torch.float32) * scale
        return weights_float

    def quantize_1bit(
        self, 
        weights: torch.Tensor, 
        scale: torch.Tensor | None = None,
        param_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weights to 1-bit (binary) representation.

        Produces values in {-scale, +scale} with STE for differentiability.

        Args:
            weights: Input weight tensor.
            scale: Optional quantization scale. Computed from mean abs if not provided.
            param_name: Optional parameter name for storing scale.

        Returns:
            Tuple of (quantized_int8, scale, quantized_with_ste). 
            - quantized_int8: int8 storage with values -1 or +1
            - scale: quantization scale
            - quantized_with_ste: differentiable tensor with STE applied
        """
        # Compute scale if not provided
        if scale is None:
            mean_abs = weights.abs().mean()
            # Avoid division by zero
            if mean_abs > 0:
                scale = mean_abs
            else:
                scale = torch.tensor(1.0, device=weights.device, dtype=weights.dtype)
        else:
            scale = scale.to(weights.device, weights.dtype)
        
        # Binary quantization: sign of weight
        # Returns -1 or +1
        binary_float = torch.sign(weights)
        # Handle zeros - map to +1
        binary_float[binary_float == 0] = 1.0
        
        # Convert to int8 for storage
        quantized_int = binary_float.to(torch.int8)
        
        # Apply STE to create differentiable version
        # This connects quantized values back to original weights
        quantized_with_ste = _STEQuantize.apply(weights, binary_float * scale)
        
        # Store scale for this parameter
        if param_name is not None:
            self.scales[param_name] = scale.detach().clone()
        
        return quantized_int, scale, quantized_with_ste

    def dequantize_1bit(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize 1-bit weights back to FP16.

        Args:
            quantized: 1-bit quantized weights (can be int8 or float with STE).
            scale: Quantization scale.

        Returns:
            Dequantized FP16 weights.
        """
        # If quantized has STE attached (float tensor), return as-is
        if quantized.dtype != torch.int8:
            # Already dequantized with STE - just return
            return quantized
        
        # Convert from int8 to float and multiply by scale
        weights_float = quantized.to(torch.float32) * scale
        return weights_float

    def apply_tiered_quantization(
        self,
        weights: torch.Tensor,
        activity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply tiered quantization based on activity scores.

        Assigns weights to tiers based on activity:
        - Hot (activity > hot_threshold): FP16, no quantization
        - Warm (warm_threshold < activity <= hot_threshold): 4-bit
        - Cold (activity <= warm_threshold): 1-bit

        Args:
            weights: Input weight tensor.
            activity: Activity scores for each weight (same shape as weights).

        Returns:
            Tuple of (quantized_weights, metadata) where metadata contains
            tier masks and scales.
        """
        # Create tier masks
        hot_mask = activity > self.hot_threshold
        warm_mask = (activity > self.warm_threshold) & ~hot_mask
        cold_mask = ~hot_mask & ~warm_mask
        
        # Initialize result tensor
        result = torch.zeros_like(weights)
        
        # Process hot weights (FP16 - no quantization)
        if hot_mask.any():
            hot_weights = weights[hot_mask]
            hot_quantized = self.quantize_fp16(hot_weights)
            result = result.masked_scatter(hot_mask, hot_quantized)
        
        # Process warm weights (4-bit)
        warm_scale = None
        if warm_mask.any():
            warm_weights = weights[warm_mask]
            # Use the STE-enabled quantized version directly
            _, warm_scale, warm_dequantized = self.quantize_4bit(warm_weights)
            result = result.masked_scatter(warm_mask, warm_dequantized)
        
        # Process cold weights (1-bit)
        cold_scale = None
        if cold_mask.any():
            cold_weights = weights[cold_mask]
            # Use the STE-enabled quantized version directly
            _, cold_scale, cold_dequantized = self.quantize_1bit(cold_weights)
            result = result.masked_scatter(cold_mask, cold_dequantized)
        
        # Build metadata
        metadata = {
            "hot_mask": hot_mask,
            "warm_mask": warm_mask,
            "cold_mask": cold_mask,
            "scales": {
                "warm": warm_scale,
                "cold": cold_scale,
            },
        }
        
        return result, metadata

    def reset(self) -> None:
        """Reset all stored scales. Clears the internal state."""
        self.scales.clear()

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing scales and configuration.
        """
        return {
            "scales": self.scales,
            "hot_threshold": torch.tensor(self.hot_threshold),
            "warm_threshold": torch.tensor(self.warm_threshold),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.scales = state_dict["scales"]
        self.hot_threshold = float(state_dict["hot_threshold"].item())
        self.warm_threshold = float(state_dict["warm_threshold"].item())
