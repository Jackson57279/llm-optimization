"""Utility functions for Synaptic Pruning.

Visualization, metrics, and helper functions.
"""

from typing import Any

import torch
import torch.nn as nn


def get_compression_stats(model: nn.Module) -> dict[str, Any]:
    """Calculate compression statistics for a model.

    Computes:
    - Raw parameter count ratio
    - Effective storage ratio considering sparsity and quantization
    - Theoretical ratio including recovery overhead

    Args:
        model: Model with SynapticLayers.

    Returns:
        Dictionary with compression metrics.
    """
    raise NotImplementedError("get_compression_stats will be implemented with layer-synaptic-layer")


def visualize_activity(
    activity_scores: torch.Tensor,
    save_path: str | None = None,
    title: str = "Activity Distribution",
) -> None:
    """Visualize activity scores as histogram.

    Args:
        activity_scores: Tensor of activity scores.
        save_path: Optional path to save figure.
        title: Plot title.
    """
    raise NotImplementedError(
        "visualize_activity will be implemented in foundation-activity-visualization"
    )


def plot_tier_distribution(
    tier_counts: dict[str, tuple[int, int, int]],
    save_path: str | None = None,
) -> None:
    """Plot tier distribution across layers.

    Args:
        tier_counts: Dictionary mapping layer names to (hot, warm, cold) counts.
        save_path: Optional path to save figure.
    """
    raise NotImplementedError(
        "plot_tier_distribution will be implemented in foundation-activity-visualization"
    )


def plot_layer_heatmap(
    activity_scores: torch.Tensor,
    layer_name: str = "layer",
    save_path: str | None = None,
) -> None:
    """Plot activity heatmap for a layer's weights.

    Args:
        activity_scores: 2D tensor of activity scores.
        layer_name: Name of the layer for title.
        save_path: Optional path to save figure.
    """
    raise NotImplementedError(
        "plot_layer_heatmap will be implemented in foundation-activity-visualization"
    )


def replace_linear_with_synaptic(model: nn.Module, **synaptic_kwargs: Any) -> nn.Module:
    """Replace all nn.Linear layers with SynapticLayers.

    Args:
        model: Model to modify.
        **synaptic_kwargs: Arguments to pass to SynapticLayer.

    Returns:
        Modified model.
    """
    raise NotImplementedError(
        "replace_linear_with_synaptic will be implemented in layer-synaptic-layer"
    )


def calculate_sparsity(tensor: torch.Tensor) -> float:
    """Calculate sparsity (fraction of zero or near-zero values).

    Args:
        tensor: Input tensor.

    Returns:
        Fraction of values near zero.
    """
    raise NotImplementedError(
        "calculate_sparsity will be implemented with quantization-tiered-quantizer"
    )
