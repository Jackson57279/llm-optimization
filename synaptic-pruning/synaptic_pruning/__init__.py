"""
Synaptic Pruning: Activity-Driven Sparse Quantization with Learned Recovery.

A novel neural network compression framework that achieves extreme compression
by combining activity-driven sparsity with multi-tier quantization and a learned
recovery mechanism.
"""

__version__ = "0.1.0"

# Main components
from synaptic_pruning.activity import EMAActivity
from synaptic_pruning.visualization import (
    plot_activity_histogram,
    plot_activity_summary,
    plot_layer_heatmap,
    plot_tier_distribution,
    save_visualization,
)

# from synaptic_pruning.quantization import TieredQuantizer
# from synaptic_pruning.recovery import HyperNetwork, CodebookVQ
# from synaptic_pruning.layers import SynapticLayer
# from synaptic_pruning.training import SynapticTrainer
# from synaptic_pruning.utils import get_compression_stats, visualize_activity

__all__ = [
    "__version__",
    "EMAActivity",
    "plot_activity_histogram",
    "plot_tier_distribution",
    "plot_layer_heatmap",
    "plot_activity_summary",
    "save_visualization",
    # "TieredQuantizer",
    # "HyperNetwork",
    # "CodebookVQ",
    # "SynapticLayer",
    # "SynapticTrainer",
    # "get_compression_stats",
    # "visualize_activity",
]
