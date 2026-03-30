"""Activity visualization module for Synaptic Pruning.

This module provides visualization tools for analyzing weight activity patterns
across neural network layers. It generates histograms, tier distribution charts,
and layer heatmaps to help understand and debug activity tracking behavior.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure

from synaptic_pruning.activity import EMAActivity


def plot_activity_histogram(
    activity_scores: Union[Dict[str, torch.Tensor], EMAActivity],
    param_names: Optional[List[str]] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Activity Score Distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot histogram of activity scores across weights.

    Creates a histogram showing the distribution of EMA activity scores
    across all tracked parameters or a selected subset. Useful for
    understanding how weight activity is distributed in the model.

    Args:
        activity_scores: Either an EMAActivity tracker instance or a dictionary
            mapping parameter names to activity score tensors.
        param_names: Optional list of parameter names to include. If None,
            all parameters are included.
        bins: Number of histogram bins. Default 50.
        figsize: Figure size as (width, height). Default (10, 6).
        title: Plot title. Default "Activity Score Distribution".
        ax: Optional matplotlib Axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib Axes object containing the histogram.

    Raises:
        ValueError: If no valid activity scores are found.
        TypeError: If activity_scores is not a valid type.

    Example:
        >>> tracker = EMAActivity()
        >>> # ... update tracker with gradients ...
        >>> ax = plot_activity_histogram(tracker)
        >>> plt.savefig('activity_histogram.png')
    """
    # Extract activity scores from EMAActivity or use dict directly
    if isinstance(activity_scores, EMAActivity):
        scores_dict = activity_scores.activity_scores
    elif isinstance(activity_scores, dict):
        scores_dict = activity_scores
    else:
        raise TypeError(
            f"activity_scores must be EMAActivity or Dict[str, Tensor], "
            f"got {type(activity_scores)}"
        )

    if not scores_dict:
        raise ValueError("No activity scores found. Update tracker first.")

    # Filter to selected parameters if specified
    if param_names is not None:
        scores_dict = {k: v for k, v in scores_dict.items() if k in param_names}
        if not scores_dict:
            raise ValueError(f"No matching parameters found for {param_names}")

    # Flatten all scores into a single array
    all_scores = []
    for name, scores in scores_dict.items():
        all_scores.append(scores.detach().cpu().numpy().flatten())

    if not all_scores:
        raise ValueError("No valid activity scores to plot.")

    all_scores = np.concatenate(all_scores)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(all_scores, bins=bins, range=(0, 1), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Activity Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    mean_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
    ax.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
    ax.legend()

    return ax


def plot_tier_distribution(
    activity_tracker: EMAActivity,
    param_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Tier Distribution by Layer",
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot tier distribution (hot/warm/cold) for each tracked layer.

    Creates a stacked bar chart showing the number or percentage of weights
    in each activity tier (hot, warm, cold) for each tracked parameter.
    Useful for understanding per-layer sparsity patterns.

    Args:
        activity_tracker: EMAActivity tracker instance with updated scores.
        param_names: Optional list of parameter names to include. If None,
            all parameters are included.
        figsize: Figure size as (width, height). Default (12, 6).
        title: Plot title. Default "Tier Distribution by Layer".
        normalize: If True, show percentages; if False, show counts. Default True.
        ax: Optional matplotlib Axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib Axes object containing the tier distribution chart.

    Raises:
        ValueError: If no activity scores are tracked.
        TypeError: If activity_tracker is not an EMAActivity instance.

    Example:
        >>> tracker = EMAActivity(hot_threshold=0.8, warm_threshold=0.3)
        >>> # ... update tracker with gradients ...
        >>> ax = plot_tier_distribution(tracker)
        >>> plt.savefig('tier_distribution.png')
    """
    if not isinstance(activity_tracker, EMAActivity):
        raise TypeError(
            f"activity_tracker must be EMAActivity, got {type(activity_tracker)}"
        )

    if not activity_tracker.activity_scores:
        raise ValueError("No activity scores tracked. Update tracker first.")

    # Filter to selected parameters if specified
    all_params = list(activity_tracker.activity_scores.keys())
    if param_names is not None:
        params_to_plot = [p for p in param_names if p in all_params]
        if not params_to_plot:
            raise ValueError(f"No matching parameters found for {param_names}")
    else:
        params_to_plot = all_params

    # Get tier counts for each parameter
    hot_counts = []
    warm_counts = []
    cold_counts = []
    total_counts = []

    for param_name in params_to_plot:
        hot, warm, cold = activity_tracker.get_tier_counts(param_name)
        hot_counts.append(hot)
        warm_counts.append(warm)
        cold_counts.append(cold)
        total_counts.append(hot + warm + cold)

    # Normalize if requested
    if normalize:
        hot_vals = [h / t * 100 for h, t in zip(hot_counts, total_counts)]
        warm_vals = [w / t * 100 for w, t in zip(warm_counts, total_counts)]
        cold_vals = [c / t * 100 for c, t in zip(cold_counts, total_counts)]
        ylabel = 'Percentage (%)'
    else:
        hot_vals = hot_counts
        warm_vals = warm_counts
        cold_vals = cold_counts
        ylabel = 'Count'

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bar chart
    x = np.arange(len(params_to_plot))
    width = 0.6

    ax.bar(x, hot_vals, width, label='Hot', color='#ff6b6b', alpha=0.9)
    ax.bar(x, warm_vals, width, bottom=hot_vals, label='Warm', color='#ffd93d', alpha=0.9)
    ax.bar(
        x, cold_vals, width,
        bottom=[h + w for h, w in zip(hot_vals, warm_vals)],
        label='Cold', color='#6bcf7f', alpha=0.9
    )

    ax.set_xlabel('Layer/Parameter', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('.', '\n') for p in params_to_plot], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add threshold information
    threshold_text = (
        f"Thresholds: Hot>{activity_tracker.hot_threshold}, "
        f"Warm>{activity_tracker.warm_threshold}"
    )
    ax.text(0.02, 0.98, threshold_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_layer_heatmap(
    activity_tracker: EMAActivity,
    param_name: str,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a heatmap of activity scores for a specific layer.

    Creates a 2D heatmap visualization of activity scores for a single
    parameter (e.g., a weight matrix). This helps identify spatial patterns
    in weight activity, such as which neurons or input features are most active.

    Args:
        activity_tracker: EMAActivity tracker instance.
        param_name: Name of the parameter to visualize.
        figsize: Figure size as (width, height). If None, auto-calculated
            based on parameter shape.
        cmap: Colormap name. Default 'viridis'.
        title: Optional custom title. If None, uses param_name.
        ax: Optional matplotlib Axes to plot on. If None, creates new figure.

    Returns:
        The matplotlib Axes object containing the heatmap.

    Raises:
        ValueError: If param_name is not tracked or has invalid shape.
        TypeError: If activity_tracker is not an EMAActivity instance.

    Example:
        >>> tracker = EMAActivity()
        >>> # ... update tracker with gradients ...
        >>> ax = plot_layer_heatmap(tracker, 'layer1.weight')
        >>> plt.savefig('layer_heatmap.png')
    """
    if not isinstance(activity_tracker, EMAActivity):
        raise TypeError(
            f"activity_tracker must be EMAActivity, got {type(activity_tracker)}"
        )

    if param_name not in activity_tracker.activity_scores:
        raise ValueError(
            f"Parameter '{param_name}' not tracked. "
            f"Available: {list(activity_tracker.activity_scores.keys())}"
        )

    activity = activity_tracker.get_activity(param_name).detach().cpu().numpy()

    # Handle different tensor shapes
    if activity.ndim == 1:
        # 1D: reshape to (1, n) for display
        activity = activity.reshape(1, -1)
    elif activity.ndim > 2:
        # Higher dims: flatten last dimensions
        activity = activity.reshape(activity.shape[0], -1)

    # Auto-calculate figure size if not provided
    if figsize is None:
        height, width = activity.shape
        # Scale to reasonable size
        fig_width = max(8, min(width / 20, 20))
        fig_height = max(6, min(height / 20, 15))
        figsize = (fig_width, fig_height)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(activity, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activity Score', fontsize=10)

    # Add tier threshold lines on colorbar
    cbar.ax.axhline(
        activity_tracker.hot_threshold, color='red', linewidth=2,
        linestyle='--', label='Hot threshold'
    )
    cbar.ax.axhline(
        activity_tracker.warm_threshold, color='yellow', linewidth=2,
        linestyle='--', label='Warm threshold'
    )

    # Set labels and title
    if title is None:
        title = f"Activity Heatmap: {param_name}"
    ax.set_title(title, fontsize=14)

    if activity.shape[0] == 1:
        ax.set_xlabel('Weight Index', fontsize=12)
        ax.set_yticks([])
    else:
        ax.set_xlabel('Input Dimension', fontsize=12)
        ax.set_ylabel('Output Dimension', fontsize=12)

    # Add grid overlay for large heatmaps
    if activity.shape[0] <= 50 and activity.shape[1] <= 50:
        ax.set_xticks(np.arange(activity.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(activity.shape[0]) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)

    # Add statistics
    mean_activity = np.mean(activity)
    hot_pct = np.mean(activity > activity_tracker.hot_threshold) * 100
    warm_pct = np.mean(
        (activity > activity_tracker.warm_threshold) &
        (activity <= activity_tracker.hot_threshold)
    ) * 100
    cold_pct = np.mean(activity <= activity_tracker.warm_threshold) * 100

    stats_text = (
        f"Mean: {mean_activity:.3f}\n"
        f"Hot: {hot_pct:.1f}% | Warm: {warm_pct:.1f}% | Cold: {cold_pct:.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return ax


def plot_activity_summary(
    activity_tracker: EMAActivity,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> Figure:
    """Create a comprehensive summary figure with all activity visualizations.

    Generates a multi-panel figure containing histogram, tier distribution,
    and heatmaps for the first few layers. Useful for getting a complete
    overview of model activity patterns.

    Args:
        activity_tracker: EMAActivity tracker instance.
        output_path: Optional path to save the figure. If None, figure is
            only returned (not saved).
        figsize: Figure size as (width, height). Default (16, 10).

    Returns:
        The matplotlib Figure object containing all subplots.

    Raises:
        ValueError: If no activity scores are tracked.
        TypeError: If activity_tracker is not an EMAActivity instance.

    Example:
        >>> tracker = EMAActivity()
        >>> # ... update tracker with gradients ...
        >>> fig = plot_activity_summary(tracker, 'summary.png')
    """
    if not isinstance(activity_tracker, EMAActivity):
        raise TypeError(
            f"activity_tracker must be EMAActivity, got {type(activity_tracker)}"
        )

    if not activity_tracker.activity_scores:
        raise ValueError("No activity scores tracked. Update tracker first.")

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Histogram (top-left)
    ax_hist = fig.add_subplot(gs[0, 0])
    plot_activity_histogram(activity_tracker, ax=ax_hist, title="Activity Distribution")

    # Tier distribution (top-right)
    ax_tier = fig.add_subplot(gs[0, 1])
    plot_tier_distribution(activity_tracker, ax=ax_tier, title="Tier Distribution")

    # Heatmaps for first 4 layers (middle and bottom rows)
    param_names = list(activity_tracker.activity_scores.keys())[:4]
    for i, param_name in enumerate(param_names):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        try:
            plot_layer_heatmap(
                activity_tracker, param_name,
                title=f"{param_name}", ax=ax
            )
        except ValueError:
            # Skip if parameter can't be visualized as heatmap
            ax.text(0.5, 0.5, f"Cannot visualize\n{param_name}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(param_name)

    # Add overall title
    fig.suptitle(
        f"Activity Summary (decay={activity_tracker.decay}, "
        f"hot>{activity_tracker.hot_threshold}, "
        f"warm>{activity_tracker.warm_threshold})",
        fontsize=16, y=0.995
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def save_visualization(
    ax_or_fig: Union[plt.Axes, Figure],
    output_path: str,
    dpi: int = 150,
    bbox_inches: str = 'tight',
) -> None:
    """Save a matplotlib visualization to file.

    Supports PNG and PDF output formats based on file extension.

    Args:
        ax_or_fig: Matplotlib Axes or Figure to save.
        output_path: Path to save the file. Extension determines format
            ('.png', '.pdf', '.jpg', '.svg', etc.).
        dpi: Resolution for raster formats. Default 150.
        bbox_inches: Bounding box option. Default 'tight'.

    Raises:
        ValueError: If output_path has unsupported extension.
        TypeError: If ax_or_fig is not a valid matplotlib object.

    Example:
        >>> ax = plot_activity_histogram(tracker)
        >>> save_visualization(ax, 'output.png')
        >>> save_visualization(fig, 'output.pdf', dpi=300)
    """
    # Get figure from axes if needed
    if isinstance(ax_or_fig, plt.Axes):
        fig = ax_or_fig.figure
    elif isinstance(ax_or_fig, Figure):
        fig = ax_or_fig
    else:
        raise TypeError(
            f"Expected Axes or Figure, got {type(ax_or_fig)}"
        )

    # Validate output format
    valid_extensions = ['.png', '.pdf', '.jpg', '.jpeg', '.svg', '.eps', '.tiff', '.tif']
    output_lower = output_path.lower()
    if not any(output_lower.endswith(ext) for ext in valid_extensions):
        raise ValueError(
            f"Unsupported output format. Supported: {', '.join(valid_extensions)}"
        )

    # Save the figure
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)


def _validate_image_file(file_path: str) -> bool:
    """Validate that a file is a valid image file.

    Checks file existence, non-zero size, and valid image header
    for PNG and PDF files.

    Args:
        file_path: Path to the image file.

    Returns:
        True if file is valid, False otherwise.
    """
    import os

    if not os.path.exists(file_path):
        return False

    if os.path.getsize(file_path) == 0:
        return False

    # Check file header for PNG
    if file_path.lower().endswith('.png'):
        with open(file_path, 'rb') as f:
            header = f.read(8)
            # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
            if header != b'\x89PNG\r\n\x1a\n':
                return False

    # Check file header for PDF
    elif file_path.lower().endswith('.pdf'):
        with open(file_path, 'rb') as f:
            header = f.read(5)
            # PDF starts with %PDF-
            if not header.startswith(b'%PDF-'):
                return False

    return True
