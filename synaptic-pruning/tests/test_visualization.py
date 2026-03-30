"""Tests for the activity visualization module.

This module contains comprehensive tests for visualization functions,
covering histogram plotting, tier distribution charts, layer heatmaps,
and PNG/PDF output validation.
"""

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from synaptic_pruning import EMAActivity
from synaptic_pruning.visualization import (
    _validate_image_file,
    plot_activity_histogram,
    plot_activity_summary,
    plot_layer_heatmap,
    plot_tier_distribution,
    save_visualization,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def sample_tracker():
    """Create an EMAActivity tracker with sample data."""
    tracker = EMAActivity(decay=0.9, hot_threshold=0.8, warm_threshold=0.3)

    # Create diverse activity patterns
    # Layer 1: Mostly hot weights (high activity)
    tracker.activity_scores["layer1.weight"] = torch.rand(100, 50) * 0.5 + 0.5

    # Layer 2: Mix of warm and cold
    scores = torch.rand(50, 100)
    scores[scores < 0.5] *= 0.3  # Cold weights
    scores[scores >= 0.5] = scores[scores >= 0.5] * 0.4 + 0.3  # Warm weights
    tracker.activity_scores["layer2.weight"] = scores

    # Layer 3: Mostly cold weights
    tracker.activity_scores["layer3.weight"] = torch.rand(30, 30) * 0.25

    # Layer 4: Uniform distribution for histogram testing
    tracker.activity_scores["layer4.weight"] = torch.linspace(0, 1, 1000).reshape(50, 20)

    # 1D layer for edge case testing
    tracker.activity_scores["layer5.bias"] = torch.rand(100)

    return tracker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPlotActivityHistogram:
    """Tests for plot_activity_histogram function."""

    def test_returns_axes(self, sample_tracker):
        """Test that function returns a matplotlib Axes object."""
        ax = plot_activity_histogram(sample_tracker)
        assert isinstance(ax, Axes)

    def test_accepts_dict_input(self):
        """Test that function accepts a dictionary of tensors."""
        scores_dict = {
            "weight1": torch.rand(100),
            "weight2": torch.rand(50, 50),
        }
        ax = plot_activity_histogram(scores_dict)
        assert isinstance(ax, Axes)

    def test_accepts_ema_activity(self, sample_tracker):
        """Test that function accepts EMAActivity instance."""
        ax = plot_activity_histogram(sample_tracker)
        assert isinstance(ax, Axes)

    def test_invalid_input_raises(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="activity_scores must be"):
            plot_activity_histogram("invalid")
        with pytest.raises(TypeError, match="activity_scores must be"):
            plot_activity_histogram(123)
        with pytest.raises(TypeError, match="activity_scores must be"):
            plot_activity_histogram([1, 2, 3])

    def test_empty_dict_raises(self):
        """Test that empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="No activity scores found"):
            plot_activity_histogram({})

    def test_empty_tracker_raises(self):
        """Test that empty tracker raises ValueError."""
        tracker = EMAActivity()
        with pytest.raises(ValueError, match="No activity scores found"):
            plot_activity_histogram(tracker)

    def test_param_filtering(self, sample_tracker):
        """Test filtering by parameter names."""
        ax = plot_activity_histogram(sample_tracker, param_names=["layer1.weight"])
        assert isinstance(ax, Axes)

    def test_invalid_param_names_raises(self, sample_tracker):
        """Test that invalid param names raise ValueError."""
        with pytest.raises(ValueError, match="No matching parameters found"):
            plot_activity_histogram(sample_tracker, param_names=["nonexistent.weight"])

    def test_custom_bins(self, sample_tracker):
        """Test custom bin count."""
        ax = plot_activity_histogram(sample_tracker, bins=100)
        assert isinstance(ax, Axes)

    def test_custom_figsize(self, sample_tracker):
        """Test custom figure size."""
        ax = plot_activity_histogram(sample_tracker, figsize=(8, 4))
        assert isinstance(ax, Axes)

    def test_custom_title(self, sample_tracker):
        """Test custom title."""
        ax = plot_activity_histogram(sample_tracker, title="Custom Title")
        assert ax.get_title() == "Custom Title"

    def test_provided_axes(self, sample_tracker):
        """Test plotting on provided axes."""
        fig, ax = plt.subplots()
        result = plot_activity_histogram(sample_tracker, ax=ax)
        assert result is ax

    def test_histogram_has_stats(self, sample_tracker):
        """Test that histogram includes mean/median lines."""
        ax = plot_activity_histogram(sample_tracker)
        lines = ax.get_lines()
        # Should have 2 lines for mean and median
        assert len(lines) == 2


class TestPlotTierDistribution:
    """Tests for plot_tier_distribution function."""

    def test_returns_axes(self, sample_tracker):
        """Test that function returns a matplotlib Axes object."""
        ax = plot_tier_distribution(sample_tracker)
        assert isinstance(ax, Axes)

    def test_invalid_input_raises(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="activity_tracker must be"):
            plot_tier_distribution("invalid")
        with pytest.raises(TypeError, match="activity_tracker must be"):
            plot_tier_distribution({"layer1": torch.rand(10)})

    def test_empty_tracker_raises(self):
        """Test that empty tracker raises ValueError."""
        tracker = EMAActivity()
        with pytest.raises(ValueError, match="No activity scores tracked"):
            plot_tier_distribution(tracker)

    def test_normalized_output(self, sample_tracker):
        """Test normalized (percentage) output."""
        ax = plot_tier_distribution(sample_tracker, normalize=True)
        ylabel = ax.get_ylabel()
        assert "Percentage" in ylabel or "%" in ylabel

    def test_count_output(self, sample_tracker):
        """Test count (non-normalized) output."""
        ax = plot_tier_distribution(sample_tracker, normalize=False)
        ylabel = ax.get_ylabel()
        assert "Count" in ylabel

    def test_param_filtering(self, sample_tracker):
        """Test filtering by parameter names."""
        ax = plot_tier_distribution(sample_tracker, param_names=["layer1.weight", "layer2.weight"])
        assert isinstance(ax, Axes)

    def test_invalid_param_names_raises(self, sample_tracker):
        """Test that invalid param names raise ValueError."""
        with pytest.raises(ValueError, match="No matching parameters found"):
            plot_tier_distribution(sample_tracker, param_names=["nonexistent.weight"])

    def test_stacked_bars(self, sample_tracker):
        """Test that chart has stacked bars."""
        ax = plot_tier_distribution(sample_tracker)
        # Check for patches (bars)
        patches = ax.patches
        assert len(patches) > 0

    def test_threshold_annotation(self, sample_tracker):
        """Test that threshold info is displayed."""
        ax = plot_tier_distribution(sample_tracker)
        # The annotation should be in the axes texts
        texts = [t.get_text() for t in ax.texts]
        assert any("Thresholds" in t for t in texts)


class TestPlotLayerHeatmap:
    """Tests for plot_layer_heatmap function."""

    def test_returns_axes(self, sample_tracker):
        """Test that function returns a matplotlib Axes object."""
        ax = plot_layer_heatmap(sample_tracker, "layer1.weight")
        assert isinstance(ax, Axes)

    def test_invalid_tracker_raises(self):
        """Test that invalid tracker type raises TypeError."""
        with pytest.raises(TypeError, match="activity_tracker must be"):
            plot_layer_heatmap("invalid", "layer1.weight")

    def test_invalid_param_name_raises(self, sample_tracker):
        """Test that invalid param name raises ValueError."""
        with pytest.raises(ValueError, match="Parameter 'nonexistent' not tracked"):
            plot_layer_heatmap(sample_tracker, "nonexistent")

    def test_1d_layer(self, sample_tracker):
        """Test heatmap for 1D parameter (bias)."""
        ax = plot_layer_heatmap(sample_tracker, "layer5.bias")
        assert isinstance(ax, Axes)

    def test_2d_layer(self, sample_tracker):
        """Test heatmap for 2D parameter (weight matrix)."""
        ax = plot_layer_heatmap(sample_tracker, "layer1.weight")
        assert isinstance(ax, Axes)

    def test_colorbar_present(self, sample_tracker):
        """Test that colorbar is included."""
        fig, ax = plt.subplots()
        plot_layer_heatmap(sample_tracker, "layer1.weight", ax=ax)
        # Check for colorbar in figure
        assert len(fig.axes) > 1  # Should have more than just the main axes

    def test_custom_cmap(self, sample_tracker):
        """Test custom colormap."""
        ax = plot_layer_heatmap(sample_tracker, "layer1.weight", cmap="hot")
        assert isinstance(ax, Axes)

    def test_custom_title(self, sample_tracker):
        """Test custom title."""
        ax = plot_layer_heatmap(sample_tracker, "layer1.weight", title="Custom Heatmap Title")
        assert ax.get_title() == "Custom Heatmap Title"

    def test_default_title(self, sample_tracker):
        """Test default title includes param name."""
        ax = plot_layer_heatmap(sample_tracker, "layer1.weight")
        title = ax.get_title()
        assert "layer1.weight" in title

    def test_statistics_overlay(self, sample_tracker):
        """Test that statistics are overlaid on heatmap."""
        ax = plot_layer_heatmap(sample_tracker, "layer1.weight")
        texts = [t.get_text() for t in ax.texts]
        assert any("Mean" in t for t in texts)


class TestPlotActivitySummary:
    """Tests for plot_activity_summary function."""

    def test_returns_figure(self, sample_tracker):
        """Test that function returns a matplotlib Figure object."""
        fig = plot_activity_summary(sample_tracker)
        assert isinstance(fig, Figure)

    def test_invalid_tracker_raises(self):
        """Test that invalid tracker type raises TypeError."""
        with pytest.raises(TypeError, match="activity_tracker must be"):
            plot_activity_summary("invalid")

    def test_empty_tracker_raises(self):
        """Test that empty tracker raises ValueError."""
        tracker = EMAActivity()
        with pytest.raises(ValueError, match="No activity scores tracked"):
            plot_activity_summary(tracker)

    def test_multiple_subplots(self, sample_tracker):
        """Test that figure contains multiple subplots."""
        fig = plot_activity_summary(sample_tracker)
        axes = fig.axes
        assert len(axes) >= 4  # Should have at least histogram, tier dist, and 2 heatmaps

    def test_with_output_path(self, sample_tracker, temp_dir):
        """VAL-ACT-003: Test saving to file creates valid PNG."""
        output_path = os.path.join(temp_dir, "summary.png")
        plot_activity_summary(sample_tracker, output_path=output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Validate it's a real PNG
        assert _validate_image_file(output_path)


class TestSaveVisualization:
    """Tests for save_visualization function."""

    def test_save_axes_to_png(self, sample_tracker, temp_dir):
        """VAL-ACT-003: Test saving Axes to PNG creates valid image."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "histogram.png")

        save_visualization(ax, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        assert _validate_image_file(output_path)

    def test_save_axes_to_pdf(self, sample_tracker, temp_dir):
        """VAL-ACT-003: Test saving Axes to PDF creates valid file."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "histogram.pdf")

        save_visualization(ax, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        assert _validate_image_file(output_path)

    def test_save_figure_to_png(self, sample_tracker, temp_dir):
        """VAL-ACT-003: Test saving Figure to PNG creates valid image."""
        fig = plot_activity_summary(sample_tracker)
        output_path = os.path.join(temp_dir, "summary.png")

        save_visualization(fig, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        assert _validate_image_file(output_path)

    def test_save_figure_to_pdf(self, sample_tracker, temp_dir):
        """VAL-ACT-003: Test saving Figure to PDF creates valid file."""
        fig = plot_activity_summary(sample_tracker)
        output_path = os.path.join(temp_dir, "summary.pdf")

        save_visualization(fig, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        assert _validate_image_file(output_path)

    def test_save_with_custom_dpi(self, sample_tracker, temp_dir):
        """Test saving with custom DPI."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "custom_dpi.png")

        save_visualization(ax, output_path, dpi=300)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_invalid_input_raises(self):
        """Test that invalid input raises TypeError."""
        with pytest.raises(TypeError, match="Expected Axes or Figure"):
            save_visualization("invalid", "output.png")
        with pytest.raises(TypeError, match="Expected Axes or Figure"):
            save_visualization(123, "output.png")

    def test_unsupported_format_raises(self, sample_tracker, temp_dir):
        """Test that unsupported format raises ValueError."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "output.xyz")

        with pytest.raises(ValueError, match="Unsupported output format"):
            save_visualization(ax, output_path)

    def test_supports_svg(self, sample_tracker, temp_dir):
        """Test that SVG format is supported."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "output.svg")

        save_visualization(ax, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_supports_jpg(self, sample_tracker, temp_dir):
        """Test that JPG format is supported."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "output.jpg")

        save_visualization(ax, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


class TestValidateImageFile:
    """Tests for _validate_image_file helper function."""

    def test_valid_png(self, sample_tracker, temp_dir):
        """Test validation of valid PNG file."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "test.png")
        fig = ax.figure
        fig.savefig(output_path)

        assert _validate_image_file(output_path)

    def test_valid_pdf(self, sample_tracker, temp_dir):
        """Test validation of valid PDF file."""
        ax = plot_activity_histogram(sample_tracker)
        output_path = os.path.join(temp_dir, "test.pdf")
        fig = ax.figure
        fig.savefig(output_path)

        assert _validate_image_file(output_path)

    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        assert not _validate_image_file("/path/to/nonexistent.png")

    def test_empty_file(self, temp_dir):
        """Test validation of empty file."""
        output_path = os.path.join(temp_dir, "empty.png")
        with open(output_path, "w"):
            pass  # Create empty file

        assert not _validate_image_file(output_path)

    def test_invalid_png_header(self, temp_dir):
        """Test validation of file with invalid PNG header."""
        output_path = os.path.join(temp_dir, "fake.png")
        with open(output_path, "wb") as f:
            f.write(b"NOT A PNG FILE")

        assert not _validate_image_file(output_path)

    def test_invalid_pdf_header(self, temp_dir):
        """Test validation of file with invalid PDF header."""
        output_path = os.path.join(temp_dir, "fake.pdf")
        with open(output_path, "wb") as f:
            f.write(b"NOT A PDF FILE")

        assert not _validate_image_file(output_path)


class TestIntegration:
    """Integration tests for the visualization module."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: track -> visualize -> save."""
        # Create tracker and simulate training
        tracker = EMAActivity(decay=0.9, hot_threshold=0.8, warm_threshold=0.3)

        # Simulate gradient updates
        for i in range(10):
            tracker.update("layer1.weight", torch.randn(64, 32))
            tracker.update("layer2.weight", torch.randn(32, 16))

        # Create visualizations
        ax1 = plot_activity_histogram(tracker)
        ax2 = plot_tier_distribution(tracker)
        ax3 = plot_layer_heatmap(tracker, "layer1.weight")

        # Save to files
        hist_path = os.path.join(temp_dir, "histogram.png")
        tier_path = os.path.join(temp_dir, "tiers.pdf")
        heatmap_path = os.path.join(temp_dir, "heatmap.png")

        save_visualization(ax1, hist_path)
        save_visualization(ax2, tier_path)
        save_visualization(ax3, heatmap_path)

        # Validate all outputs
        assert _validate_image_file(hist_path)
        assert _validate_image_file(tier_path)
        assert _validate_image_file(heatmap_path)

    def test_summary_with_save(self, temp_dir):
        """Test generating and saving summary figure."""
        tracker = EMAActivity(decay=0.9)

        # Add some data
        tracker.update("layer1.weight", torch.randn(32, 32))
        tracker.update("layer2.weight", torch.randn(16, 32))

        # Generate and save summary
        summary_path = os.path.join(temp_dir, "summary.png")
        fig = plot_activity_summary(tracker, output_path=summary_path)

        assert isinstance(fig, Figure)
        assert _validate_image_file(summary_path)

    def test_all_formats(self, sample_tracker, temp_dir):
        """VAL-ACT-003: Test saving to all supported formats."""
        ax = plot_activity_histogram(sample_tracker)
        formats = ["png", "pdf", "jpg", "svg"]

        for fmt in formats:
            output_path = os.path.join(temp_dir, f"output.{fmt}")
            save_visualization(ax, output_path)

            assert os.path.exists(output_path), f"Failed to create {fmt}"
            assert os.path.getsize(output_path) > 0, f"Empty {fmt} file"


class TestEdgeCases:
    """Edge case tests for visualization functions."""

    def test_single_weight_histogram(self):
        """Test histogram with single weight."""
        tracker = EMAActivity()
        tracker.update("weight", torch.tensor([0.5]))

        ax = plot_activity_histogram(tracker)
        assert isinstance(ax, Axes)

    def test_all_same_activity(self):
        """Test with all weights having same activity."""
        tracker = EMAActivity()
        tracker.activity_scores["weight"] = torch.ones(100, 100) * 0.5

        ax = plot_activity_histogram(tracker)
        assert isinstance(ax, Axes)

    def test_extreme_activity_values(self):
        """Test with extreme activity values (all 0 or all 1)."""
        tracker = EMAActivity()
        tracker.activity_scores["all_zero"] = torch.zeros(50, 50)
        tracker.activity_scores["all_one"] = torch.ones(50, 50)

        ax = plot_activity_histogram(tracker)
        assert isinstance(ax, Axes)

    def test_large_matrix_heatmap(self):
        """Test heatmap with large matrix."""
        tracker = EMAActivity()
        tracker.update("large.weight", torch.randn(500, 500))

        ax = plot_layer_heatmap(tracker, "large.weight")
        assert isinstance(ax, Axes)

    def test_narrow_matrix_heatmap(self):
        """Test heatmap with very narrow matrix."""
        tracker = EMAActivity()
        tracker.activity_scores["narrow.weight"] = torch.rand(1000, 2)

        ax = plot_layer_heatmap(tracker, "narrow.weight")
        assert isinstance(ax, Axes)

    def test_device_handling_cpu(self):
        """Test visualization works with CPU tensors."""
        tracker = EMAActivity()
        tracker.update("weight", torch.randn(50, 50))

        ax = plot_layer_heatmap(tracker, "weight")
        assert isinstance(ax, Axes)

    def test_handles_special_chars_in_names(self):
        """Test handling of special characters in parameter names."""
        tracker = EMAActivity()
        tracker.activity_scores["model.encoder.layer.0.attention.weight"] = torch.rand(50, 50)

        ax = plot_tier_distribution(tracker)
        assert isinstance(ax, Axes)

    def test_partial_param_filtering(self, sample_tracker):
        """Test filtering with some valid and some invalid param names."""
        ax = plot_tier_distribution(
            sample_tracker, param_names=["layer1.weight", "nonexistent.weight", "layer2.weight"]
        )
        assert isinstance(ax, Axes)
