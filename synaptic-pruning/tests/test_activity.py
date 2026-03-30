"""Tests for the EMAActivity tracker.

This module contains comprehensive tests for the EMAActivity class,
covering EMA decay correctness, tier classification, and edge cases.
"""

import pytest
import torch

from synaptic_pruning import EMAActivity


class TestEMAActivityInitialization:
    """Tests for EMAActivity initialization and configuration."""

    def test_default_initialization(self):
        """Test that EMAActivity initializes with correct defaults."""
        tracker = EMAActivity()
        assert tracker.decay == 0.9
        assert tracker.hot_threshold == 0.8
        assert tracker.warm_threshold == 0.3
        assert tracker.activity_scores == {}

    def test_custom_initialization(self):
        """Test that EMAActivity accepts custom parameters."""
        tracker = EMAActivity(decay=0.95, hot_threshold=0.75, warm_threshold=0.25)
        assert tracker.decay == 0.95
        assert tracker.hot_threshold == 0.75
        assert tracker.warm_threshold == 0.25

    def test_invalid_decay_raises(self):
        """Test that invalid decay values raise ValueError."""
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            EMAActivity(decay=0.0)
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            EMAActivity(decay=1.0)
        with pytest.raises(ValueError, match="decay must be between 0 and 1"):
            EMAActivity(decay=-0.1)

    def test_invalid_thresholds_raises(self):
        """Test that invalid threshold configurations raise ValueError."""
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            EMAActivity(warm_threshold=0.5, hot_threshold=0.5)
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            EMAActivity(warm_threshold=0.9, hot_threshold=0.8)
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            EMAActivity(warm_threshold=0.0, hot_threshold=0.5)


class TestEMAActivityUpdate:
    """Tests for the update() method."""

    def test_first_update_initializes_scores(self):
        """Test that first update initializes activity scores."""
        tracker = EMAActivity()
        gradients = torch.randn(10, 10)
        
        tracker.update("weight1", gradients)
        
        assert "weight1" in tracker.activity_scores
        assert tracker.activity_scores["weight1"].shape == gradients.shape

    def test_update_with_nan_raises(self):
        """Test that NaN gradients raise ValueError."""
        tracker = EMAActivity()
        gradients = torch.tensor([1.0, float('nan'), 3.0])
        
        with pytest.raises(ValueError, match="contain NaN"):
            tracker.update("weight1", gradients)

    def test_update_with_inf_raises(self):
        """Test that Inf gradients raise ValueError."""
        tracker = EMAActivity()
        gradients = torch.tensor([1.0, float('inf'), 3.0])
        
        with pytest.raises(ValueError, match="contain NaN or Inf"):
            tracker.update("weight1", gradients)

    def test_update_with_zero_gradients(self):
        """Test that zero gradients result in zero activity."""
        tracker = EMAActivity()
        gradients = torch.zeros(5, 5)
        
        tracker.update("weight1", gradients)
        activity = tracker.get_activity("weight1")
        
        assert torch.allclose(activity, torch.zeros_like(activity))

    def test_update_multiple_parameters(self):
        """Test tracking multiple parameters independently."""
        tracker = EMAActivity()
        
        tracker.update("weight1", torch.randn(10))
        tracker.update("weight2", torch.randn(5, 5))
        
        assert "weight1" in tracker.activity_scores
        assert "weight2" in tracker.activity_scores
        assert tracker.activity_scores["weight1"].shape == (10,)
        assert tracker.activity_scores["weight2"].shape == (5, 5)

    def test_update_handles_shape_change(self):
        """Test that update handles shape changes gracefully."""
        tracker = EMAActivity()
        
        # First update with one shape
        tracker.update("weight1", torch.randn(10, 10))
        old_activity = tracker.get_activity("weight1").clone()
        
        # Update with different shape - should reinitialize
        tracker.update("weight1", torch.randn(5, 5))
        new_activity = tracker.get_activity("weight1")
        
        assert new_activity.shape == (5, 5)
        # Should be different from old activity
        assert new_activity.shape != old_activity.shape


class TestEMAActivityDecay:
    """Tests for EMA decay behavior (VAL-ACT-001)."""

    def test_ema_decay_curve_single_weight(self):
        """VAL-ACT-001: EMA approaches 1.0 for active weights, decays to 0 for inactive.
        
        With decay=0.9 and consistent activity=1.0, after n updates:
        score_n = (1-decay) * (1 + decay + decay^2 + ... + decay^(n-1))
                = 1 - decay^n
        As n -> infinity, score -> 1.0
        """
        tracker = EMAActivity(decay=0.9)
        
        # Simulate consistent high activity (all gradients = 1.0)
        for _ in range(50):
            tracker.update("weight1", torch.ones(1))
        
        activity = tracker.get_activity("weight1")
        # After many updates with activity=1, EMA should be very close to 1.0
        assert activity.item() > 0.99, f"Expected > 0.99, got {activity.item()}"

    def test_ema_decay_for_inactive_weights(self):
        """VAL-ACT-001: EMA decays toward 0 for inactive weights."""
        tracker = EMAActivity(decay=0.9)
        
        # First establish high activity
        tracker.update("weight1", torch.ones(1) * 10)
        
        # Now apply zero gradients (inactive)
        for _ in range(50):
            tracker.update("weight1", torch.zeros(1))
        
        activity = tracker.get_activity("weight1")
        # After many zero updates, EMA should be very close to 0.0
        assert activity.item() < 0.01, f"Expected < 0.01, got {activity.item()}"

    def test_ema_formula_correctness(self):
        """Test that EMA formula is computed correctly."""
        tracker = EMAActivity(decay=0.5)  # High decay for faster convergence
        
        # First update - initializes directly with normalized activity
        grad1 = 1.0
        tracker.update("weight1", torch.tensor([grad1]))
        # First update: score = normalized_grad = 1.0 / 1.0 = 1.0
        expected_activity = 1.0
        actual_activity = tracker.get_activity("weight1")
        assert torch.allclose(actual_activity, torch.tensor([expected_activity]), atol=1e-5)
        
        # Second update: EMA kicks in
        grad2 = 0.5
        tracker.update("weight1", torch.tensor([grad2]))
        # normalized_grad = 0.5 / 0.5 = 1.0
        # new_score = decay * old_score + (1 - decay) * normalized_grad
        expected_activity = 0.5 * 1.0 + 0.5 * 1.0  # = 1.0
        actual_activity = tracker.get_activity("weight1")
        assert torch.allclose(actual_activity, torch.tensor([expected_activity]), atol=1e-5)
        
        # Third update with zero gradient
        grad3 = 0.0
        tracker.update("weight1", torch.tensor([grad3]))
        # normalized_grad = 0.0 (all zeros)
        expected_activity = 0.5 * 1.0 + 0.5 * 0.0  # = 0.5
        actual_activity = tracker.get_activity("weight1")
        assert torch.allclose(actual_activity, torch.tensor([expected_activity]), atol=1e-5)
        
        # Fourth update with full gradient again
        grad4 = 1.0
        tracker.update("weight1", torch.tensor([grad4]))
        # normalized_grad = 1.0
        expected_activity = 0.5 * 0.5 + 0.5 * 1.0  # = 0.75
        actual_activity = tracker.get_activity("weight1")
        assert torch.allclose(actual_activity, torch.tensor([expected_activity]), atol=1e-5)

    def test_different_decay_rates(self):
        """Test that different decay rates produce different decay speeds."""
        # Fast decay tracker
        fast_tracker = EMAActivity(decay=0.5)
        # Slow decay tracker
        slow_tracker = EMAActivity(decay=0.95)
        
        # Initialize both with same activity
        fast_tracker.update("w", torch.ones(1) * 10)
        slow_tracker.update("w", torch.ones(1) * 10)
        
        # Apply zeros
        for _ in range(10):
            fast_tracker.update("w", torch.zeros(1))
            slow_tracker.update("w", torch.zeros(1))
        
        fast_activity = fast_tracker.get_activity("w")
        slow_activity = slow_tracker.get_activity("w")
        
        # Fast decay should result in lower activity
        assert fast_activity < slow_activity


class TestTierClassification:
    """Tests for tier classification (VAL-ACT-002)."""

    def test_tier_classification_basic(self):
        """VAL-ACT-002: Weights are correctly classified into activity tiers."""
        tracker = EMAActivity(hot_threshold=0.8, warm_threshold=0.3)
        
        # Create artificial activity scores
        tracker.activity_scores["weight1"] = torch.tensor([
            0.9,  # hot (>= 0.8)
            0.5,  # warm (0.3-0.8)
            0.1,  # cold (< 0.3)
            0.85, # hot
            0.35, # warm
            0.05, # cold
        ])
        
        hot_mask, warm_mask, cold_mask = tracker.get_tier_mask("weight1")
        
        # Check hot weights
        assert hot_mask[0].item() == True
        assert hot_mask[3].item() == True
        
        # Check warm weights
        assert warm_mask[1].item() == True
        assert warm_mask[4].item() == True
        
        # Check cold weights
        assert cold_mask[2].item() == True
        assert cold_mask[5].item() == True

    def test_tier_counts_correct(self):
        """VAL-ACT-002: Tier counts match expected values."""
        tracker = EMAActivity(hot_threshold=0.8, warm_threshold=0.3)
        
        # Create activity with known distribution
        tracker.activity_scores["weight1"] = torch.tensor([
            0.9, 0.95, 0.85,  # 3 hot
            0.5, 0.4, 0.35,   # 3 warm
            0.1, 0.2, 0.25,   # 3 cold
        ])
        
        hot_count, warm_count, cold_count = tracker.get_tier_counts("weight1")
        
        assert hot_count == 3
        assert warm_count == 3
        assert cold_count == 3

    def test_tier_masks_are_mutually_exclusive(self):
        """Test that tier masks don't overlap."""
        tracker = EMAActivity()
        
        # Create random activity scores
        tracker.activity_scores["weight1"] = torch.rand(100)
        
        hot_mask, warm_mask, cold_mask = tracker.get_tier_mask("weight1")
        
        # No overlap between masks
        assert not (hot_mask & warm_mask).any()
        assert not (hot_mask & cold_mask).any()
        assert not (warm_mask & cold_mask).any()
        
        # All weights are covered
        total_covered = (hot_mask | warm_mask | cold_mask).sum()
        assert total_covered == 100

    def test_tier_classification_edge_cases(self):
        """Test tier classification at exact threshold boundaries."""
        tracker = EMAActivity(hot_threshold=0.8, warm_threshold=0.3)
        
        tracker.activity_scores["weight1"] = torch.tensor([
            0.8,  # Exactly at hot threshold -> warm (not > hot_threshold)
            0.81, # Just above hot threshold -> hot
            0.3,  # Exactly at warm threshold -> cold (not > warm_threshold)
            0.31, # Just above warm threshold -> warm
        ])
        
        hot_mask, warm_mask, cold_mask = tracker.get_tier_mask("weight1")
        
        assert hot_mask[1].item() == True  # 0.81 > 0.8
        assert warm_mask[0].item() == True  # 0.8 is not > 0.8, but > 0.3
        assert warm_mask[3].item() == True  # 0.31 > 0.3 and <= 0.8
        assert cold_mask[2].item() == True  # 0.3 is not > 0.3


class TestActivityAccess:
    """Tests for get_activity() method."""

    def test_get_activity_before_update_raises(self):
        """Test that get_activity raises KeyError before any update."""
        tracker = EMAActivity()
        
        with pytest.raises(KeyError, match="has not been registered"):
            tracker.get_activity("unknown_weight")

    def test_get_activity_returns_correct_shape(self):
        """Test that get_activity returns tensor with correct shape."""
        tracker = EMAActivity()
        
        shapes = [(10,), (5, 5), (3, 4, 5)]
        for i, shape in enumerate(shapes):
            param_name = f"weight_{i}"
            tracker.update(param_name, torch.randn(*shape))
            activity = tracker.get_activity(param_name)
            assert activity.shape == shape


class TestStateManagement:
    """Tests for state_dict() and load_state_dict()."""

    def test_state_dict_contains_all_data(self):
        """Test that state_dict contains all necessary information."""
        tracker = EMAActivity(decay=0.95, hot_threshold=0.75, warm_threshold=0.25)
        tracker.update("weight1", torch.randn(10))
        tracker.update("weight2", torch.randn(5, 5))
        
        state = tracker.state_dict()
        
        assert "activity_scores" in state
        assert "decay" in state
        assert "hot_threshold" in state
        assert "warm_threshold" in state
        
        # Use approximate comparison for float tensors
        assert abs(float(state["decay"].item()) - 0.95) < 1e-5
        assert abs(float(state["hot_threshold"].item()) - 0.75) < 1e-5
        assert abs(float(state["warm_threshold"].item()) - 0.25) < 1e-5
        assert "weight1" in state["activity_scores"]
        assert "weight2" in state["activity_scores"]

    def test_load_state_dict_restores_state(self):
        """Test that load_state_dict correctly restores tracker state."""
        tracker1 = EMAActivity(decay=0.95, hot_threshold=0.75, warm_threshold=0.25)
        tracker1.update("weight1", torch.randn(10))
        
        state = tracker1.state_dict()
        
        # Create new tracker with different config, then load
        tracker2 = EMAActivity()
        tracker2.load_state_dict(state)
        
        # Use approximate comparison for floats
        assert abs(tracker2.decay - 0.95) < 1e-5
        assert abs(tracker2.hot_threshold - 0.75) < 1e-5
        assert abs(tracker2.warm_threshold - 0.25) < 1e-5
        assert "weight1" in tracker2.activity_scores
        assert torch.allclose(
            tracker1.get_activity("weight1"),
            tracker2.get_activity("weight1")
        )

    def test_reset_clears_scores(self):
        """Test that reset() clears all activity scores."""
        tracker = EMAActivity()
        tracker.update("weight1", torch.randn(10))
        tracker.update("weight2", torch.randn(5))
        
        assert len(tracker.activity_scores) == 2
        
        tracker.reset()
        
        assert len(tracker.activity_scores) == 0


class TestDeviceHandling:
    """Tests for GPU/CPU device handling."""

    def test_update_handles_device_change(self):
        """Test that update handles tensors on different devices gracefully."""
        tracker = EMAActivity()
        
        # First update on CPU
        tracker.update("weight1", torch.randn(10))
        
        # Simulate device change (if CUDA available)
        if torch.cuda.is_available():
            tracker.update("weight1", torch.randn(10).cuda())
            # Should work without error
            activity = tracker.get_activity("weight1")
            assert activity.device.type == "cuda"
        # If no CUDA, at least verify CPU works
        else:
            tracker.update("weight1", torch.randn(10))
            activity = tracker.get_activity("weight1")
            assert activity.device.type == "cpu"


class TestGradientNormalization:
    """Tests for gradient normalization in activity computation."""

    def test_gradient_magnitude_normalization(self):
        """Test that gradients are normalized by max magnitude."""
        tracker = EMAActivity()
        
        # Large gradients
        tracker.update("weight1", torch.tensor([10.0, 5.0, 0.0]))
        activity1 = tracker.get_activity("weight1").clone()
        
        tracker.reset()
        
        # Small gradients (same relative pattern)
        tracker.update("weight1", torch.tensor([0.1, 0.05, 0.0]))
        activity2 = tracker.get_activity("weight1")
        
        # Both should produce same normalized activity
        assert torch.allclose(activity1, activity2, atol=1e-5)

    def test_activity_range_is_zero_to_one(self):
        """Test that activity values are always in [0, 1] range."""
        tracker = EMAActivity()
        
        # Various gradient magnitudes
        for _ in range(10):
            tracker.update("weight1", torch.randn(100) * 100)
            activity = tracker.get_activity("weight1")
            assert (activity >= 0).all()
            assert (activity <= 1).all()
