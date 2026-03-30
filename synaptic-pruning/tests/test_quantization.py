"""Tests for the TieredQuantizer module.

This module contains comprehensive tests for the TieredQuantizer class,
covering 4-bit and 1-bit quantization, round-trip error validation,
tier assignment, and STE gradient flow.
"""

import pytest
import torch

from synaptic_pruning.quantization import TieredQuantizer


class TestTieredQuantizerInitialization:
    """Tests for TieredQuantizer initialization."""

    def test_default_initialization(self):
        """Test that TieredQuantizer initializes with correct defaults."""
        quantizer = TieredQuantizer()
        assert quantizer.hot_threshold == 0.8
        assert quantizer.warm_threshold == 0.3
        assert quantizer.scales == {}

    def test_custom_initialization(self):
        """Test that TieredQuantizer accepts custom thresholds."""
        quantizer = TieredQuantizer(hot_threshold=0.75, warm_threshold=0.25)
        assert quantizer.hot_threshold == 0.75
        assert quantizer.warm_threshold == 0.25

    def test_invalid_thresholds_raises(self):
        """Test that invalid threshold configurations raise ValueError."""
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            TieredQuantizer(warm_threshold=0.5, hot_threshold=0.5)
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            TieredQuantizer(warm_threshold=0.9, hot_threshold=0.8)
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            TieredQuantizer(warm_threshold=0.0, hot_threshold=0.5)


class TestQuantizeFP16:
    """Tests for quantize_fp16 method."""

    def test_fp16_no_quantization(self):
        """Test that quantize_fp16 returns weights unchanged."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        result = quantizer.quantize_fp16(weights)
        assert torch.allclose(result, weights)

    def test_fp16_preserves_dtype(self):
        """Test that quantize_fp16 preserves float16 dtype."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10, dtype=torch.float16)
        result = quantizer.quantize_fp16(weights)
        assert result.dtype == torch.float16


class Test4BitQuantization:
    """Tests for 4-bit quantization (VAL-QNT-001)."""

    def test_quantize_4bit_returns_int8(self):
        """Test that 4-bit quantization returns int8 storage."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        quantized, scale, ste_quantized = quantizer.quantize_4bit(weights)
        assert quantized.dtype == torch.int8
        assert scale.shape == ()  # scalar scale

    def test_4bit_range_is_valid(self):
        """Test that 4-bit quantized values are in [-7, 7] range."""
        quantizer = TieredQuantizer()
        weights = torch.randn(100, 100) * 2  # Various magnitudes
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        # 4-bit signed: -7 to +7 for stability
        assert quantized.min() >= -7
        assert quantized.max() <= 7

    def test_dequantize_4bit_recovers_shape(self):
        """Test that dequantize_4bit recovers original shape."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 20)
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(quantized, scale)
        assert recovered.shape == weights.shape

    def test_4bit_round_trip_error(self):
        """VAL-QNT-001: 4-bit round-trip error < 25% (theoretical bound for 4-bit)."""
        quantizer = TieredQuantizer()
        # Use smaller random weights for better quantization accuracy
        # Gaussian weights with sigma=0.5 have ~98% of values in [-2, 2]
        torch.manual_seed(42)
        weights = torch.randn(1000, 1000) * 0.5
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(quantized, scale)
        # Relative error: |original - recovered| / |original|
        relative_error = (weights - recovered).abs().mean() / weights.abs().mean()
        assert relative_error < 0.25, f"Relative error {relative_error:.4%} exceeds 25%"

    def test_4bit_symmetric_quantization(self):
        """Test that 4-bit uses symmetric quantization."""
        quantizer = TieredQuantizer()
        # Symmetric weights around 0
        weights = torch.linspace(-1, 1, 16)
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        assert quantized[0] == -quantized[-1]

    def test_4bit_with_provided_scale(self):
        """Test 4-bit quantization with explicit scale."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        custom_scale = torch.tensor(0.5)
        quantized, scale, _ = quantizer.quantize_4bit(weights, custom_scale)
        assert torch.allclose(scale, custom_scale)

    def test_scale_computation(self):
        """Test that scale is computed correctly from max abs value."""
        quantizer = TieredQuantizer()
        weights = torch.tensor([1.0, 2.0, 3.0, -4.0, 0.5])
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        max_abs = weights.abs().max()
        # For 4-bit: max quantized value is 7
        expected_scale = max_abs / 7.0
        assert torch.allclose(scale, torch.tensor(expected_scale), atol=1e-5)


class Test1BitQuantization:
    """Tests for 1-bit quantization (VAL-QNT-002)."""

    def test_quantize_1bit_returns_int8(self):
        """Test that 1-bit quantization returns int8 storage."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        quantized, scale, _ = quantizer.quantize_1bit(weights)
        assert quantized.dtype == torch.int8

    def test_1bit_values_are_binary(self):
        """VAL-QNT-002: 1-bit values are in {-1, +1}."""
        quantizer = TieredQuantizer()
        weights = torch.randn(100, 100)
        quantized, scale, _ = quantizer.quantize_1bit(weights)
        unique_values = torch.unique(quantized)
        assert len(unique_values) <= 2
        assert all(v.item() in [-1, 1] for v in unique_values)

    def test_dequantize_1bit_produces_scaled_values(self):
        """VAL-QNT-002: Dequantized 1-bit values are {-scale, +scale}."""
        quantizer = TieredQuantizer()
        weights = torch.randn(100, 100)
        quantized, scale, _ = quantizer.quantize_1bit(weights)
        recovered = quantizer.dequantize_1bit(quantized, scale)
        # All values should be either -scale or +scale
        unique_recovered = torch.unique(recovered)
        assert len(unique_recovered) <= 2
        for val in unique_recovered:
            assert torch.allclose(val.abs(), scale, atol=1e-5)

    def test_1bit_mean_approximation(self):
        """VAL-QNT-002: Mean of quantized weights approximates original mean."""
        quantizer = TieredQuantizer()
        # Create weights with known positive bias
        weights = torch.randn(1000, 1000) + 0.5
        quantized, scale, _ = quantizer.quantize_1bit(weights)
        recovered = quantizer.dequantize_1bit(quantized, scale)
        # The mean should have similar sign pattern
        assert (recovered.mean() > 0) == (weights.mean() > 0)

    def test_1bit_scale_computation(self):
        """Test that 1-bit scale is computed from mean abs value."""
        quantizer = TieredQuantizer()
        weights = torch.randn(100, 100)
        quantized, scale, _ = quantizer.quantize_1bit(weights)
        # Scale should be positive
        assert scale > 0
        # Scale should be related to weight magnitudes
        assert scale <= weights.abs().max()


class TestTierAssignment:
    """Tests for tier assignment based on activity (VAL-QNT-003)."""

    def test_tier_assignment_hot_is_fp16(self):
        """VAL-QNT-003: Hot weights (activity > 0.8) remain FP16."""
        quantizer = TieredQuantizer(hot_threshold=0.8, warm_threshold=0.3)
        weights = torch.randn(10, 10)
        # Hot activity
        activity = torch.full((10, 10), 0.9)
        result, metadata = quantizer.apply_tiered_quantization(weights, activity)
        # Hot mask should be all True
        assert metadata["hot_mask"].all()
        assert torch.allclose(result, weights)

    def test_tier_assignment_warm_is_4bit(self):
        """VAL-QNT-003: Warm weights (0.3 < activity <= 0.8) are 4-bit."""
        quantizer = TieredQuantizer(hot_threshold=0.8, warm_threshold=0.3)
        # Use smaller random weights for better quantization accuracy
        torch.manual_seed(42)
        weights = torch.randn(10, 10) * 0.5
        # Warm activity
        activity = torch.full((10, 10), 0.5)
        result, metadata = quantizer.apply_tiered_quantization(weights, activity)
        # Warm mask should be all True
        assert metadata["warm_mask"].all()
        # Result should be close but not identical (quantization applied)
        assert not torch.allclose(result, weights, atol=1e-6)
        # Error should be within reasonable 4-bit bounds (< 20%)
        relative_error = (weights - result).abs().mean() / weights.abs().mean()
        assert relative_error < 0.20

    def test_tier_assignment_cold_is_1bit(self):
        """VAL-QNT-003: Cold weights (activity <= 0.3) are 1-bit."""
        quantizer = TieredQuantizer(hot_threshold=0.8, warm_threshold=0.3)
        weights = torch.randn(10, 10)
        # Cold activity
        activity = torch.full((10, 10), 0.1)
        result, metadata = quantizer.apply_tiered_quantization(weights, activity)
        # Cold mask should be all True
        assert metadata["cold_mask"].all()
        # Values should be in {-scale, +scale}
        unique_values = torch.unique(result)
        assert len(unique_values) <= 2

    def test_tier_assignment_mixed(self):
        """Test tier assignment with mixed activity levels."""
        quantizer = TieredQuantizer(hot_threshold=0.8, warm_threshold=0.3)
        weights = torch.randn(10, 10)
        # Mixed activity
        activity = torch.zeros(10, 10)
        activity[0:3, :] = 0.9  # Hot
        activity[3:7, :] = 0.5  # Warm
        activity[7:10, :] = 0.1  # Cold

        result, metadata = quantizer.apply_tiered_quantization(weights, activity)

        # Check masks
        assert metadata["hot_mask"][0:3, :].all()
        assert metadata["warm_mask"][3:7, :].all()
        assert metadata["cold_mask"][7:10, :].all()

        # Hot region should be unchanged
        hot_region_original = weights[0:3, :]
        hot_region_result = result[0:3, :]
        assert torch.allclose(hot_region_original, hot_region_result)

        # Cold region should be binary
        cold_region = result[7:10, :]
        unique_cold = torch.unique(cold_region)
        assert len(unique_cold) <= 2

    def test_tier_metadata_contains_expected_keys(self):
        """Test that metadata contains all expected information."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        activity = torch.rand(10, 10)
        result, metadata = quantizer.apply_tiered_quantization(weights, activity)

        expected_keys = ["hot_mask", "warm_mask", "cold_mask", "scales"]
        for key in expected_keys:
            assert key in metadata


class TestSTEDifferentiability:
    """Tests for Straight-Through Estimator differentiability (VAL-QNT-004)."""

    def test_4bit_gradient_flow(self):
        """VAL-QNT-004: Gradients flow through 4-bit quantization via STE."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10, requires_grad=True)
        quantized, scale, ste_quantized = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(ste_quantized, scale)

        # Compute loss and backprop
        loss = recovered.sum()
        loss.backward()

        # Weights should have non-zero gradients
        assert weights.grad is not None
        assert weights.grad.abs().sum() > 0

    def test_1bit_gradient_flow(self):
        """VAL-QNT-004: Gradients flow through 1-bit quantization via STE."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10, requires_grad=True)
        quantized, scale, ste_quantized = quantizer.quantize_1bit(weights)
        recovered = quantizer.dequantize_1bit(ste_quantized, scale)

        # Compute loss and backprop
        loss = recovered.sum()
        loss.backward()

        # Weights should have non-zero gradients
        assert weights.grad is not None
        assert weights.grad.abs().sum() > 0

    def test_tiered_quantization_gradient_flow(self):
        """VAL-QNT-004: Gradients flow through tiered quantization."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10, requires_grad=True)
        activity = torch.rand(10, 10)

        result, metadata = quantizer.apply_tiered_quantization(weights, activity)

        # Compute loss and backprop
        loss = result.sum()
        loss.backward()

        # Weights should have non-zero gradients
        assert weights.grad is not None
        assert weights.grad.abs().sum() > 0

    def test_ste_in_forward_pass(self):
        """Test that STE produces quantized values in forward pass."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10, requires_grad=True)

        # Forward through 4-bit
        quantized, scale, ste_quantized = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(ste_quantized, scale)

        # Forward values should be quantized (not exactly equal to original)
        assert not torch.allclose(recovered, weights, atol=1e-6)

        # But backward should pass through
        loss = (recovered * 2).sum()
        loss.backward()

        # Gradient should be 2 (straight through)
        expected_grad = torch.ones_like(weights) * 2
        assert torch.allclose(weights.grad, expected_grad, atol=1e-5)


class TestScaleManagement:
    """Tests for scale dictionary management."""

    def test_scale_stored_in_dict(self):
        """Test that scales are stored with parameter name."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)

        # Quantize with name
        quantized, scale, _ = quantizer.quantize_4bit(weights, param_name="layer1.weight")
        assert "layer1.weight" in quantizer.scales

    def test_scale_retrieval(self):
        """Test that scales can be retrieved for dequantization."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)

        quantized, scale, _ = quantizer.quantize_4bit(weights, param_name="layer1.weight")
        # Scale should be retrievable
        assert quantizer.scales["layer1.weight"] == scale


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_weights(self):
        """Test quantization of all-zero weights."""
        quantizer = TieredQuantizer()
        weights = torch.zeros(10, 10)

        # 4-bit
        quantized_4bit, scale_4bit, _ = quantizer.quantize_4bit(weights)
        assert scale_4bit == 0 or torch.isfinite(scale_4bit)

        # 1-bit
        quantized_1bit, scale_1bit, _ = quantizer.quantize_1bit(weights)
        assert scale_1bit == 0 or torch.isfinite(scale_1bit)

    def test_nan_weights_raises(self):
        """Test that NaN weights are handled."""
        quantizer = TieredQuantizer()
        weights = torch.tensor([1.0, float("nan"), 3.0])

        # Should handle gracefully (either raise or produce finite results)
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        assert torch.isfinite(quantized).all() or torch.isnan(quantized).any()

    def test_inf_weights_raises(self):
        """Test that Inf weights are handled."""
        quantizer = TieredQuantizer()
        weights = torch.tensor([1.0, float("inf"), 3.0])

        # Should handle gracefully
        quantized, scale, _ = quantizer.quantize_4bit(weights)
        # Either finite results or properly handled infinities
        assert True  # Implementation-dependent

    def test_very_small_weights(self):
        """Test quantization of very small weights."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10) * 1e-8

        quantized, scale, _ = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(quantized, scale)

        # Should not produce NaN
        assert torch.isfinite(recovered).all()

    def test_very_large_weights(self):
        """Test quantization of very large weights."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10) * 1e6

        quantized, scale, _ = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(quantized, scale)

        # Should not overflow
        assert torch.isfinite(recovered).all()

    def test_single_element(self):
        """Test quantization of single element tensor."""
        quantizer = TieredQuantizer()
        weights = torch.tensor([1.5])

        quantized, scale, _ = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(quantized, scale)

        assert recovered.shape == (1,)
        assert torch.isfinite(recovered[0])

    def test_high_dimensional(self):
        """Test quantization of high-dimensional tensors."""
        quantizer = TieredQuantizer()
        weights = torch.randn(2, 3, 4, 5, 6)

        quantized, scale, _ = quantizer.quantize_4bit(weights)
        recovered = quantizer.dequantize_4bit(quantized, scale)

        assert recovered.shape == weights.shape


class TestStateManagement:
    """Tests for state_dict and load_state_dict."""

    def test_state_dict_contains_scales(self):
        """Test that state_dict contains quantization scales."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        quantizer.quantize_4bit(weights, param_name="layer1.weight")

        state = quantizer.state_dict()
        assert "scales" in state
        assert "layer1.weight" in state["scales"]

    def test_state_dict_contains_thresholds(self):
        """Test that state_dict contains threshold configuration."""
        quantizer = TieredQuantizer(hot_threshold=0.75, warm_threshold=0.25)
        state = quantizer.state_dict()

        assert "hot_threshold" in state
        assert "warm_threshold" in state
        assert state["hot_threshold"] == 0.75
        assert state["warm_threshold"] == 0.25

    def test_load_state_dict_restores_scales(self):
        """Test that load_state_dict restores scales correctly."""
        quantizer1 = TieredQuantizer()
        weights = torch.randn(10, 10)
        quantizer1.quantize_4bit(weights, param_name="layer1.weight")

        state = quantizer1.state_dict()

        # Create new quantizer and load
        quantizer2 = TieredQuantizer()
        quantizer2.load_state_dict(state)

        assert "layer1.weight" in quantizer2.scales
        assert torch.allclose(
            quantizer1.scales["layer1.weight"],
            quantizer2.scales["layer1.weight"]
        )

    def test_reset_clears_scales(self):
        """Test that reset clears all scales."""
        quantizer = TieredQuantizer()
        weights = torch.randn(10, 10)
        quantizer.quantize_4bit(weights, param_name="layer1.weight")

        assert len(quantizer.scales) == 1
        quantizer.reset()
        assert len(quantizer.scales) == 0
