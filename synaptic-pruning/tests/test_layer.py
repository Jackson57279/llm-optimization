"""Tests for SynapticLayer - the unified layer combining all components.

This module tests the SynapticLayer implementation, validating:
- VAL-LAY-001: SynapticLayer produces correct output shape
- VAL-LAY-002: Gradients flow through all active weight tiers
- VAL-LAY-003: State save and load works correctly
- VAL-LAY-004: Drop-in nn.Linear replacement compatibility
"""

import pytest
import torch
import torch.nn as nn

from synaptic_pruning.layers import SynapticLayer


class TestSynapticLayerInitialization:
    """Tests for SynapticLayer initialization."""

    def test_default_initialization(self):
        """Test SynapticLayer initializes with correct defaults."""
        layer = SynapticLayer(768, 3072)
        assert layer.in_features == 768
        assert layer.out_features == 3072
        assert layer.bias is not None
        assert layer.weight.shape == (3072, 768)

    def test_custom_initialization(self):
        """Test SynapticLayer accepts custom parameters."""
        layer = SynapticLayer(
            in_features=512,
            out_features=1024,
            bias=False,
            decay=0.95,
            hot_threshold=0.75,
            warm_threshold=0.25,
        )
        assert layer.in_features == 512
        assert layer.out_features == 1024
        assert layer.bias is None
        assert layer.activity_tracker.decay == 0.95
        assert layer.activity_tracker.hot_threshold == 0.75
        assert layer.activity_tracker.warm_threshold == 0.25

    def test_weight_initialization(self):
        """Test that weights are initialized properly."""
        layer = SynapticLayer(64, 128)
        # Weights should be finite and reasonable
        assert torch.isfinite(layer.weight).all()
        assert layer.weight.shape == (128, 64)

    def test_bias_initialization(self):
        """Test that bias is initialized properly."""
        layer = SynapticLayer(64, 128, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (128,)
        assert torch.isfinite(layer.bias).all()

    def test_no_bias(self):
        """Test that bias can be disabled."""
        layer = SynapticLayer(64, 128, bias=False)
        assert layer.bias is None


class TestSynapticLayerForwardPass:
    """Tests for SynapticLayer forward pass (VAL-LAY-001)."""

    def test_forward_output_shape_2d(self):
        """VAL-LAY-001: SynapticLayer produces correct output shape for 2D input."""
        layer = SynapticLayer(768, 3072)
        x = torch.randn(4, 768)  # batch=4, features=768
        output = layer(x)
        assert output.shape == (4, 3072)

    def test_forward_output_shape_1d(self):
        """VAL-LAY-001: SynapticLayer handles 1D input."""
        layer = SynapticLayer(768, 3072)
        x = torch.randn(768)  # single input
        output = layer(x)
        assert output.shape == (3072,)

    def test_forward_output_shape_3d(self):
        """VAL-LAY-001: SynapticLayer handles 3D input (e.g., sequences)."""
        layer = SynapticLayer(768, 3072)
        x = torch.randn(2, 10, 768)  # batch=2, seq=10, features=768
        output = layer(x)
        assert output.shape == (2, 10, 3072)

    def test_forward_without_bias(self):
        """VAL-LAY-001: Forward pass works without bias."""
        layer = SynapticLayer(64, 128, bias=False)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 128)

    def test_forward_with_bias(self):
        """VAL-LAY-001: Forward pass works with bias."""
        layer = SynapticLayer(64, 128, bias=True)
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.shape == (4, 128)

    def test_forward_matches_linear_without_quantization(self):
        """Test that SynapticLayer matches nn.Linear when no quantization applied."""
        # Create layers with same parameters
        torch.manual_seed(42)
        synaptic = SynapticLayer(64, 128, bias=True)
        
        # Copy weights to a standard Linear layer
        torch.manual_seed(42)
        linear = nn.Linear(64, 128, bias=True)
        
        # Manually copy weights and bias
        with torch.no_grad():
            linear.weight.copy_(synaptic.weight)
            if synaptic.bias is not None and linear.bias is not None:
                linear.bias.copy_(synaptic.bias)
        
        x = torch.randn(4, 64)
        
        synaptic_output = synaptic(x)
        linear_output = linear(x)
        
        # Outputs should match when no quantization is applied
        assert torch.allclose(synaptic_output, linear_output, atol=1e-5)


class TestSynapticLayerBackwardPass:
    """Tests for gradient flow (VAL-LAY-002)."""

    def test_gradients_flow_to_weights(self):
        """VAL-LAY-002: Gradients flow to weight parameters."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(4, 64, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Weight gradients should exist and be non-zero
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0

    def test_gradients_flow_to_bias(self):
        """VAL-LAY-002: Gradients flow to bias parameters."""
        layer = SynapticLayer(64, 128, bias=True)
        x = torch.randn(4, 64)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Bias gradients should exist and be non-zero
        assert layer.bias is not None
        assert layer.bias.grad is not None
        assert layer.bias.grad.abs().sum() > 0

    def test_gradients_flow_to_input(self):
        """VAL-LAY-002: Gradients flow back to input."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(4, 64, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Input gradients should exist and be non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradients_with_quantized_weights(self):
        """VAL-LAY-002: Gradients flow through quantized weights via STE."""
        layer = SynapticLayer(64, 128)
        
        # Manually set activity to force quantization of some weights
        # Set all weights to warm tier (will be 4-bit quantized)
        layer.activity_tracker.activity_scores["weight"] = torch.full(
            (128, 64), 0.5  # Warm tier
        )
        
        x = torch.randn(4, 64, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Gradients should still flow to weights via STE
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0


class TestSynapticLayerActivityTracking:
    """Tests for activity tracking integration."""

    def test_activity_tracker_initialized(self):
        """Test that activity tracker is properly initialized."""
        layer = SynapticLayer(64, 128)
        assert layer.activity_tracker is not None
        assert hasattr(layer.activity_tracker, 'update')
        assert hasattr(layer.activity_tracker, 'get_activity')

    def test_activity_updated_on_backward(self):
        """Test that activity is updated during backward pass."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(4, 64, requires_grad=True)
        
        # Before backward, activity scores should be empty
        assert len(layer.activity_tracker.activity_scores) == 0
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # After backward, activity scores should be populated
        assert "weight" in layer.activity_tracker.activity_scores
        assert layer.activity_tracker.activity_scores["weight"].shape == (128, 64)

    def test_quantizer_initialized(self):
        """Test that quantizer is properly initialized."""
        layer = SynapticLayer(64, 128)
        assert layer.quantizer is not None
        assert hasattr(layer.quantizer, 'apply_tiered_quantization')


class TestSynapticLayerStateManagement:
    """Tests for state save/load (VAL-LAY-003)."""

    def test_state_dict_contains_all_components(self):
        """VAL-LAY-003: State dict contains all layer components."""
        layer = SynapticLayer(64, 128)
        
        # Forward and backward to populate activity
        x = torch.randn(4, 64)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        state = layer.state_dict()
        
        # Should contain standard nn.Module state
        assert "weight" in state
        assert "bias" in state
        
        # Should contain activity tracker state
        assert "activity_tracker" in state
        
        # Should contain quantizer state
        assert "quantizer" in state

    def test_load_state_dict_restores_weights(self):
        """VAL-LAY-003: Loading state restores weights correctly."""
        layer1 = SynapticLayer(64, 128)
        
        # Save state
        state = layer1.state_dict()
        
        # Create new layer and load
        layer2 = SynapticLayer(64, 128)
        layer2.load_state_dict(state)
        
        # Weights should match
        assert torch.allclose(layer1.weight, layer2.weight)
        if layer1.bias is not None:
            assert torch.allclose(layer1.bias, layer2.bias)

    def test_load_state_dict_restores_activity(self):
        """VAL-LAY-003: Loading state restores activity scores."""
        layer1 = SynapticLayer(64, 128)
        
        # Populate activity
        x = torch.randn(4, 64)
        output = layer1(x)
        loss = output.sum()
        loss.backward()
        
        # Save state
        state = layer1.state_dict()
        
        # Create new layer and load
        layer2 = SynapticLayer(64, 128)
        layer2.load_state_dict(state)
        
        # Activity should match
        assert "weight" in layer2.activity_tracker.activity_scores
        assert torch.allclose(
            layer1.activity_tracker.get_activity("weight"),
            layer2.activity_tracker.get_activity("weight")
        )

    def test_save_load_produces_same_output(self):
        """VAL-LAY-003: Same input produces same output after save/load."""
        layer1 = SynapticLayer(64, 128)
        
        # Train a bit to populate state
        for _ in range(3):
            x = torch.randn(4, 64)
            output = layer1(x)
            loss = output.sum()
            loss.backward()
        
        # Save state
        state = layer1.state_dict()
        
        # Create new layer and load
        layer2 = SynapticLayer(64, 128)
        layer2.load_state_dict(state)
        
        # Same input should produce same output
        x = torch.randn(4, 64)
        with torch.no_grad():
            output1 = layer1(x)
            output2 = layer2(x)
        
        assert torch.allclose(output1, output2, atol=1e-5)


class TestSynapticLayerNNLinearCompatibility:
    """Tests for nn.Linear compatibility (VAL-LAY-004)."""

    def test_same_constructor_signature(self):
        """VAL-LAY-004: SynapticLayer accepts same constructor args as nn.Linear."""
        # Both should accept in_features, out_features, bias
        linear = nn.Linear(64, 128, bias=True)
        synaptic = SynapticLayer(64, 128, bias=True)
        
        assert linear.in_features == synaptic.in_features
        assert linear.out_features == synaptic.out_features

    def test_same_forward_interface(self):
        """VAL-LAY-004: SynapticLayer has same forward interface as nn.Linear."""
        linear = nn.Linear(64, 128)
        synaptic = SynapticLayer(64, 128)
        
        x = torch.randn(4, 64)
        
        # Both should accept the same input
        linear_output = linear(x)
        synaptic_output = synaptic(x)
        
        # Both should produce same shape output
        assert linear_output.shape == synaptic_output.shape

    def test_swappable_in_model(self):
        """VAL-LAY-004: Can replace nn.Linear with SynapticLayer in a model."""
        class SimpleModel(nn.Module):
            def __init__(self, layer_class):
                super().__init__()
                self.layer1 = layer_class(64, 128)
                self.layer2 = layer_class(128, 64)
            
            def forward(self, x):
                x = self.layer1(x)
                x = torch.relu(x)
                x = self.layer2(x)
                return x
        
        # Create models with both layer types
        x = torch.randn(4, 64)
        
        model_linear = SimpleModel(nn.Linear)
        output_linear = model_linear(x)
        
        model_synaptic = SimpleModel(SynapticLayer)
        output_synaptic = model_synaptic(x)
        
        # Both should work and produce same shape output
        assert output_linear.shape == (4, 64)
        assert output_synaptic.shape == (4, 64)

    def test_has_weight_attribute(self):
        """VAL-LAY-004: SynapticLayer has weight attribute like nn.Linear."""
        linear = nn.Linear(64, 128)
        synaptic = SynapticLayer(64, 128)
        
        # Both should have weight attribute
        assert hasattr(linear, 'weight')
        assert hasattr(synaptic, 'weight')
        
        # Weights should have same shape
        assert linear.weight.shape == synaptic.weight.shape

    def test_has_bias_attribute(self):
        """VAL-LAY-004: SynapticLayer has bias attribute like nn.Linear."""
        linear = nn.Linear(64, 128, bias=True)
        synaptic = SynapticLayer(64, 128, bias=True)
        
        # Both should have bias attribute
        assert hasattr(linear, 'bias')
        assert hasattr(synaptic, 'bias')
        
        # Both should have same bias shape (when bias is present)
        if linear.bias is not None and synaptic.bias is not None:
            assert linear.bias.shape == synaptic.bias.shape


class TestSynapticLayerEdgeCases:
    """Tests for edge cases."""

    def test_single_batch(self):
        """Test with batch size 1."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(1, 64)
        output = layer(x)
        assert output.shape == (1, 128)

    def test_large_batch(self):
        """Test with large batch size."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(1000, 64)
        output = layer(x)
        assert output.shape == (1000, 128)

    def test_small_dimensions(self):
        """Test with small dimensions."""
        layer = SynapticLayer(2, 4)
        x = torch.randn(1, 2)
        output = layer(x)
        assert output.shape == (1, 4)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        layer = SynapticLayer(4096, 4096)
        x = torch.randn(2, 4096)
        output = layer(x)
        assert output.shape == (2, 4096)

    def test_training_mode(self):
        """Test behavior in training vs eval mode."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(4, 64)
        
        # Training mode
        layer.train()
        output_train = layer(x)
        
        # Eval mode
        layer.eval()
        with torch.no_grad():
            output_eval = layer(x)
        
        # Both should produce valid outputs
        assert torch.isfinite(output_train).all()
        assert torch.isfinite(output_eval).all()

    def test_device_transfer(self):
        """Test layer can be moved to different devices."""
        layer = SynapticLayer(64, 128)
        
        # Check CPU
        assert layer.weight.device.type == "cpu"
        
        x = torch.randn(4, 64)
        output = layer(x)
        assert output.device.type == "cpu"

    def test_half_precision(self):
        """Test with half precision input."""
        layer = SynapticLayer(64, 128)
        x = torch.randn(4, 64, dtype=torch.float16)
        
        # Layer weights are float32 by default
        output = layer(x.float())
        assert output.shape == (4, 128)


class TestSynapticLayerIntegration:
    """Integration tests for SynapticLayer."""

    def test_full_training_step(self):
        """Test a complete training step."""
        layer = SynapticLayer(64, 128)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        
        x = torch.randn(4, 64)
        target = torch.randn(4, 128)
        
        # Forward pass
        output = layer(x)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert loss.item() >= 0

    def test_multiple_backward_passes(self):
        """Test multiple backward passes update activity correctly."""
        layer = SynapticLayer(64, 128)
        
        for i in range(5):
            x = torch.randn(4, 64)
            output = layer(x)
            loss = output.sum()
            loss.backward()
        
        # Activity scores should be populated
        assert "weight" in layer.activity_tracker.activity_scores
        activity = layer.activity_tracker.get_activity("weight")
        assert activity.shape == (128, 64)
        assert (activity >= 0).all() and (activity <= 1).all()

    def test_compression_stats(self):
        """Test that compression stats can be computed."""
        layer = SynapticLayer(64, 128)
        
        # Populate some activity
        for _ in range(3):
            x = torch.randn(4, 64)
            output = layer(x)
            loss = output.sum()
            loss.backward()
        
        # Should be able to get tier counts
        hot, warm, cold = layer.activity_tracker.get_tier_counts("weight")
        total = hot + warm + cold
        assert total == 128 * 64
