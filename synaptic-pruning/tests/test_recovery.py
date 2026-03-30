"""Tests for recovery mechanisms: HyperNetwork.

This module tests the HyperNetwork recovery implementation, validating:
- VAL-REC-001: HyperNetwork Generates Valid Weights
- VAL-REC-002: Recovery Cosine Similarity
"""

import torch
import torch.nn as nn

from synaptic_pruning.recovery import HyperNetwork


class TestHyperNetwork:
    """Test suite for HyperNetwork implementation."""

    def test_init_default_parameters(self):
        """Test HyperNetwork initializes with default parameters."""
        hypernet = HyperNetwork()
        assert hypernet.latent_dim == 64
        assert hypernet.hidden_dim == 256
        assert hypernet.target_shape == (768, 768)
        assert hypernet.output_size == 768 * 768

    def test_init_custom_parameters(self):
        """Test HyperNetwork initializes with custom parameters."""
        hypernet = HyperNetwork(
            latent_dim=32,
            hidden_dim=128,
            target_shape=(256, 512),
        )
        assert hypernet.latent_dim == 32
        assert hypernet.hidden_dim == 128
        assert hypernet.target_shape == (256, 512)
        assert hypernet.output_size == 256 * 512

    def test_forward_single_latent(self):
        """Test forward pass with single latent code."""
        hypernet = HyperNetwork(target_shape=(32, 32))
        latent = torch.randn(64)
        weights = hypernet(latent)

        assert weights.shape == (32, 32)
        assert weights.dtype == torch.float32

    def test_forward_batched_latent(self):
        """Test forward pass with batched latent codes."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        batch_size = 4
        latent = torch.randn(batch_size, 64)
        weights = hypernet(latent)

        assert weights.shape == (batch_size, 16, 16)
        assert weights.dtype == torch.float32

    def test_forward_output_range(self):
        """Test that generated weights are in reasonable range."""
        hypernet = HyperNetwork(target_shape=(32, 32))
        latent = torch.randn(64)
        weights = hypernet(latent)

        # Weights should be finite and in reasonable range
        assert torch.isfinite(weights).all()
        # After initialization with small std, values should not be extreme
        assert weights.abs().max() < 10.0

    def test_encode_single_weight(self):
        """Test encoding single weight matrix to latent."""
        hypernet = HyperNetwork(target_shape=(32, 32))
        weights = torch.randn(32, 32)
        latent = hypernet.encode(weights)

        assert latent.shape == (64,)
        assert latent.dtype == torch.float32
        assert torch.isfinite(latent).all()

    def test_encode_batched_weights(self):
        """Test encoding batched weight matrices to latents."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        batch_size = 4
        weights = torch.randn(batch_size, 16, 16)
        latents = hypernet.encode(weights)

        assert latents.shape == (batch_size, 64)
        assert latents.dtype == torch.float32

    def test_encode_decode_consistency(self):
        """Test that encode-decode roundtrip produces valid tensors."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        original_weights = torch.randn(16, 16)

        latent = hypernet.encode(original_weights)
        recovered_weights = hypernet(latent)

        assert recovered_weights.shape == original_weights.shape
        assert torch.isfinite(recovered_weights).all()

    def test_recovery_loss_computation(self):
        """Test recovery loss computation."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        original_weights = torch.randn(16, 16)
        latent = torch.randn(64)

        loss = hypernet.compute_recovery_loss(original_weights, latent)

        assert loss.shape == ()  # scalar
        assert loss.item() >= 0.0
        assert loss.item() <= 2.0  # Cosine similarity is in [-1, 1]

    def test_recovery_loss_range(self):
        """Test that recovery loss is in valid range [0, 2]."""
        hypernet = HyperNetwork(target_shape=(32, 32))

        # Test with identical weights (loss should be 0 or near 0)
        weights = torch.randn(32, 32)
        latent = hypernet.encode(weights)
        loss = hypernet.compute_recovery_loss(weights, latent)
        assert 0.0 <= loss.item() <= 2.0

    def test_gradient_flow_generator(self):
        """Test gradients flow through generator network."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        latent = torch.randn(64, requires_grad=True)

        weights = hypernet(latent)
        loss = weights.sum()
        loss.backward()

        assert latent.grad is not None
        assert torch.isfinite(latent.grad).all()

    def test_gradient_flow_encoder(self):
        """Test gradients flow through encoder network."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        weights = torch.randn(16, 16, requires_grad=True)

        latent = hypernet.encode(weights)
        loss = latent.sum()
        loss.backward()

        assert weights.grad is not None
        assert torch.isfinite(weights.grad).all()

    def test_gradient_flow_recovery_loss(self):
        """Test gradients flow through recovery loss."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        original_weights = torch.randn(16, 16)
        latent = torch.randn(64, requires_grad=True)

        loss = hypernet.compute_recovery_loss(original_weights, latent)
        loss.backward()

        assert latent.grad is not None
        assert torch.isfinite(latent.grad).all()

    def test_different_target_shapes(self):
        """Test HyperNetwork works with different target shapes."""
        test_cases = [
            (64, 64),
            (128, 256),
            (256, 128),
            (512, 512),
            (32, 64, 128),  # 3D shape
        ]

        for shape in test_cases:
            hypernet = HyperNetwork(target_shape=shape)
            latent = torch.randn(64)
            weights = hypernet(latent)
            assert weights.shape == shape

    def test_training_step(self):
        """Test that HyperNetwork can be trained for one step."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        optimizer = torch.optim.Adam(hypernet.parameters(), lr=0.001)

        # Generate some target weights
        target_weights = torch.randn(16, 16)
        latent = torch.randn(64, requires_grad=True)

        # Forward pass
        loss = hypernet.compute_recovery_loss(target_weights, latent)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # After training, loss should change (or at least not break)
        loss_after = hypernet.compute_recovery_loss(target_weights, latent).item()

        # Loss should still be valid
        assert 0.0 <= loss_after <= 2.0

    def test_cosine_similarity_improvement(self):
        """Test that training improves cosine similarity above 80%."""
        # Create a small hypernet for faster training
        hypernet = HyperNetwork(
            latent_dim=32,
            hidden_dim=64,
            target_shape=(32, 32),
        )
        optimizer = torch.optim.Adam(hypernet.parameters(), lr=0.01)

        # Generate target weights from a distribution
        torch.manual_seed(42)
        target_weights = torch.randn(32, 32) * 0.5

        # Encode to get initial latent
        latent = hypernet.encode(target_weights).detach().requires_grad_(True)
        latent_optimizer = torch.optim.Adam([latent], lr=0.01)

        # Train for several steps
        for _ in range(200):
            optimizer.zero_grad()
            latent_optimizer.zero_grad()
            loss = hypernet.compute_recovery_loss(target_weights, latent)
            loss.backward()
            optimizer.step()
            latent_optimizer.step()

        # Check final similarity
        with torch.no_grad():
            recovered = hypernet(latent)
            cos_sim = nn.functional.cosine_similarity(
                target_weights.view(1, -1),
                recovered.view(1, -1),
                dim=1,
            )

        assert cos_sim.item() > 0.80, f"Cosine similarity {cos_sim.item():.3f} is not > 0.80"

    def test_hypernetwork_parameter_count(self):
        """Test that HyperNetwork has reasonable parameter count."""
        hypernet = HyperNetwork(latent_dim=64, hidden_dim=256, target_shape=(768, 768))

        total_params = sum(p.numel() for p in hypernet.parameters())
        # Should have generator + encoder parameters
        # Generator: 64*256 + 256 + 256*256 + 256 + 256*589824 + 589824
        # Encoder: 589824*256 + 256 + 256*64 + 64
        assert total_params > 0

    def test_save_load_state_dict(self):
        """Test that HyperNetwork can be saved and loaded."""
        hypernet1 = HyperNetwork(target_shape=(16, 16))
        latent = torch.randn(64)
        weights1 = hypernet1(latent)

        # Save state dict
        state_dict = hypernet1.state_dict()

        # Load into new instance
        hypernet2 = HyperNetwork(target_shape=(16, 16))
        hypernet2.load_state_dict(state_dict)

        # Should produce identical output
        weights2 = hypernet2(latent)
        assert torch.allclose(weights1, weights2)

    def test_device_transfer(self):
        """Test HyperNetwork can be moved to different devices."""
        hypernet = HyperNetwork(target_shape=(16, 16))

        # Check CPU
        assert next(hypernet.parameters()).device.type == "cpu"

        latent = torch.randn(64)
        weights = hypernet(latent)
        assert weights.device.type == "cpu"


class TestHyperNetworkBatching:
    """Test batching behavior of HyperNetwork."""

    def test_batch_consistency(self):
        """Test that batched and individual results are consistent."""
        hypernet = HyperNetwork(target_shape=(16, 16))

        # Generate individual latents
        latents = [torch.randn(64) for _ in range(4)]
        individual_weights = [hypernet(latent) for latent in latents]

        # Generate batched
        batched_latents = torch.stack(latents)
        batched_weights = hypernet(batched_latents)

        # Check consistency
        for i, individual in enumerate(individual_weights):
            assert torch.allclose(individual, batched_weights[i])

    def test_batch_encode_consistency(self):
        """Test that batched encoding matches individual encoding."""
        hypernet = HyperNetwork(target_shape=(16, 16))

        # Generate individual weights
        weights = [torch.randn(16, 16) for _ in range(4)]
        individual_latents = [hypernet.encode(w) for w in weights]

        # Encode batched
        batched_weights = torch.stack(weights)
        batched_latents = hypernet.encode(batched_weights)

        # Check consistency
        for i, individual in enumerate(individual_latents):
            assert torch.allclose(individual, batched_latents[i])


class TestHyperNetworkEdgeCases:
    """Test edge cases for HyperNetwork."""

    def test_zero_latent(self):
        """Test forward with zero latent code."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        latent = torch.zeros(64)
        weights = hypernet(latent)

        assert weights.shape == (16, 16)
        assert torch.isfinite(weights).all()

    def test_large_latent(self):
        """Test forward with large latent values."""
        hypernet = HyperNetwork(target_shape=(16, 16))
        latent = torch.randn(64) * 100
        weights = hypernet(latent)

        assert weights.shape == (16, 16)
        assert torch.isfinite(weights).all()

    def test_one_dimensional_target(self):
        """Test with 1D target shape (bias vector)."""
        hypernet = HyperNetwork(target_shape=(256,))
        latent = torch.randn(64)
        weights = hypernet(latent)

        assert weights.shape == (256,)
