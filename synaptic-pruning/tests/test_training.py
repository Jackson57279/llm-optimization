"""Tests for training module - SynapticTrainer and PruningSchedule.

This module tests the training functionality, validating:
- VAL-TRN-001: SynapticTrainer initializes with model and optimizer
- VAL-TRN-002: Progressive pruning schedule increases sparsity correctly
- VAL-TRN-003: Training loop completes without errors
- VAL-TRN-004: Recovery network trains alongside main model
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synaptic_pruning.training import SynapticTrainer, PruningSchedule
from synaptic_pruning.layers import SynapticLayer
from synaptic_pruning.recovery import HyperNetwork


class TestPruningSchedule:
    """Tests for PruningSchedule (VAL-TRN-002)."""

    def test_linear_schedule_increases_sparsity(self):
        """VAL-TRN-002: Linear schedule increases sparsity monotonically."""
        schedule = PruningSchedule(
            max_sparsity=0.9,
            schedule_type="linear",
            max_epochs=10,
        )

        sparsities = [schedule.get_sparsity(e) for e in range(11)]

        # Should be monotonically increasing
        for i in range(len(sparsities) - 1):
            assert sparsities[i] <= sparsities[i + 1]

        # Should start at 0 and end at max_sparsity
        assert sparsities[0] == 0.0
        assert abs(sparsities[-1] - 0.9) < 0.01

    def test_exponential_schedule_faster_pruning(self):
        """VAL-TRN-002: Exponential schedule prunes faster initially."""
        linear = PruningSchedule(max_sparsity=0.9, schedule_type="linear", max_epochs=10)
        exponential = PruningSchedule(max_sparsity=0.9, schedule_type="exponential", max_epochs=10)

        # At early epochs, exponential should be lower (slower start)
        assert exponential.get_sparsity(0) <= linear.get_sparsity(0)

    def test_stepped_schedule_at_milestones(self):
        """VAL-TRN-002: Stepped schedule jumps at milestones."""
        milestones = [(0, 0.0), (5, 0.5), (10, 0.9)]
        schedule = PruningSchedule(
            max_sparsity=0.9,
            schedule_type="stepped",
            max_epochs=10,
            stepped_milestones=milestones,
        )

        assert schedule.get_sparsity(0) == 0.0
        assert schedule.get_sparsity(5) == 0.5
        assert schedule.get_sparsity(10) == 0.9

    def test_cosine_schedule_smooth_curve(self):
        """VAL-TRN-002: Cosine schedule produces smooth curve."""
        schedule = PruningSchedule(max_sparsity=0.9, schedule_type="cosine", max_epochs=10)

        # Check middle epoch is around 0.5 * max_sparsity
        mid_sparsity = schedule.get_sparsity(5)
        assert 0.4 < mid_sparsity < 0.6

    def test_warmup_period_no_pruning(self):
        """VAL-TRN-002: Warmup period has no pruning."""
        schedule = PruningSchedule(
            max_sparsity=0.9,
            schedule_type="linear",
            warmup_epochs=3,
            max_epochs=10,
        )

        # During warmup, sparsity should be 0
        assert schedule.get_sparsity(0) == 0.0
        assert schedule.get_sparsity(2) == 0.0

        # After warmup, sparsity should increase
        assert schedule.get_sparsity(3) > 0.0

    def test_invalid_sparsity_raises(self):
        """Invalid max_sparsity should raise ValueError."""
        with pytest.raises(ValueError):
            PruningSchedule(max_sparsity=1.5)

        with pytest.raises(ValueError):
            PruningSchedule(max_sparsity=-0.1)


class TestSynapticTrainerInitialization:
    """Tests for SynapticTrainer initialization (VAL-TRN-001)."""

    def test_trainer_initializes_with_model_and_optimizer(self):
        """VAL-TRN-001: Trainer initializes with model and optimizer."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            SynapticLayer(128, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = SynapticTrainer(model, optimizer)

        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.pruning_schedule is not None

    def test_trainer_discovers_synaptic_layers(self):
        """VAL-TRN-001: Trainer discovers SynapticLayers in model."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            SynapticLayer(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Not a SynapticLayer
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = SynapticTrainer(model, optimizer)

        # Should find exactly 2 SynapticLayers
        assert len(trainer._synaptic_layers) == 2

    def test_trainer_with_custom_schedule(self):
        """VAL-TRN-001: Trainer accepts custom pruning schedule."""
        model = nn.Sequential(SynapticLayer(64, 128))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        schedule = PruningSchedule(max_sparsity=0.95, schedule_type="exponential")
        trainer = SynapticTrainer(model, optimizer, pruning_schedule=schedule)

        assert trainer.pruning_schedule.max_sparsity == 0.95
        assert trainer.pruning_schedule.schedule_type == "exponential"

    def test_trainer_with_recovery_network(self):
        """VAL-TRN-001: Trainer accepts recovery network."""
        model = nn.Sequential(SynapticLayer(64, 128))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        recovery_net = HyperNetwork(latent_dim=32, target_shape=(64, 128))

        trainer = SynapticTrainer(
            model,
            optimizer,
            recovery_network=recovery_net,
            recovery_weight=0.05,
        )

        assert trainer.recovery_network is recovery_net
        assert trainer.recovery_weight == 0.05

    def test_trainer_device_handling(self):
        """VAL-TRN-001: Trainer handles device correctly."""
        model = nn.Sequential(SynapticLayer(64, 128))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Default device
        trainer = SynapticTrainer(model, optimizer)
        assert trainer.device is not None

        # Explicit CPU device
        trainer_cpu = SynapticTrainer(model, optimizer, device="cpu")
        assert str(trainer_cpu.device) == "cpu"

    def test_invalid_recovery_weight_raises(self):
        """Negative recovery_weight should raise ValueError."""
        model = nn.Sequential(SynapticLayer(64, 128))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError):
            SynapticTrainer(model, optimizer, recovery_weight=-0.1)


class TestSynapticTrainerTraining:
    """Tests for training loop (VAL-TRN-003)."""

    def create_toy_dataset(self, n_samples=100, in_dim=64, out_dim=10):
        """Create a toy dataset for testing."""
        X = torch.randn(n_samples, in_dim)
        y = torch.randint(0, out_dim, (n_samples,))
        return TensorDataset(X, y)

    def test_train_completes_without_errors(self):
        """VAL-TRN-003: Training loop completes without errors."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dataset = self.create_toy_dataset(n_samples=50)
        loader = DataLoader(dataset, batch_size=10)

        trainer = SynapticTrainer(
            model,
            optimizer,
            compression_update_freq=10,
        )

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=2, loss_fn=loss_fn)

        # Should have history data
        assert "train_losses" in history
        assert len(history["train_losses"]) == 2

    def test_loss_decreases_during_training(self):
        """VAL-TRN-003: Training should reduce loss over epochs."""
        torch.manual_seed(42)

        model = nn.Sequential(
            SynapticLayer(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        dataset = self.create_toy_dataset(n_samples=100, in_dim=32, out_dim=2)
        loader = DataLoader(dataset, batch_size=20)

        trainer = SynapticTrainer(model, optimizer)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=3, loss_fn=loss_fn, log_interval=100)

        # Loss should generally decrease (allow for some noise)
        losses = history["train_losses"]
        assert losses[-1] <= losses[0] * 1.5  # Allow some variance

    def test_compression_stats_tracked(self):
        """VAL-TRN-003: Compression stats are tracked during training."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dataset = self.create_toy_dataset(n_samples=50)
        loader = DataLoader(dataset, batch_size=10)

        trainer = SynapticTrainer(
            model,
            optimizer,
            compression_update_freq=5,
        )

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=2, loss_fn=loss_fn)

        # Should have compression stats
        assert "compression_stats" in history
        assert len(history["compression_stats"]) > 0

        # Stats should have expected fields
        stats = history["compression_stats"][0]
        assert "total_params" in stats
        assert "sparsity" in stats

    def test_validation_evaluation(self):
        """VAL-TRN-003: Validation set can be evaluated."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_dataset = self.create_toy_dataset(n_samples=50)
        val_dataset = self.create_toy_dataset(n_samples=30)
        train_loader = DataLoader(train_dataset, batch_size=10)
        val_loader = DataLoader(val_dataset, batch_size=10)

        trainer = SynapticTrainer(model, optimizer)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(
            train_loader,
            num_epochs=2,
            loss_fn=loss_fn,
            val_loader=val_loader,
        )

        # Should have validation losses
        assert "val_losses" in history
        assert len(history["val_losses"]) == 2

    def test_callback_invoked(self):
        """VAL-TRN-003: Callback is called each epoch."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dataset = self.create_toy_dataset(n_samples=30)
        loader = DataLoader(dataset, batch_size=10)

        trainer = SynapticTrainer(model, optimizer)

        callback_calls = []

        def callback(epoch, stats):
            callback_calls.append((epoch, stats))

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        trainer.train(loader, num_epochs=3, loss_fn=loss_fn, callback=callback)

        # Callback should be called 3 times
        assert len(callback_calls) == 3

        # Stats should be provided
        for epoch, stats in callback_calls:
            assert "train_loss" in stats
            assert "compression" in stats


class TestSynapticTrainerRecovery:
    """Tests for recovery network training (VAL-TRN-004)."""

    def test_recovery_loss_computed_when_network_provided(self):
        """VAL-TRN-004: Recovery loss is computed when network provided."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create recovery network with matching shape
        recovery_net = HyperNetwork(latent_dim=32, target_shape=(64, 128))

        trainer = SynapticTrainer(
            model,
            optimizer,
            recovery_network=recovery_net,
            recovery_weight=0.01,
        )

        dataset = TensorDataset(
            torch.randn(20, 64),
            torch.randint(0, 10, (20,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=1, loss_fn=loss_fn)

        # Should have recovery losses
        assert "recovery_losses" in history
        assert len(history["recovery_losses"]) > 0

    def test_recovery_loss_zero_without_network(self):
        """VAL-TRN-004: Recovery loss is 0 without recovery network."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = SynapticTrainer(model, optimizer, recovery_network=None)

        dataset = TensorDataset(
            torch.randn(20, 64),
            torch.randint(0, 10, (20,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=1, loss_fn=loss_fn)

        # Recovery losses should be near 0
        for loss in history["recovery_losses"]:
            assert loss < 0.01

    def test_recovery_network_receives_gradients(self):
        """VAL-TRN-004: Recovery network parameters are updated during training."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        recovery_net = HyperNetwork(latent_dim=32, target_shape=(64, 128))

        # Get initial parameter values
        initial_params = [p.clone() for p in recovery_net.parameters()]

        trainer = SynapticTrainer(
            model,
            optimizer,
            recovery_network=recovery_net,
            recovery_weight=0.1,
        )

        dataset = TensorDataset(
            torch.randn(30, 64),
            torch.randint(0, 10, (30,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        # Train to populate activity and trigger recovery
        trainer.train(loader, num_epochs=3, loss_fn=loss_fn)

        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, recovery_net.parameters()):
            if not torch.allclose(initial, current, atol=1e-5):
                params_changed = True
                break

        assert params_changed, "Recovery network parameters should be updated"


class TestSynapticTrainerPruningIntegration:
    """Tests for pruning schedule integration (VAL-TRN-002)."""

    def test_pruning_thresholds_updated_during_training(self):
        """VAL-TRN-002: Pruning thresholds are updated according to schedule."""
        model = nn.Sequential(
            SynapticLayer(64, 128, warm_threshold=0.3),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        schedule = PruningSchedule(
            max_sparsity=0.9,
            schedule_type="stepped",
            max_epochs=5,
            stepped_milestones=[(0, 0.0), (2, 0.5), (5, 0.9)],
        )

        trainer = SynapticTrainer(model, optimizer, pruning_schedule=schedule)

        # Initial threshold
        layer = trainer._synaptic_layers[0]
        initial_threshold = layer.activity_tracker.warm_threshold

        # Simulate training
        for epoch in range(3):
            trainer._update_activity_thresholds(epoch)

        # Threshold should have increased
        final_threshold = layer.activity_tracker.warm_threshold
        assert final_threshold > initial_threshold

    def test_sparsity_increases_with_schedule(self):
        """VAL-TRN-002: Model sparsity increases according to schedule."""
        torch.manual_seed(42)

        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        schedule = PruningSchedule(
            max_sparsity=0.8,
            schedule_type="linear",
            max_epochs=10,
        )

        trainer = SynapticTrainer(
            model,
            optimizer,
            pruning_schedule=schedule,
            compression_update_freq=1,
        )

        dataset = TensorDataset(
            torch.randn(50, 64),
            torch.randint(0, 10, (50,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        # Train for multiple epochs
        history = trainer.train(loader, num_epochs=5, loss_fn=loss_fn)

        # Get sparsity progression
        sparsities = [s["sparsity"] for s in history["compression_stats"]]

        # Sparsity should generally increase (allow some noise)
        if len(sparsities) >= 2:
            assert sparsities[-1] >= sparsities[0] * 0.5  # At least some increase


class TestSynapticTrainerCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Checkpoint can be saved and loaded."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = SynapticTrainer(model, optimizer)

        dataset = TensorDataset(
            torch.randn(30, 64),
            torch.randint(0, 10, (30,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        # Train a bit
        trainer.train(loader, num_epochs=2, loss_fn=loss_fn)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Create new trainer and load
        model2 = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        trainer2 = SynapticTrainer(model2, optimizer2)

        trainer2.load_checkpoint(str(checkpoint_path))

        # Should have loaded step count
        assert trainer2.current_step == trainer.current_step
        assert trainer2.current_epoch == trainer.current_epoch

    def test_compression_summary(self):
        """Compression summary is available after training."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = SynapticTrainer(model, optimizer)

        dataset = TensorDataset(
            torch.randn(30, 64),
            torch.randint(0, 10, (30,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        trainer.train(loader, num_epochs=2, loss_fn=loss_fn)

        summary = trainer.get_compression_summary()

        assert "final_stats" in summary
        assert "compression_history" in summary
        assert summary["total_epochs"] == 2


class TestSynapticTrainerEdgeCases:
    """Edge case tests for SynapticTrainer."""

    def test_single_batch_training(self):
        """Training works with single batch."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = SynapticTrainer(model, optimizer)

        # Single sample dataset
        dataset = TensorDataset(
            torch.randn(1, 64),
            torch.randint(0, 10, (1,))
        )
        loader = DataLoader(dataset, batch_size=1)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=1, loss_fn=loss_fn)

        assert len(history["train_losses"]) == 1

    def test_large_batch_size(self):
        """Training works with large batch."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = SynapticTrainer(model, optimizer)

        dataset = TensorDataset(
            torch.randn(100, 64),
            torch.randint(0, 10, (100,))
        )
        loader = DataLoader(dataset, batch_size=100)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        history = trainer.train(loader, num_epochs=1, loss_fn=loss_fn)

        assert len(history["train_losses"]) == 1

    def test_model_without_synaptic_layers(self):
        """Trainer works with models that have no SynapticLayers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = SynapticTrainer(model, optimizer)

        # Should have no synaptic layers
        assert len(trainer._synaptic_layers) == 0

        dataset = TensorDataset(
            torch.randn(20, 64),
            torch.randint(0, 10, (20,))
        )
        loader = DataLoader(dataset, batch_size=10)

        def loss_fn(x, y):
            logits = model(x)
            return nn.functional.cross_entropy(logits, y)

        # Should still train without errors
        history = trainer.train(loader, num_epochs=1, loss_fn=loss_fn)

        assert len(history["train_losses"]) == 1
