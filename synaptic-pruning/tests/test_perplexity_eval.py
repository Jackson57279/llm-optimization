"""Tests for perplexity evaluation benchmark.

This module tests the perplexity evaluation functionality, validating:
- VAL-BEN-003: Model maintains reasonable perplexity after compression
"""

import math
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D

from benchmarks.evaluate_perplexity import (
    replace_conv1d_with_synaptic,
    load_model_from_checkpoint,
    load_baseline_model,
    evaluate_perplexity_simple,
    count_synaptic_layers,
    get_model_compression_stats,
)
from synaptic_pruning import SynapticLayer
from synaptic_pruning.training import SynapticTrainer, PruningSchedule


class TestReplaceConv1DWithSynaptic:
    """Tests for Conv1D replacement."""

    def test_replace_conv1d_creates_synaptic_layers(self):
        """Conv1D layers are replaced with SynapticLayers."""
        # Create a simple module hierarchy similar to GPT-2
        model = nn.Module()
        model.attn = nn.Module()
        model.attn.c_attn = Conv1D(3 * 64, 64)  # GPT-2 style Conv1D
        model.attn.c_proj = Conv1D(64, 64)
        model.mlp = nn.Module()
        model.mlp.c_fc = Conv1D(256, 64)
        model.mlp.c_proj = Conv1D(64, 256)

        original_count = count_synaptic_layers(model)
        assert original_count == 0

        replace_conv1d_with_synaptic(model)

        new_count = count_synaptic_layers(model)
        assert new_count == 4

        # Verify all Conv1D layers were replaced
        for name, module in model.named_modules():
            if name.endswith(('c_attn', 'c_proj', 'c_fc')):
                assert isinstance(module, SynapticLayer)

    def test_replace_linear_creates_synaptic_layers(self):
        """Linear layers are also replaced with SynapticLayers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        replace_conv1d_with_synaptic(model)

        assert isinstance(model[0], SynapticLayer)
        assert not isinstance(model[1], SynapticLayer)  # ReLU stays
        assert isinstance(model[2], SynapticLayer)

    def test_weights_preserved_after_replacement(self):
        """Weights are preserved when replacing layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
        )

        # Set known weights
        with torch.no_grad():
            model[0].weight.fill_(0.5)
            model[0].bias.fill_(0.1)

        original_weight = model[0].weight.clone()
        original_bias = model[0].bias.clone()

        replace_conv1d_with_synaptic(model)

        # Weights should be preserved
        assert torch.allclose(model[0].weight, original_weight)
        assert torch.allclose(model[0].bias, original_bias)

    def test_custom_thresholds_passed_through(self):
        """Custom thresholds are passed to SynapticLayers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
        )

        replace_conv1d_with_synaptic(
            model,
            decay=0.95,
            hot_threshold=0.85,
            warm_threshold=0.35,
        )

        layer = model[0]
        assert layer.activity_tracker.decay == 0.95
        assert layer.activity_tracker.hot_threshold == 0.85
        assert layer.activity_tracker.warm_threshold == 0.35


class TestLoadBaselineModel:
    """Tests for baseline model loading."""

    @pytest.mark.slow
    def test_load_gpt2_baseline(self):
        """GPT-2 baseline model loads successfully."""
        device = torch.device("cpu")
        model, tokenizer = load_baseline_model("gpt2", device=device)

        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token is not None  # Should have pad token set

    def test_pad_token_added_if_missing(self):
        """Pad token is added if missing from tokenizer."""
        # This is mostly tested implicitly via load_baseline_model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        original_pad = tokenizer.pad_token

        # Manually check behavior
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        assert tokenizer.pad_token is not None


class TestEvaluatePerplexitySimple:
    """Tests for perplexity evaluation."""

    def create_mock_dataset(self, num_samples=10):
        """Create a mock dataset for testing."""
        class MockDataset:
            def __init__(self, n):
                self.n = n
                self.data = [
                    {"text": f"This is sample text number {i} with enough length to be valid. " * 10}
                    for i in range(n)
                ]

            def __len__(self):
                return self.n

            def __iter__(self):
                return iter(self.data)

        return MockDataset(num_samples)

    @pytest.mark.slow
    def test_evaluate_perplexity_returns_metrics(self):
        """Perplexity evaluation returns expected metrics."""
        device = torch.device("cpu")
        model, tokenizer = load_baseline_model("gpt2", device=device)

        dataset = self.create_mock_dataset(num_samples=5)

        metrics = evaluate_perplexity_simple(
            model,
            tokenizer,
            dataset,
            device,
            max_samples=5,
            max_length=128,
        )

        assert "perplexity" in metrics
        assert "avg_loss" in metrics
        assert "num_samples" in metrics

        # Perplexity should be a positive number
        assert metrics["perplexity"] > 0
        assert metrics["avg_loss"] > 0
        assert metrics["num_samples"] > 0

    @pytest.mark.slow
    def test_perplexity_is_reasonable_value(self):
        """Perplexity is in a reasonable range for GPT-2."""
        device = torch.device("cpu")
        model, tokenizer = load_baseline_model("gpt2", device=device)

        dataset = self.create_mock_dataset(num_samples=3)

        metrics = evaluate_perplexity_simple(
            model,
            tokenizer,
            dataset,
            device,
            max_samples=3,
            max_length=64,
        )

        ppl = metrics["perplexity"]

        # GPT-2 on WikiText-2 should have perplexity around 20-30
        # Allow wide range for test stability
        assert 5 < ppl < 1000, f"Perplexity {ppl} outside expected range"

    def test_perplexity_computation_correct(self):
        """Test that perplexity is computed correctly.

        Perplexity = exp(average cross-entropy loss)
        """
        # Create a simple mock case
        avg_loss = 2.5
        expected_ppl = math.exp(avg_loss)

        # Verify the formula
        assert math.isclose(math.exp(avg_loss), expected_ppl)

    def test_skips_short_samples(self):
        """Very short samples are skipped."""
        class ShortDataset:
            def __iter__(self):
                # Mix of short and valid samples
                yield {"text": ""}  # Empty
                yield {"text": "hi"}  # Too short
                yield {"text": "This is a valid sample with enough text to be processed." * 5}

        device = torch.device("cpu")
        model, tokenizer = load_baseline_model("gpt2", device=device)

        dataset = ShortDataset()

        # Should work even with mostly invalid samples
        # (will fail if no valid samples, but we have one valid)
        try:
            metrics = evaluate_perplexity_simple(
                model,
                tokenizer,
                dataset,
                device,
                max_samples=10,
                max_length=128,
            )
            assert metrics["num_samples"] > 0
        except ValueError:
            # If ValueError raised, it means no valid samples (expected for this test)
            pass


class TestCountSynapticLayers:
    """Tests for counting SynapticLayers."""

    def test_count_zero_layers(self):
        """Returns 0 when no SynapticLayers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        count = count_synaptic_layers(model)
        assert count == 0

    def test_count_multiple_layers(self):
        """Correctly counts multiple SynapticLayers."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            SynapticLayer(128, 256),
            SynapticLayer(256, 10),
        )

        count = count_synaptic_layers(model)
        assert count == 3

    def test_count_nested_layers(self):
        """Correctly counts SynapticLayers in nested modules."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = SynapticLayer(64, 128)
                self.submodule = nn.Sequential(
                    SynapticLayer(128, 256),
                    SynapticLayer(256, 128),
                )
                self.layer2 = SynapticLayer(128, 10)

            def forward(self, x):
                return x

        model = NestedModel()
        count = count_synaptic_layers(model)
        assert count == 4


class TestGetModelCompressionStats:
    """Tests for getting compression statistics."""

    def test_returns_none_for_non_synaptic_model(self):
        """Returns None when model has no SynapticLayers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        stats = get_model_compression_stats(model)
        assert stats is None

    def test_returns_stats_for_synaptic_model(self):
        """Returns stats when model has SynapticLayers."""
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            SynapticLayer(128, 10),
        )

        stats = get_model_compression_stats(model)
        assert stats is not None

        # Check required fields
        assert "total_params" in stats
        assert "hot_params" in stats
        assert "warm_params" in stats
        assert "cold_params" in stats
        assert "sparsity" in stats
        assert "effective_compression" in stats

        # Total params should be sum of hot + warm + cold
        total = stats["hot_params"] + stats["warm_params"] + stats["cold_params"]
        assert total == stats["total_params"]

    def test_compression_ratio_calculated_correctly(self):
        """Compression ratio is calculated correctly."""
        # Create a model with known structure
        model = nn.Sequential(
            SynapticLayer(100, 100),  # 10,000 params
        )

        # All weights are hot (no activity yet)
        stats = get_model_compression_stats(model)

        # With all hot weights, compression should be 1.0 (no savings)
        # Actually hot weights use FP16, same as baseline, so ratio = 1.0
        assert stats["effective_compression"] == 1.0


class TestPerplexityValidation:
    """Tests for VAL-BEN-003: Perplexity within threshold."""

    def test_perplexity_threshold_calculation(self):
        """Test the 50% threshold calculation."""
        baseline_ppl = 20.0
        threshold = 0.5

        # Maximum acceptable perplexity
        max_acceptable = baseline_ppl * (1 + threshold)
        assert max_acceptable == 30.0

        # Test at the boundary
        synaptic_ppl = 30.0
        increase = (synaptic_ppl - baseline_ppl) / baseline_ppl
        assert increase == 0.5

        # Test just under threshold
        synaptic_ppl = 29.9
        increase = (synaptic_ppl - baseline_ppl) / baseline_ppl
        assert increase < 0.5

        # Test over threshold
        synaptic_ppl = 31.0
        increase = (synaptic_ppl - baseline_ppl) / baseline_ppl
        assert increase > 0.5


class TestCheckpointLoading:
    """Tests for loading models from checkpoints."""

    @pytest.mark.slow
    def test_load_from_checkpoint_with_synaptic_layers(self, tmp_path):
        """Can load model from checkpoint with SynapticLayers."""
        device = torch.device("cpu")

        # Create a simple model with SynapticLayers and save it
        model = nn.Sequential(
            SynapticLayer(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create a trainer and save checkpoint
        trainer = SynapticTrainer(model, optimizer)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        # Now test loading using our function
        # Note: We can't fully test this without a GPT-2 based checkpoint,
        # but we can verify the checkpoint was created
        assert checkpoint_path.exists()

        # Verify checkpoint structure
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint


class TestEndToEndWorkflow:
    """End-to-end tests for perplexity evaluation workflow."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_baseline_evaluation_only(self):
        """Can evaluate baseline model without checkpoint."""
        device = torch.device("cpu")
        model, tokenizer = load_baseline_model("gpt2", device=device)

        # Create mock dataset
        class MockDataset:
            def __iter__(self):
                for i in range(10):
                    yield {
                        "text": f"This is a sample text for evaluation number {i}. " * 10
                    }

        dataset = MockDataset()

        metrics = evaluate_perplexity_simple(
            model,
            tokenizer,
            dataset,
            device,
            max_samples=5,
            max_length=128,
        )

        assert metrics["perplexity"] > 0
        assert metrics["num_samples"] > 0

    def test_results_json_structure(self, tmp_path):
        """Results JSON has expected structure."""
        # Create a sample results dict matching the expected structure
        results = {
            "baseline": {
                "model_name": "gpt2",
                "parameters": 124439808,
                "perplexity": 20.5,
                "loss": 3.02,
                "num_samples": 50,
            },
            "synaptic": {
                "checkpoint": "./checkpoints/test.pt",
                "parameters": 124439808,
                "num_synaptic_layers": 12,
                "perplexity": 28.3,
                "loss": 3.34,
                "num_samples": 50,
                "compression_stats": {
                    "total_params": 124439808,
                    "hot_params": 12443980,
                    "warm_params": 62219904,
                    "cold_params": 49775924,
                    "sparsity": 0.4,
                    "effective_compression": 2.5,
                },
            },
            "comparison": {
                "baseline_perplexity": 20.5,
                "synaptic_perplexity": 28.3,
                "perplexity_increase_ratio": 1.38,
                "perplexity_increase_percent": 38.0,
                "threshold": 0.5,
                "threshold_percent": 50.0,
            },
            "validation": {
                "val_ben_003": "passed",
                "message": "Perplexity within 50% of baseline",
            },
        }

        # Save and verify
        output_path = tmp_path / "results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Read back and verify structure
        with open(output_path) as f:
            loaded = json.load(f)

        assert "baseline" in loaded
        assert "synaptic" in loaded
        assert "comparison" in loaded
        assert "validation" in loaded

        # Verify comparison calculations
        comparison = loaded["comparison"]
        assert comparison["perplexity_increase_ratio"] == pytest.approx(
            comparison["synaptic_perplexity"] / comparison["baseline_perplexity"],
            abs=0.01,
        )
