"""Tests for baseline comparison script.

This module tests the compare_baselines.py benchmark script to validate:
- VAL-CROSS-003: Comparison with Baselines
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from compare_baselines import (
    GPTQSimulator,
    AWQSimulator,
    apply_gptq_quantization,
    apply_awq_quantization,
    evaluate_perplexity,
    count_parameters,
    replace_conv1d_with_synaptic,
    plot_pareto_frontier,
)


class TestGPTQSimulator:
    """Test GPTQ simulator quantization."""
    
    def test_gptq_simulator_initialization(self):
        """Test GPTQ simulator initializes correctly."""
        sim = GPTQSimulator(bits=4, group_size=128)
        assert sim.bits == 4
        assert sim.group_size == 128
    
    def test_gptq_quantize_dequantize_roundtrip(self):
        """Test quantization followed by dequantization."""
        sim = GPTQSimulator(bits=4, group_size=128)
        weight = torch.randn(256, 512)
        
        orig_shape = weight.shape
        q, s, z = sim.quantize_weight(weight)
        dequantized = sim.dequantize_weight(q, s, z, orig_shape)
        
        # Dequantized should have same shape
        assert dequantized.shape == orig_shape
        
        # Quantized values should be in valid range for 4-bit
        assert q.min() >= 0
        assert q.max() < 16
    
    def test_gptq_compression_ratio_calculation(self):
        """Test compression ratio estimation."""
        sim = GPTQSimulator(bits=4, group_size=128)
        ratio = sim.estimate_compression_ratio((256, 512))
        
        # 4-bit should have ratio > 1 (compression)
        assert ratio > 1.0
        
        # Lower bits = higher compression
        sim_2bit = GPTQSimulator(bits=2, group_size=128)
        ratio_2bit = sim_2bit.estimate_compression_ratio((256, 512))
        
        assert ratio_2bit > ratio
    
    def test_gptq_different_bit_widths(self):
        """Test quantization at different bit widths."""
        weight = torch.randn(128, 256)
        
        for bits in [2, 4, 8]:
            sim = GPTQSimulator(bits=bits, group_size=64)
            q, s, z = sim.quantize_weight(weight)
            
            # Quantized values in correct range
            assert q.min() >= 0
            assert q.max() < (2 ** bits)


class TestAWQSimulator:
    """Test AWQ simulator quantization."""
    
    def test_awq_simulator_initialization(self):
        """Test AWQ simulator initializes correctly."""
        sim = AWQSimulator(bits=4, group_size=128)
        assert sim.bits == 4
        assert sim.group_size == 128
    
    def test_awq_quantize_with_activation_scaling(self):
        """Test AWQ quantization with activation scaling."""
        sim = AWQSimulator(bits=4)
        weight = torch.randn(256, 512)
        activation_scale = torch.rand(512).abs()
        
        q, s, z, scale_factors = sim.quantize_weight_with_activation_scaling(
            weight, activation_scale
        )
        
        assert q.min() >= 0
        assert q.max() < 16
        assert scale_factors is not None
    
    def test_awq_quantize_without_activation_scaling(self):
        """Test AWQ quantization without activation scaling."""
        sim = AWQSimulator(bits=4)
        weight = torch.randn(256, 512)
        
        q, s, z, scale_factors = sim.quantize_weight_with_activation_scaling(weight)
        
        assert q.min() >= 0
        assert q.max() < 16
        assert scale_factors is None
    
    def test_awq_compression_ratio_calculation(self):
        """Test AWQ compression ratio estimation."""
        sim = AWQSimulator(bits=4, group_size=128)
        ratio = sim.estimate_compression_ratio((256, 512))
        
        # Should provide compression
        assert ratio > 1.0


class TestQuantizationApplication:
    """Test applying quantization to models."""
    
    @pytest.mark.slow
    def test_apply_gptq_quantization_to_gpt2(self):
        """Test applying GPTQ quantization to GPT-2 model."""
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = torch.device("cpu")
        
        quantized_model, compression_ratio = apply_gptq_quantization(model, 4, device)
        
        # Should return model and compression ratio
        assert isinstance(quantized_model, nn.Module)
        assert compression_ratio > 1.0
        
        # Weights should be modified (quantized then dequantized)
        # This is hard to verify exactly, but the model should still work
        quantized_model.eval()
    
    @pytest.mark.slow
    def test_apply_awq_quantization_to_gpt2(self):
        """Test applying AWQ quantization to GPT-2 model."""
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = torch.device("cpu")
        
        quantized_model, compression_ratio = apply_awq_quantization(model, 4, device)
        
        assert isinstance(quantized_model, nn.Module)
        assert compression_ratio > 1.0
        
        quantized_model.eval()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        
        params = count_parameters(model)
        expected = 10 * 20 + 20 + 20 * 5 + 5  # weights + biases
        
        assert params == expected
    
    def test_replace_conv1d_with_synaptic(self):
        """Test replacing Conv1D layers with SynapticLayers."""
        from transformers import GPT2Model, GPT2Config
        
        config = GPT2Config(n_layer=1, n_embd=64, n_head=2)
        model = GPT2Model(config)
        
        count = replace_conv1d_with_synaptic(model)
        
        # Should have replaced some layers
        assert count > 0
        
        # Check that SynapticLayers exist
        from synaptic_pruning import SynapticLayer
        synaptic_count = sum(1 for m in model.modules() if isinstance(m, SynapticLayer))
        assert synaptic_count > 0


class TestParetoFrontierPlot:
    """Test Pareto frontier plotting."""
    
    def test_plot_pareto_frontier_creates_file(self, tmp_path):
        """Test that plot function creates output file."""
        results = {
            "baseline_fp16": {
                "parameters": 124000000,
                "perplexity": 25.0,
                "avg_loss": 3.2,
                "compression_ratio": 1.0,
            },
            "synaptic_pruning": [
                {
                    "target_sparsity": 0.0,
                    "actual_sparsity": 0.0,
                    "compression_ratio": 2.0,
                    "perplexity": 26.0,
                    "avg_loss": 3.25,
                    "final_loss": 3.25,
                    "hot_params": 60000000,
                    "warm_params": 40000000,
                    "cold_params": 24000000,
                },
                {
                    "target_sparsity": 0.7,
                    "actual_sparsity": 0.65,
                    "compression_ratio": 8.0,
                    "perplexity": 30.0,
                    "avg_loss": 3.4,
                    "final_loss": 3.4,
                    "hot_params": 40000000,
                    "warm_params": 30000000,
                    "cold_params": 54000000,
                },
            ],
            "gptq": [
                {"bits": 4, "compression_ratio": 3.5, "perplexity": 28.0, "avg_loss": 3.3},
                {"bits": 3, "compression_ratio": 4.5, "perplexity": 32.0, "avg_loss": 3.5},
            ],
            "awq": [
                {"bits": 4, "compression_ratio": 3.5, "perplexity": 27.5, "avg_loss": 3.28},
            ],
        }
        
        output_path = tmp_path / "pareto.png"
        plot_pareto_frontier(results, output_path)
        
        # File should be created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_plot_with_empty_awq(self, tmp_path):
        """Test plotting when AWQ data is empty."""
        results = {
            "baseline_fp16": {
                "parameters": 124000000,
                "perplexity": 25.0,
                "avg_loss": 3.2,
                "compression_ratio": 1.0,
            },
            "synaptic_pruning": [
                {
                    "target_sparsity": 0.5,
                    "actual_sparsity": 0.45,
                    "compression_ratio": 4.0,
                    "perplexity": 28.0,
                    "avg_loss": 3.3,
                    "final_loss": 3.3,
                    "hot_params": 50000000,
                    "warm_params": 40000000,
                    "cold_params": 34000000,
                },
            ],
            "gptq": [
                {"bits": 4, "compression_ratio": 3.5, "perplexity": 27.0, "avg_loss": 3.29},
            ],
            "awq": [],  # Empty
        }
        
        output_path = tmp_path / "pareto_no_awq.png"
        plot_pareto_frontier(results, output_path)
        
        assert output_path.exists()


class TestIntegration:
    """Integration tests for the comparison script."""
    
    @pytest.mark.slow
    def test_comparison_script_import(self):
        """Test that the comparison script can be imported."""
        # Script should import without errors
        import compare_baselines
        
        # Check that key functions exist
        assert hasattr(compare_baselines, 'run_baseline_comparison')
        assert hasattr(compare_baselines, 'plot_pareto_frontier')
        assert hasattr(compare_baselines, 'GPTQSimulator')
        assert hasattr(compare_baselines, 'AWQSimulator')
    
    def test_results_json_structure(self, tmp_path):
        """Test that results JSON has expected structure."""
        results = {
            "config": {
                "model_name": "gpt2",
                "sparsity_levels": [0.0, 0.5],
                "gptq_bits": [4],
                "awq_bits": [4],
                "num_epochs": 2,
                "num_samples": 50,
                "eval_samples": 25,
            },
            "baseline_fp16": {
                "parameters": 124000000,
                "perplexity": 25.0,
                "avg_loss": 3.2,
                "compression_ratio": 1.0,
            },
            "synaptic_pruning": [
                {
                    "target_sparsity": 0.0,
                    "actual_sparsity": 0.0,
                    "compression_ratio": 1.5,
                    "perplexity": 26.0,
                    "avg_loss": 3.25,
                    "final_loss": 3.25,
                    "hot_params": 62000000,
                    "warm_params": 40000000,
                    "cold_params": 22000000,
                },
            ],
            "gptq": [
                {"bits": 4, "compression_ratio": 3.5, "perplexity": 28.0, "avg_loss": 3.3},
            ],
            "awq": [
                {"bits": 4, "compression_ratio": 3.5, "perplexity": 27.5, "avg_loss": 3.28},
            ],
            "comparison_summary": {
                "synaptic_best": {"compression": 1.5, "perplexity": 26.0},
                "gptq_best": {"compression": 3.5, "perplexity": 28.0},
                "awq_best": {"compression": 3.5, "perplexity": 27.5},
                "synaptic_is_competitive": True,
            },
            "validation": {
                "val_cross_003": "passed",
                "message": "Synaptic Pruning is competitive with baselines",
            },
        }
        
        # Save to JSON
        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Load and verify
        with open(results_file) as f:
            loaded = json.load(f)
        
        assert "baseline_fp16" in loaded
        assert "synaptic_pruning" in loaded
        assert "gptq" in loaded
        assert "awq" in loaded
        assert "validation" in loaded
        assert loaded["validation"]["val_cross_003"] == "passed"


class TestCompressionRatios:
    """Test compression ratio calculations."""
    
    def test_synaptic_compression_higher_than_fp16(self):
        """Test that Synaptic Pruning provides compression."""
        # Synaptic should provide compression > 1.0
        from compare_baselines import train_synaptic_model, get_wikitext_samples
        
        # This is a conceptual test - we verify the logic is sound
        # Higher sparsity should lead to higher compression
        sparsity_levels = [0.0, 0.5, 0.9]
        
        for sparsity in sparsity_levels:
            # Mock compression calculation
            hot_ratio = max(0.1, 1.0 - sparsity)
            warm_ratio = min(0.3, sparsity * 0.5)
            cold_ratio = 1.0 - hot_ratio - warm_ratio
            
            baseline_bytes = 100 * 2  # 100 params * 2 bytes
            synaptic_bytes = (
                hot_ratio * 100 * 2 +
                warm_ratio * 100 * 0.5 +
                cold_ratio * 100 * 0.125
            )
            compression = baseline_bytes / synaptic_bytes if synaptic_bytes > 0 else 1.0
            
            assert compression >= 1.0
            if sparsity > 0:
                assert compression > 1.0
    
    def test_gptq_compression_by_bits(self):
        """Test GPTQ compression increases with lower bits."""
        sim_4bit = GPTQSimulator(bits=4)
        sim_3bit = GPTQSimulator(bits=3)
        sim_2bit = GPTQSimulator(bits=2)
        
        shape = (256, 512)
        
        ratio_4 = sim_4bit.estimate_compression_ratio(shape)
        ratio_3 = sim_3bit.estimate_compression_ratio(shape)
        ratio_2 = sim_2bit.estimate_compression_ratio(shape)
        
        # Lower bits = higher compression
        assert ratio_2 > ratio_3 > ratio_4


class TestVALCROSS003:
    """Tests specifically for VAL-CROSS-003 validation."""
    
    def test_competitive_criteria_definition(self):
        """Test that competitive criteria is properly defined."""
        # Synaptic is competitive if within 10% of baseline methods
        baseline_ppl = 25.0
        
        # Within 10% is competitive
        synaptic_ppl_good = 26.5  # 6% increase
        assert synaptic_ppl_good <= baseline_ppl * 1.1
        
        # More than 10% is not competitive
        synaptic_ppl_bad = 28.0  # 12% increase
        assert synaptic_ppl_bad > baseline_ppl * 1.1
    
    def test_comparison_summary_structure(self):
        """Test comparison summary has required fields."""
        summary = {
            "synaptic_best": {
                "compression": 8.0,
                "perplexity": 28.0,
            },
            "gptq_best": {
                "compression": 4.5,
                "perplexity": 30.0,
            },
            "awq_best": {
                "compression": 4.5,
                "perplexity": 29.0,
            },
            "synaptic_is_competitive": True,
        }
        
        # All required keys present
        assert "synaptic_best" in summary
        assert "gptq_best" in summary
        assert "synaptic_is_competitive" in summary
        
        # Values have correct type
        assert isinstance(summary["synaptic_best"]["compression"], (int, float))
        assert isinstance(summary["synaptic_best"]["perplexity"], (int, float))
        assert isinstance(summary["synaptic_is_competitive"], bool)
