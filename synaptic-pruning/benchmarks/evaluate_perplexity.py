"""Perplexity evaluation on WikiText-2.

This script implements perplexity evaluation to validate:
- VAL-BEN-003: Model maintains reasonable perplexity after compression

Usage:
    python benchmarks/evaluate_perplexity.py --checkpoint ./checkpoints/gpt2_synaptic_20steps.pt
    python benchmarks/evaluate_perplexity.py --model gpt2  # Evaluate baseline model

The script will:
1. Load a model (baseline GPT-2 or from a checkpoint with SynapticLayers)
2. Evaluate on WikiText-2 test set
3. Compute perplexity
4. Compare against baseline and verify within 50% threshold
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D

from synaptic_pruning import SynapticLayer


def replace_conv1d_with_synaptic(
    module: nn.Module,
    decay: float = 0.9,
    hot_threshold: float = 0.8,
    warm_threshold: float = 0.3,
) -> None:
    """Recursively replace Conv1D layers with SynapticLayers.

    GPT-2 uses Conv1D layers which are functionally equivalent to Linear layers
    but with a transposed weight matrix. This function replaces them with
    SynapticLayers that have the same dimensions.

    Args:
        module: The module to modify in-place.
        decay: EMA decay factor for activity tracking.
        hot_threshold: Activity threshold for FP16 tier.
        warm_threshold: Activity threshold for 4-bit tier.
    """
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            # Conv1D in GPT-2 has weight shape (in_features, out_features)
            # and performs x @ weight + bias (transposed operation)
            # We need to create SynapticLayer with transposed dimensions
            in_features = child.weight.shape[0]  # nf in Conv1D terms
            out_features = child.weight.shape[1]   # nx in Conv1D terms

            # Create SynapticLayer (weight will be (out_features, in_features))
            synaptic_layer = SynapticLayer(
                in_features=in_features,
                out_features=out_features,
                bias=child.bias is not None,
                decay=decay,
                hot_threshold=hot_threshold,
                warm_threshold=warm_threshold,
            )

            # Copy weights - Conv1D weight is (in, out), SynapticLayer weight is (out, in)
            with torch.no_grad():
                # Transpose to match SynapticLayer's expected shape
                synaptic_layer.weight.copy_(child.weight.data.t())
                if child.bias is not None:
                    synaptic_layer.bias.copy_(child.bias.data)

            # Replace the module
            setattr(module, name, synaptic_layer)
        elif isinstance(child, nn.Linear):
            # Also handle standard Linear layers (like lm_head)
            synaptic_layer = SynapticLayer(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                decay=decay,
                hot_threshold=hot_threshold,
                warm_threshold=warm_threshold,
            )

            # Copy weights
            with torch.no_grad():
                synaptic_layer.weight.copy_(child.weight.data)
                if child.bias is not None:
                    synaptic_layer.bias.copy_(child.bias.data)

            # Replace the module
            setattr(module, name, synaptic_layer)
        else:
            # Recursively process child modules
            replace_conv1d_with_synaptic(
                child, decay=decay, hot_threshold=hot_threshold, warm_threshold=warm_threshold
            )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, AutoTokenizer, dict[str, Any]]:
    """Load a model from a checkpoint with SynapticLayers.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, tokenizer, checkpoint_info).
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load base model
    model_name = "gpt2"  # Default to gpt2, could be stored in checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Replace Conv1D/Linear layers with SynapticLayers
    replace_conv1d_with_synaptic(model)

    # Load state dict - use strict=False because the checkpoint contains
    # SynapticLayer-specific keys (activity_tracker, quantizer) that will be
    # initialized fresh for evaluation. This is expected behavior.
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    
    # Log any state dict mismatches for debugging
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        # Filter out expected SynapticLayer-specific keys (activity_tracker, quantizer)
        synaptic_keys = [k for k in unexpected_keys if 'activity_tracker' in k or 'quantizer' in k]
        other_keys = [k for k in unexpected_keys if 'activity_tracker' not in k and 'quantizer' not in k]
        if synaptic_keys:
            print(f"Info: Ignoring {len(synaptic_keys)} SynapticLayer-specific keys "
                  f"(activity_tracker/quantizer will use fresh initialization)")
        if other_keys:
            print(f"Warning: Unexpected keys in checkpoint: {other_keys}")
    
    model = model.to(device)
    model.eval()

    # Extract checkpoint info
    checkpoint_info = {
        "path": checkpoint_path,
        "current_step": checkpoint.get("current_step", 0),
        "current_epoch": checkpoint.get("current_epoch", 0),
    }

    # Get compression stats if available in checkpoint
    compression_stats = checkpoint.get("compression_stats", [])
    if compression_stats:
        checkpoint_info["final_compression_stats"] = compression_stats[-1]

    return model, tokenizer, checkpoint_info


def load_baseline_model(
    model_name: str = "gpt2",
    device: torch.device | None = None,
) -> tuple[nn.Module, AutoTokenizer]:
    """Load a baseline (non-synaptic) model.

    Args:
        model_name: HuggingFace model name.
        device: Device to load the model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading baseline model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device is not None:
        model = model.to(device)

    model.eval()

    return model, tokenizer


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Any,
    device: torch.device,
    max_samples: int | None = None,
    max_length: int = 512,
    batch_size: int = 1,
    stride: int = 512,
) -> dict[str, float]:
    """Evaluate perplexity on a dataset.

    Uses sliding window approach for accurate perplexity calculation on long texts.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer for encoding text.
        dataset: The dataset to evaluate on.
        device: Device to run evaluation on.
        max_samples: Maximum number of samples to evaluate (None for all).
        max_length: Maximum sequence length for each window.
        batch_size: Batch size for evaluation.
        stride: Stride for sliding window (set to max_length for non-overlapping windows).

    Returns:
        Dictionary with perplexity and other metrics.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_samples = 0

    encodings_list = []

    # Prepare all samples
    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        text = example.get("text", "")
        if len(text.strip()) < 50:  # Skip very short/empty samples
            continue

        # Tokenize
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length * 4,  # Allow longer texts that will be chunked
        )

        if encoding["input_ids"].shape[1] < 10:  # Skip very short sequences
            continue

        encodings_list.append(encoding["input_ids"])
        num_samples += 1

    if num_samples == 0:
        raise ValueError("No valid samples found in dataset")

    print(f"Evaluating on {num_samples} samples...")

    with torch.no_grad():
        for encoding in tqdm(encodings_list, desc="Evaluating"):
            seq_len = encoding.shape[1]

            # Sliding window evaluation
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - begin_loc

                input_ids = encoding[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()

                # Don't compute loss for the first stride tokens (no context)
                if begin_loc > 0:
                    target_ids[:, :-trg_len] = -100

                outputs = model(input_ids, labels=target_ids)
                loss = outputs.loss

                # Weight by number of valid tokens (not -100)
                valid_tokens = (target_ids != -100).sum().item()
                if valid_tokens > 0:
                    total_loss += loss.item() * valid_tokens
                    total_tokens += valid_tokens

    # Compute perplexity
    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity calculation")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": num_samples,
    }


def evaluate_perplexity_simple(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Any,
    device: torch.device,
    max_samples: int = 50,
    max_length: int = 512,
) -> dict[str, float]:
    """Simple perplexity evaluation without sliding window.

    Faster but less accurate for long contexts. Uses standard cross-entropy loss
    on fixed-length sequences.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer for encoding text.
        dataset: The dataset to evaluate on.
        device: Device to run evaluation on.
        max_samples: Maximum number of samples to evaluate.
        max_length: Maximum sequence length.

    Returns:
        Dictionary with perplexity and other metrics.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    valid_samples = []

    # Collect valid samples
    for i, example in enumerate(dataset):
        if i >= max_samples * 2:  # Load extra to account for filtering
            break

        text = example.get("text", "")
        if len(text.strip()) < 50:  # Skip very short/empty samples
            continue

        # Tokenize
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if tokens["input_ids"].shape[1] > 10:  # Skip very short sequences
            valid_samples.append(tokens["input_ids"].squeeze(0))

        if len(valid_samples) >= max_samples:
            break

    valid_samples = valid_samples[:max_samples]
    num_samples = len(valid_samples)

    if num_samples == 0:
        raise ValueError("No valid samples found in dataset")

    print(f"Evaluating on {num_samples} samples...")

    with torch.no_grad():
        for sample in tqdm(valid_samples, desc="Evaluating"):
            # Add batch dimension
            input_ids = sample.unsqueeze(0).to(device)
            labels = input_ids.clone()

            # Set pad tokens to -100 to ignore in loss
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "num_samples": num_samples,
    }


def count_synaptic_layers(model: nn.Module) -> int:
    """Count the number of SynapticLayers in a model.

    Args:
        model: The model to analyze.

    Returns:
        Number of SynapticLayers found.
    """
    return sum(1 for m in model.modules() if isinstance(m, SynapticLayer))


def get_model_compression_stats(model: nn.Module) -> dict[str, Any] | None:
    """Get compression statistics from a model with SynapticLayers.

    Args:
        model: The model to analyze.

    Returns:
        Dictionary with compression stats, or None if no SynapticLayers.
    """
    from synaptic_pruning.layers import SynapticLayer

    stats = {
        "total_params": 0,
        "hot_params": 0,
        "warm_params": 0,
        "cold_params": 0,
        "hot_bytes": 0.0,
        "warm_bytes": 0.0,
        "cold_bytes": 0.0,
        "total_bytes": 0.0,
        "sparsity": 0.0,
        "effective_compression": 1.0,
    }

    has_synaptic = False

    for module in model.modules():
        if isinstance(module, SynapticLayer):
            has_synaptic = True
            layer_stats = module.get_compression_stats()
            total = layer_stats["total_params"]

            stats["total_params"] += total
            stats["hot_params"] += layer_stats["hot_count"]
            stats["warm_params"] += layer_stats["warm_count"]
            stats["cold_params"] += layer_stats["cold_count"]

    if not has_synaptic:
        return None

    # Calculate compression metrics
    if stats["total_params"] > 0:
        stats["hot_bytes"] = stats["hot_params"] * 2
        stats["warm_bytes"] = int(stats["warm_params"] * 0.5)
        stats["cold_bytes"] = int(stats["cold_params"] * 0.125)
        stats["total_bytes"] = stats["hot_bytes"] + stats["warm_bytes"] + stats["cold_bytes"]
        stats["sparsity"] = stats["cold_params"] / stats["total_params"]

        baseline_bytes = stats["total_params"] * 2
        if stats["total_bytes"] > 0:
            stats["effective_compression"] = baseline_bytes / stats["total_bytes"]

    return stats


def main():
    """Main entry point for perplexity evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity on WikiText-2"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (Synaptic model). If not provided, evaluates baseline GPT-2.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="gpt2",
        help="Baseline model name (default: gpt2)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset to use (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file (optional)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Acceptable threshold for perplexity increase (default: 0.5 = 50%%)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SYNAPTIC PRUNING - PERPLEXITY EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.dataset} ({args.dataset_config}) {args.split} split...")
    try:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
        print(f"Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying without config...")
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"Loaded {len(dataset)} examples")

    results = {}

    # Evaluate baseline model
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE MODEL")
    print("=" * 60)

    baseline_model, baseline_tokenizer = load_baseline_model(
        model_name=args.baseline_model,
        device=device,
    )

    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline model parameters: {baseline_params:,}")

    baseline_metrics = evaluate_perplexity_simple(
        baseline_model,
        baseline_tokenizer,
        dataset,
        device,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )

    baseline_ppl = baseline_metrics["perplexity"]
    print(f"\nBaseline Perplexity: {baseline_ppl:.2f}")
    print(f"Baseline Loss: {baseline_metrics['avg_loss']:.4f}")

    results["baseline"] = {
        "model_name": args.baseline_model,
        "parameters": baseline_params,
        "perplexity": baseline_ppl,
        "loss": baseline_metrics["avg_loss"],
        "num_samples": baseline_metrics["num_samples"],
    }

    # Evaluate synaptic model if checkpoint provided
    if args.checkpoint:
        print("\n" + "=" * 60)
        print("EVALUATING SYNAPTIC MODEL")
        print("=" * 60)

        synaptic_model, synaptic_tokenizer, checkpoint_info = load_model_from_checkpoint(
            args.checkpoint,
            device=device,
        )

        synaptic_params = sum(p.numel() for p in synaptic_model.parameters())
        print(f"Synaptic model parameters: {synaptic_params:,}")

        num_synaptic = count_synaptic_layers(synaptic_model)
        print(f"Number of SynapticLayers: {num_synaptic}")

        # Get compression stats
        compression_stats = get_model_compression_stats(synaptic_model)
        if compression_stats:
            print(f"\nCompression Statistics:")
            print(f"  Total params: {compression_stats['total_params']:,}")
            print(f"  Hot (FP16): {compression_stats['hot_params']:,} "
                  f"({compression_stats['hot_params']/compression_stats['total_params']:.1%})")
            print(f"  Warm (4-bit): {compression_stats['warm_params']:,} "
                  f"({compression_stats['warm_params']/compression_stats['total_params']:.1%})")
            print(f"  Cold (1-bit): {compression_stats['cold_params']:,} "
                  f"({compression_stats['cold_params']/compression_stats['total_params']:.1%})")
            print(f"  Sparsity: {compression_stats['sparsity']:.2%}")
            print(f"  Effective compression: {compression_stats['effective_compression']:.2f}x")

        synaptic_metrics = evaluate_perplexity_simple(
            synaptic_model,
            synaptic_tokenizer,
            dataset,
            device,
            max_samples=args.max_samples,
            max_length=args.max_length,
        )

        synaptic_ppl = synaptic_metrics["perplexity"]
        print(f"\nSynaptic Model Perplexity: {synaptic_ppl:.2f}")
        print(f"Synaptic Model Loss: {synaptic_metrics['avg_loss']:.4f}")

        # Compare perplexities
        ppl_increase = (synaptic_ppl - baseline_ppl) / baseline_ppl
        ppl_ratio = synaptic_ppl / baseline_ppl

        print(f"\nPerplexity Comparison:")
        print(f"  Baseline: {baseline_ppl:.2f}")
        print(f"  Synaptic: {synaptic_ppl:.2f}")
        print(f"  Increase: {ppl_increase:.2%}")
        print(f"  Ratio: {ppl_ratio:.2f}x")

        results["synaptic"] = {
            "checkpoint": args.checkpoint,
            "parameters": synaptic_params,
            "num_synaptic_layers": num_synaptic,
            "perplexity": synaptic_ppl,
            "loss": synaptic_metrics["avg_loss"],
            "num_samples": synaptic_metrics["num_samples"],
            "compression_stats": compression_stats,
            "checkpoint_info": checkpoint_info,
        }

        results["comparison"] = {
            "baseline_perplexity": baseline_ppl,
            "synaptic_perplexity": synaptic_ppl,
            "perplexity_increase_ratio": ppl_ratio,
            "perplexity_increase_percent": ppl_increase * 100,
            "threshold": args.threshold,
            "threshold_percent": args.threshold * 100,
        }

        # Validation check
        print("\n" + "=" * 60)
        print("VALIDATION CHECKS")
        print("=" * 60)

        # VAL-BEN-003: Perplexity within threshold of baseline
        if ppl_increase <= args.threshold:
            print(f"✓ VAL-BEN-003 PASSED: Perplexity increase {ppl_increase:.2%} "
                  f"<= {args.threshold:.0%} threshold")
            print(f"  Baseline: {baseline_ppl:.2f}, Synaptic: {synaptic_ppl:.2f}")
            results["validation"] = {
                "val_ben_003": "passed",
                "message": f"Perplexity within {args.threshold:.0%} of baseline",
            }
        else:
            print(f"✗ VAL-BEN-003 FAILED: Perplexity increase {ppl_increase:.2%} "
                  f"> {args.threshold:.0%} threshold")
            print(f"  Baseline: {baseline_ppl:.2f}, Synaptic: {synaptic_ppl:.2f}")
            results["validation"] = {
                "val_ben_003": "failed",
                "message": f"Perplexity exceeds {args.threshold:.0%} threshold",
            }
    else:
        print("\nNote: No checkpoint provided. Only baseline evaluation performed.")
        results["validation"] = {
            "val_ben_003": "skipped",
            "message": "No synaptic checkpoint to compare",
        }

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
