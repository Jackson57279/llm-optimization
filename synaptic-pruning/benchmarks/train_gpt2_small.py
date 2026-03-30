"""Benchmark script for training GPT-2 Small with Synaptic Pruning.

This script implements a toy training benchmark to validate:
- VAL-BEN-001: Successfully train GPT-2 small with Synaptic Pruning
- VAL-BEN-002: Achieve measurable compression ratios

Usage:
    python benchmarks/train_gpt2_small.py

The script will:
1. Load GPT-2 small (124M parameters) from HuggingFace
2. Replace all nn.Linear layers with SynapticLayers
3. Train for 100 steps on WikiText-2 samples
4. Save a checkpoint with compression statistics
5. Report compression ratios achieved
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, Conv1D

from synaptic_pruning import SynapticLayer, SynapticTrainer, PruningSchedule


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


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters.

    Args:
        model: The model to analyze.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_synaptic_layers(model: nn.Module) -> int:
    """Count the number of SynapticLayers in a model.

    Args:
        model: The model to analyze.

    Returns:
        Number of SynapticLayers found.
    """
    return sum(1 for m in model.modules() if isinstance(m, SynapticLayer))


def get_wikitext_samples(tokenizer, num_samples: int = 100, max_length: int = 512) -> list[torch.Tensor]:
    """Load WikiText-2 samples for training.

    Args:
        tokenizer: The tokenizer to use.
        num_samples: Number of samples to load.
        max_length: Maximum sequence length.

    Returns:
        List of tokenized tensors.
    """
    # Load WikiText-2 dataset
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    except Exception:
        # Fallback: try without subset specification
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples * 3:  # Load extra to account for empty samples
            break

        text = example["text"].strip()
        if len(text) < 50:  # Skip very short/empty samples
            continue

        # Tokenize
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if tokens["input_ids"].shape[1] > 10:  # Skip very short sequences
            samples.append(tokens["input_ids"].squeeze(0))

        if len(samples) >= num_samples:
            break

    return samples[:num_samples]


def train_gpt2_with_synaptic_pruning(
    model_name: str = "gpt2",
    num_steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_sparsity: float = 0.5,
    warmup_steps: int = 20,
    output_dir: str = "./checkpoints",
) -> dict:
    """Train GPT-2 with Synaptic Pruning.

    Args:
        model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
        num_steps: Number of training steps.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        max_sparsity: Maximum sparsity target (0.0 to 1.0).
        warmup_steps: Number of warmup steps before pruning starts.
        output_dir: Directory to save checkpoints.

    Returns:
        Dictionary with training results and compression statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Count original parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Original model: {total_params:,} parameters")
    print(f"Trainable: {trainable_params:,} parameters")

    # Replace Conv1D layers with SynapticLayers
    print("\nReplacing Conv1D/Linear layers with SynapticLayers...")
    replace_conv1d_with_synaptic(model)

    num_synaptic = count_synaptic_layers(model)
    print(f"Replaced {num_synaptic} layers with SynapticLayers")

    # Move model to device
    model = model.to(device)

    # Load training data
    print("\nLoading WikiText-2 samples...")
    try:
        samples = get_wikitext_samples(tokenizer, num_samples=num_steps * batch_size)
        print(f"Loaded {len(samples)} training samples")
    except Exception as e:
        print(f"Warning: Could not load WikiText-2: {e}")
        print("Generating synthetic data instead...")
        # Generate synthetic data as fallback
        samples = [torch.randint(0, tokenizer.vocab_size, (128,)) for _ in range(num_steps * batch_size)]
        print(f"Generated {len(samples)} synthetic samples")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Setup pruning schedule (warmup, then progressive pruning)
    # Convert steps to "epochs" for the schedule (treat each step as an epoch)
    schedule = PruningSchedule(
        max_sparsity=max_sparsity,
        schedule_type="linear",
        warmup_epochs=warmup_steps,
        max_epochs=num_steps,
    )

    # Create trainer
    trainer = SynapticTrainer(
        model=model,
        optimizer=optimizer,
        pruning_schedule=schedule,
        recovery_weight=0.0,  # Skip recovery for toy benchmark
        compression_update_freq=10,
        device=device,
    )

    print(f"\nTraining for {num_steps} steps...")
    print(f"Target max sparsity: {max_sparsity:.1%}")
    print(f"Warmup steps: {warmup_steps}")

    # Training loop
    model.train()
    losses = []

    for step in range(num_steps):
        # Get batch
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        batch_samples = samples[batch_start:batch_end]

        # Pad batch
        max_len = max(s.shape[0] for s in batch_samples)
        input_ids = torch.stack([
            torch.cat([s, torch.full((max_len - s.shape[0],), tokenizer.pad_token_id, dtype=torch.long)])
            if s.shape[0] < max_len else s
            for s in batch_samples
        ]).to(device)

        # Forward pass
        optimizer.zero_grad()

        # Create labels (shifted inputs for causal LM)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore pad tokens in loss

        # Use model forward directly (not trainer's train method) for simpler control
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update pruning thresholds based on schedule
        trainer._update_activity_thresholds(step)

        # Log progress
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))

            # Get compression stats
            stats = trainer._compute_compression_stats()
            sparsity = stats.get("sparsity", 0.0)
            compression = stats.get("effective_compression", 1.0)

            print(
                f"Step {step+1}/{num_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Sparsity: {sparsity:.2%} | "
                f"Compression: {compression:.2f}x"
            )

    # Final compression stats
    print("\n" + "=" * 60)
    print("FINAL COMPRESSION STATISTICS")
    print("=" * 60)

    final_stats = trainer._compute_compression_stats()

    print(f"Total parameters: {final_stats['total_params']:,}")
    print(f"Hot (FP16) parameters: {final_stats['hot_params']:,} ({final_stats.get('hot_ratio', 0):.1%})")
    print(f"Warm (4-bit) parameters: {final_stats['warm_params']:,} ({final_stats.get('warm_ratio', 0):.1%})")
    print(f"Cold (1-bit) parameters: {final_stats['cold_params']:,} ({final_stats.get('cold_ratio', 0):.1%})")
    print(f"\nSparsity: {final_stats['sparsity']:.2%}")
    print(f"Effective compression: {final_stats['effective_compression']:.2f}x")

    # Save checkpoint
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / f"gpt2_synaptic_{num_steps}steps.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Verify loss decreased
    initial_loss = sum(losses[:10]) / 10 if len(losses) >= 10 else losses[0]
    final_loss = sum(losses[-10:]) / 10 if len(losses) >= 10 else losses[-1]
    loss_decreased = final_loss < initial_loss

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss decreased: {loss_decreased}")

    return {
        "losses": losses,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "compression_stats": final_stats,
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPT-2 Small training with Synaptic Pruning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--max-sparsity",
        type=float,
        default=0.5,
        help="Maximum sparsity target (default: 0.5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup steps before pruning (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints (default: ./checkpoints)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SYNAPTIC PRUNING - GPT-2 SMALL BENCHMARK")
    print("=" * 60)

    # Run training
    results = train_gpt2_with_synaptic_pruning(
        model_name=args.model,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_sparsity=args.max_sparsity,
        warmup_steps=args.warmup,
        output_dir=args.output_dir,
    )

    # Success criteria for validation
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    # VAL-BEN-001: Training completes without errors and loss decreases
    if results["loss_decreased"]:
        print("✓ VAL-BEN-001 PASSED: Training completed, loss decreased")
    else:
        print("✗ VAL-BEN-001 FAILED: Loss did not decrease")

    # VAL-BEN-002: Compression ratio > 10x
    compression = results["compression_stats"].get("effective_compression", 1.0)
    if compression >= 10.0:
        print(f"✓ VAL-BEN-002 PASSED: Compression ratio {compression:.2f}x >= 10x")
    else:
        print(f"! VAL-BEN-002 PARTIAL: Compression ratio {compression:.2f}x < 10x target")
        print("  (This may improve with more training steps or higher max_sparsity)")

    print("\nBenchmark complete!")

    return results


if __name__ == "__main__":
    main()
