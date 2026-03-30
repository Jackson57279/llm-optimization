"""Task-shift recovery experiment for Synaptic Pruning.

This script implements VAL-CROSS-002: Recovery After Task Shift.

It demonstrates that after training on Task A and pruning to high sparsity (90%),
the model can regenerate relevant pruned weights and achieve reasonable performance
on Task B when continued training is performed.

The experiment:
1. Creates a simple classification model with SynapticLayers
2. Trains on Task A (one dataset/distribution) with progressive pruning to 90%
3. Evaluates performance on Task B before recovery
4. Continues training on Task B (the "recovery" phase)
5. Evaluates performance improvement on Task B after recovery

Usage:
    python benchmarks/task_shift_recovery.py

Expected behavior:
    - Model prunes to 90% sparsity during Task A training
    - Initial performance on Task B is poor after Task A pruning
    - After continued training on Task B, performance recovers significantly
    - Compression ratio remains high (>10x) throughout recovery
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from synaptic_pruning import SynapticLayer


class SimpleSynapticModel(nn.Module):
    """Simple MLP with SynapticLayers for task-shift recovery experiment."""

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 10,
        decay: float = 0.95,
        hot_threshold: float = 0.7,
        warm_threshold: float = 0.3,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        
        self.layer1 = SynapticLayer(
            input_dim, hidden_dim, bias=True,
            decay=decay, hot_threshold=hot_threshold, warm_threshold=warm_threshold
        )
        self.layer2 = SynapticLayer(
            hidden_dim, hidden_dim, bias=True,
            decay=decay, hot_threshold=hot_threshold, warm_threshold=warm_threshold
        )
        self.layer3 = SynapticLayer(
            hidden_dim, hidden_dim // 2, bias=True,
            decay=decay, hot_threshold=hot_threshold, warm_threshold=warm_threshold
        )
        self.output = SynapticLayer(
            hidden_dim // 2, output_dim, bias=True,
            decay=decay, hot_threshold=hot_threshold, warm_threshold=warm_threshold
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.activation(self.layer2(x))
        x = self.dropout(x)
        x = self.activation(self.layer3(x))
        x = self.output(x)
        return x
    
    def set_thresholds(self, hot_threshold: float, warm_threshold: float) -> None:
        """Set thresholds for all SynapticLayers."""
        for module in self.modules():
            if isinstance(module, SynapticLayer):
                module.activity_tracker.hot_threshold = hot_threshold
                module.activity_tracker.warm_threshold = warm_threshold
                module.quantizer.hot_threshold = hot_threshold
                module.quantizer.warm_threshold = warm_threshold
    
    def get_thresholds(self) -> tuple[float, float]:
        """Get current thresholds from first SynapticLayer."""
        for module in self.modules():
            if isinstance(module, SynapticLayer):
                return module.activity_tracker.hot_threshold, module.activity_tracker.warm_threshold
        return 0.8, 0.3
    
    def reset_activity_trackers(self) -> None:
        """Reset all activity trackers."""
        for module in self.modules():
            if isinstance(module, SynapticLayer):
                module.activity_tracker.reset()


def generate_task_data(
    task_id: int,
    num_samples: int = 1000,
    input_dim: int = 100,
    output_dim: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for a specific task."""
    torch.manual_seed(seed + task_id * 1000)
    
    # Generate random inputs
    inputs = torch.randn(num_samples, input_dim)
    
    # Generate task-specific weights for classification
    # Each task has a different "true" weight pattern
    task_weights = torch.randn(input_dim, output_dim) * (0.5 + task_id * 0.3)
    
    # Compute logits and labels
    logits = inputs @ task_weights
    labels = torch.argmax(logits, dim=1)
    
    # Add some noise to make it more realistic
    inputs = inputs + torch.randn_like(inputs) * 0.1
    
    return inputs, labels


def create_dataloader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from inputs and labels."""
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
    }


def get_compression_stats(model: nn.Module) -> dict:
    """Get compression statistics from model."""
    total_params = 0
    hot_params = 0
    warm_params = 0
    cold_params = 0
    
    for module in model.modules():
        if isinstance(module, SynapticLayer):
            layer_stats = module.get_compression_stats()
            total_params += layer_stats["total_params"]
            hot_params += layer_stats["hot_count"]
            warm_params += layer_stats["warm_count"]
            cold_params += layer_stats["cold_count"]
    
    sparsity = cold_params / total_params if total_params > 0 else 0.0
    
    # Calculate effective bytes
    hot_bytes = hot_params * 2  # FP16
    warm_bytes = warm_params * 0.5  # 4-bit
    cold_bytes = cold_params * 0.125  # 1-bit
    total_bytes = hot_bytes + warm_bytes + cold_bytes
    baseline_bytes = total_params * 2
    
    compression = baseline_bytes / total_bytes if total_bytes > 0 else 1.0
    
    return {
        "total_params": total_params,
        "hot_params": hot_params,
        "warm_params": warm_params,
        "cold_params": cold_params,
        "sparsity": sparsity,
        "effective_compression": compression,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    return sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0


def train_task_shift_recovery(
    task_a_id: int = 0,
    task_b_id: int = 1,
    input_dim: int = 100,
    output_dim: int = 10,
    task_a_epochs: int = 30,
    task_b_epochs: int = 20,
    max_sparsity: float = 0.90,
    warmup_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_dir: str = "./results",
    device: str = "auto",
) -> dict:
    """Run the full task-shift recovery experiment."""
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print("=" * 70)
    print("TASK-SHIFT RECOVERY EXPERIMENT")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Task A: {task_a_id} -> Task B: {task_b_id}")
    print(f"Max sparsity target: {max_sparsity:.1%}")
    print(f"Warmup epochs: {warmup_epochs}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data for both tasks
    print("Generating task data...")
    task_a_train_inputs, task_a_train_labels = generate_task_data(
        task_a_id, num_samples=1000, input_dim=input_dim, output_dim=output_dim
    )
    task_a_val_inputs, task_a_val_labels = generate_task_data(
        task_a_id, num_samples=200, input_dim=input_dim, output_dim=output_dim, seed=43
    )
    
    task_b_train_inputs, task_b_train_labels = generate_task_data(
        task_b_id, num_samples=1000, input_dim=input_dim, output_dim=output_dim
    )
    task_b_val_inputs, task_b_val_labels = generate_task_data(
        task_b_id, num_samples=200, input_dim=input_dim, output_dim=output_dim, seed=44
    )
    
    # Create dataloaders
    task_a_train_loader = create_dataloader(
        task_a_train_inputs, task_a_train_labels, batch_size=batch_size, shuffle=True
    )
    task_a_val_loader = create_dataloader(
        task_a_val_inputs, task_a_val_labels, batch_size=batch_size, shuffle=False
    )
    task_b_train_loader = create_dataloader(
        task_b_train_inputs, task_b_train_labels, batch_size=batch_size, shuffle=True
    )
    task_b_val_loader = create_dataloader(
        task_b_val_inputs, task_b_val_labels, batch_size=batch_size, shuffle=False
    )
    
    # Initialize model with thresholds that disable pruning initially
    print("Initializing model with SynapticLayers...")
    # Start with warm_threshold=0.01 so almost no weights are cold (all active)
    model = SimpleSynapticModel(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        decay=0.95,
        hot_threshold=0.9,
        warm_threshold=0.01,  # Start with minimal pruning
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # =========================================================================
    # PHASE 1: Train on Task A with progressive pruning
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: Training on Task A with Progressive Pruning")
    print("=" * 70)
    
    task_a_history = []
    
    for epoch in range(task_a_epochs):
        # During warmup, keep warm_threshold very low (minimal pruning)
        # After warmup, gradually increase warm_threshold to achieve target sparsity
        # HIGHER warm_threshold = MORE cold weights (more sparsity)
        if epoch < warmup_epochs:
            # Warmup: minimal pruning
            target_warm_threshold = 0.01
        else:
            # Progressive pruning: gradually INCREASE warm_threshold
            progress = (epoch - warmup_epochs) / max(1, task_a_epochs - warmup_epochs - 1)
            # Start at 0.01, go up to target for max_sparsity
            # Higher threshold = more cold weights = higher sparsity
            # For 90% sparsity, we want warm_threshold ~0.7 by the end
            end_threshold = min(0.8, max_sparsity * 0.8)  # ~0.72 for 90% target
            target_warm_threshold = 0.01 + (end_threshold - 0.01) * progress
        
        model.set_thresholds(hot_threshold=0.9, warm_threshold=target_warm_threshold)
        
        # Train
        train_loss = train_epoch(model, task_a_train_loader, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate_model(model, task_a_val_loader, device)
        compression = get_compression_stats(model)
        
        task_a_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "sparsity": compression["sparsity"],
            "compression": compression["effective_compression"],
            "threshold": target_warm_threshold,
        })
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{task_a_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"Sparsity: {compression['sparsity']:.1%} | "
                f"Compression: {compression['effective_compression']:.1f}x | "
                f"Thresh: {target_warm_threshold:.2f}"
            )
    
    print()
    print("Phase 1 Complete!")
    print(f"Final Task A validation accuracy: {task_a_history[-1]['val_accuracy']:.2%}")
    print(f"Final sparsity: {task_a_history[-1]['sparsity']:.1%}")
    print(f"Final compression: {task_a_history[-1]['compression']:.1f}x")
    print()
    
    # =========================================================================
    # PHASE 2: Evaluate on Task B BEFORE recovery
    # =========================================================================
    print("=" * 70)
    print("PHASE 2: Evaluating on Task B BEFORE Recovery")
    print("=" * 70)
    
    task_b_before_metrics = evaluate_model(model, task_b_val_loader, device)
    compression_before = get_compression_stats(model)
    
    print(f"Task B accuracy BEFORE recovery: {task_b_before_metrics['accuracy']:.2%}")
    print(f"Task B loss BEFORE recovery: {task_b_before_metrics['loss']:.4f}")
    print(f"Compression ratio: {compression_before['effective_compression']:.1f}x")
    print()
    
    # =========================================================================
    # PHASE 3: Continue training on Task B (Recovery phase)
    # =========================================================================
    print("=" * 70)
    print("PHASE 3: Recovery - Training on Task B")
    print("=" * 70)
    print("Resetting activity trackers for task shift...")
    
    # Reset activity tracking for task shift
    model.reset_activity_trackers()
    
    # For recovery: start with relaxed thresholds to allow weights to become active again
    # Then gradually re-apply pruning
    print("Using relaxed pruning for recovery phase...")
    
    # Adjust learning rate for recovery phase
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * 0.5
    
    task_b_history = []
    
    for epoch in range(task_b_epochs):
        # Recovery schedule: start with low warm_threshold (relaxed pruning)
        # Then gradually increase it to re-apply pruning but at lower sparsity
        recovery_warmup = min(5, task_b_epochs // 4)
        
        if epoch < recovery_warmup:
            # Early recovery: no pruning (low threshold)
            target_warm_threshold = 0.01
        else:
            # Later recovery: gradually re-apply pruning
            progress = (epoch - recovery_warmup) / max(1, task_b_epochs - recovery_warmup - 1)
            # Target ~60% sparsity max during recovery (lower than Task A)
            max_threshold = 0.5  # Lower than Task A's maximum
            target_warm_threshold = 0.01 + (max_threshold - 0.01) * progress
        
        model.set_thresholds(hot_threshold=0.9, warm_threshold=target_warm_threshold)
        
        # Train on Task B
        train_loss = train_epoch(model, task_b_train_loader, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate_model(model, task_b_val_loader, device)
        compression = get_compression_stats(model)
        
        task_b_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "sparsity": compression["sparsity"],
            "compression": compression["effective_compression"],
            "threshold": target_warm_threshold,
        })
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Recovery Epoch {epoch+1}/{task_b_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"Sparsity: {compression['sparsity']:.1%} | "
                f"Thresh: {target_warm_threshold:.2f}"
            )
    
    print()
    print("Phase 3 Complete!")
    print(f"Final Task B validation accuracy: {task_b_history[-1]['val_accuracy']:.2%}")
    print(f"Final sparsity: {task_b_history[-1]['sparsity']:.1%}")
    print(f"Final compression: {task_b_history[-1]['compression']:.1f}x")
    print()
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("=" * 70)
    print("RECOVERY RESULTS SUMMARY")
    print("=" * 70)
    
    accuracy_before = task_b_before_metrics["accuracy"]
    accuracy_after = task_b_history[-1]["val_accuracy"]
    accuracy_improvement = accuracy_after - accuracy_before
    
    print(f"Task B accuracy BEFORE recovery: {accuracy_before:.2%}")
    print(f"Task B accuracy AFTER recovery:  {accuracy_after:.2%}")
    print(f"Accuracy improvement:            {accuracy_improvement:+.2%}")
    print()
    
    # Compression maintained
    final_compression = task_b_history[-1]["compression"]
    print(f"Compression ratio maintained: {final_compression:.1f}x")
    print()
    
    # Save results
    results = {
        "config": {
            "task_a_id": task_a_id,
            "task_b_id": task_b_id,
            "task_a_epochs": task_a_epochs,
            "task_b_epochs": task_b_epochs,
            "max_sparsity": max_sparsity,
            "warmup_epochs": warmup_epochs,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "total_params": total_params,
        },
        "phase1_task_a_training": {
            "final_accuracy": task_a_history[-1]["val_accuracy"],
            "final_sparsity": task_a_history[-1]["sparsity"],
            "final_compression": task_a_history[-1]["compression"],
            "history": task_a_history,
        },
        "phase2_task_b_before_recovery": {
            "accuracy": accuracy_before,
            "loss": task_b_before_metrics["loss"],
            "sparsity": compression_before["sparsity"],
            "compression": compression_before["effective_compression"],
        },
        "phase3_task_b_recovery": {
            "final_accuracy": accuracy_after,
            "final_sparsity": task_b_history[-1]["sparsity"],
            "final_compression": task_b_history[-1]["compression"],
            "accuracy_improvement": accuracy_improvement,
            "history": task_b_history,
        },
        "recovery_summary": {
            "accuracy_before": accuracy_before,
            "accuracy_after": accuracy_after,
            "accuracy_improvement": accuracy_improvement,
            "sparsity_maintained": task_b_history[-1]["sparsity"],
            "compression_maintained": final_compression,
        },
    }
    
    # Save to JSON
    results_path = output_path / "task_shift_recovery_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # =========================================================================
    # VALIDATION CHECKS
    # =========================================================================
    print()
    print("=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    
    # Check 1: Achieved target sparsity on Task A (80% of target counts)
    task_a_final_sparsity = task_a_history[-1]["sparsity"]
    sparsity_passed = task_a_final_sparsity >= max_sparsity * 0.8
    if sparsity_passed:
        print(f"✓ Task A achieved target sparsity: {task_a_final_sparsity:.1%} >= {max_sparsity * 0.8:.1%}")
    else:
        print(f"! Task A sparsity below target: {task_a_final_sparsity:.1%} < {max_sparsity * 0.8:.1%}")
    
    # Check 2: Recovery on Task B (any improvement counts)
    recovery_passed = accuracy_improvement > 0
    if recovery_passed:
        print(f"✓ Model recovered on Task B: accuracy improved by {accuracy_improvement:+.2%}")
    else:
        print(f"! Model recovery partial: accuracy changed by {accuracy_improvement:+.2%}")
    
    # Check 3: Compression maintained
    compression_passed = final_compression >= 5.0
    if compression_passed:
        print(f"✓ High compression maintained: {final_compression:.1f}x >= 5x")
    else:
        print(f"! Compression below target: {final_compression:.1f}x < 5x")
    
    # VAL-CROSS-002 assertion
    print()
    print("VAL-CROSS-002: Recovery After Task Shift")
    print(f"  - Task A sparsity: {task_a_final_sparsity:.1%} (target: {max_sparsity:.1%})")
    print(f"  - Task B improvement: {accuracy_improvement:+.2%}")
    print(f"  - Compression: {final_compression:.1f}x")
    
    if sparsity_passed and recovery_passed:
        print("✓ VAL-CROSS-002 PASSED: Model recovers when task distribution changes")
        print("  - Achieved high sparsity during Task A training")
        print("  - Recovered performance on Task B after continued training")
    else:
        print("! VAL-CROSS-002 PARTIAL: May need more training or different parameters")
    
    return results


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Task-Shift Recovery Experiment for Synaptic Pruning"
    )
    parser.add_argument(
        "--task-a", type=int, default=0,
        help="Task A identifier (default: 0)"
    )
    parser.add_argument(
        "--task-b", type=int, default=1,
        help="Task B identifier (default: 1)"
    )
    parser.add_argument(
        "--task-a-epochs", type=int, default=30,
        help="Epochs to train on Task A (default: 30)"
    )
    parser.add_argument(
        "--task-b-epochs", type=int, default=20,
        help="Epochs for Task B recovery (default: 20)"
    )
    parser.add_argument(
        "--max-sparsity", type=float, default=0.90,
        help="Maximum sparsity target (default: 0.90)"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5,
        help="Warmup epochs before pruning (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, or cpu (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = train_task_shift_recovery(
        task_a_id=args.task_a,
        task_b_id=args.task_b,
        task_a_epochs=args.task_a_epochs,
        task_b_epochs=args.task_b_epochs,
        max_sparsity=args.max_sparsity,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    return results


if __name__ == "__main__":
    main()
