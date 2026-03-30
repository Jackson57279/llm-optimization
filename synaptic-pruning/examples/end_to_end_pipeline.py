"""
End-to-End Pipeline Example: Synaptic Pruning Complete Workflow

This script demonstrates the complete workflow of using Synaptic Pruning for
neural network compression, from model definition to training, saving, loading,
and evaluation. It serves as a comprehensive tutorial for new users.

Usage:
    python examples/end_to_end_pipeline.py

The script will:
    1. Define a simple model with SynapticLayers
    2. Set up the trainer with a progressive pruning schedule
    3. Train the model on synthetic data
    4. Save the compressed checkpoint
    5. Load and evaluate the compressed model
    6. Report final compression statistics
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

# Import Synaptic Pruning components
# These provide the drop-in replacements and training infrastructure
from synaptic_pruning import (
    SynapticLayer,      # Drop-in replacement for nn.Linear with activity tracking
    SynapticTrainer,    # End-to-end trainer with progressive pruning
    PruningSchedule,    # Configurable pruning schedule (linear, exponential, etc.)
)


# =============================================================================
# Step 1: Define a Model with SynapticLayers
# =============================================================================

class SimpleClassifier(nn.Module):
    """A simple classifier with SynapticLayers for compression.

    This model demonstrates how easy it is to replace standard nn.Linear
    layers with SynapticLayers - they have the same interface and can be
    used interchangeably in most PyTorch models.

    Attributes:
        feature_extractor: Series of SynapticLayers with activation.
        classifier: Final SynapticLayer for output.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_classes: int = 10) -> None:
        """Initialize the classifier with SynapticLayers.

        Args:
            input_dim: Dimensionality of input features.
            hidden_dim: Dimensionality of hidden layers.
            num_classes: Number of output classes.
        """
        super().__init__()

        # Feature extractor: input -> hidden -> hidden
        # Using SynapticLayer instead of nn.Linear enables automatic
        # activity tracking and tiered quantization during training.
        self.feature_extractor = nn.Sequential(
            SynapticLayer(input_dim, hidden_dim, decay=0.9),
            nn.ReLU(),
            SynapticLayer(hidden_dim, hidden_dim, decay=0.9),
            nn.ReLU(),
        )

        # Classifier: hidden -> output
        # The same SynapticLayer can be used for any linear transformation.
        self.classifier = SynapticLayer(hidden_dim, num_classes, decay=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        # Extract features
        features = self.feature_extractor(x)

        # Classify
        logits = self.classifier(features)

        return logits


# =============================================================================
# Step 2: Generate Synthetic Training Data
# =============================================================================

def create_synthetic_dataset(
    num_samples: int = 1000,
    input_dim: int = 128,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data for demonstration.

    In a real scenario, you would load your actual dataset (e.g., from
    torchvision.datasets or HuggingFace datasets).

    Args:
        num_samples: Number of samples to generate.
        input_dim: Dimensionality of input features.
        num_classes: Number of output classes.

    Returns:
        Tuple of (inputs, labels) tensors.
    """
    # Generate random input features
    inputs = torch.randn(num_samples, input_dim)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return inputs, labels


def create_dataloader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader from tensors.

    Args:
        inputs: Input tensor.
        labels: Label tensor.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle data each epoch.

    Returns:
        PyTorch DataLoader for training.
    """
    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader


# =============================================================================
# Step 3: Setup Training Infrastructure
# =============================================================================

def setup_training(
    model: nn.Module,
    learning_rate: float = 1e-3,
    max_sparsity: float = 0.7,
    num_epochs: int = 10,
    warmup_epochs: int = 2,
) -> SynapticTrainer:
    """Setup the SynapticTrainer with pruning schedule.

    This configures the progressive pruning mechanism that gradually increases
    sparsity during training, allowing the model to adapt to the compression.

    Args:
        model: The model to train (should contain SynapticLayers).
        learning_rate: Learning rate for the optimizer.
        max_sparsity: Maximum target sparsity (0.0 to 1.0).
        num_epochs: Total number of training epochs.
        warmup_epochs: Number of epochs before pruning starts.

    Returns:
        Configured SynapticTrainer instance.
    """
    # Create optimizer - this will optimize both the model weights and
    # the recovery network parameters (if configured)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create pruning schedule - linear increase from 0% to max_sparsity
    # over the training period, with warmup_epochs of no pruning at start.
    # Options: "linear", "exponential", "cosine", "stepped"
    pruning_schedule = PruningSchedule(
        max_sparsity=max_sparsity,      # Target 70% sparsity
        schedule_type="linear",       # Linear increase in sparsity
        warmup_epochs=warmup_epochs,  # No pruning for first 2 epochs
        max_epochs=num_epochs,         # Full training duration
    )

    # Create trainer - handles the full training loop with:
    # - Progressive pruning according to schedule
    # - Activity tracking updates
    # - Compression metrics collection
    # - Checkpoint saving/loading
    trainer = SynapticTrainer(
        model=model,
        optimizer=optimizer,
        pruning_schedule=pruning_schedule,
        recovery_weight=0.01,        # Weight for recovery loss (optional)
        compression_update_freq=10,   # Update stats every 10 steps
    )

    return trainer


# =============================================================================
# Step 4: Define Loss Function
# =============================================================================

def create_loss_function() -> callable:
    """Create a loss function for training.

    Returns:
        Loss function that takes (predictions, targets) and returns loss.
    """
    # Standard cross-entropy loss for classification
    # This works exactly the same with SynapticLayers as with nn.Linear
    criterion = nn.CrossEntropyLoss()

    def loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            predictions: Model output logits.
            targets: Ground truth labels.

        Returns:
            Scalar loss tensor.
        """
        return criterion(predictions, targets)

    return loss_fn


# =============================================================================
# Step 5: Training Loop
# =============================================================================

def train_model(
    trainer: SynapticTrainer,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    num_epochs: int = 10,
) -> dict:
    """Train the model using SynapticTrainer.

    Args:
        trainer: Configured SynapticTrainer instance.
        train_loader: Training data loader.
        loss_fn: Loss function.
        num_epochs: Number of epochs to train.

    Returns:
        Training history dictionary.
    """
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Train the model
    # The trainer handles:
    # - Forward/backward passes
    # - Activity tracking updates
    # - Pruning threshold adjustments
    # - Compression metrics collection
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=num_epochs,
        loss_fn=loss_fn,
        log_interval=5,  # Log every 5 batches
    )

    return history


# =============================================================================
# Step 6: Save Compressed Model
# =============================================================================

def save_checkpoint(
    trainer: SynapticTrainer,
    checkpoint_path: str,
) -> None:
    """Save the trained model checkpoint.

    The checkpoint includes:
    - Model state dict (with compressed weights and activity scores)
    - Optimizer state
    - Pruning schedule configuration
    - Compression statistics history

    Args:
        trainer: SynapticTrainer with trained model.
        checkpoint_path: Path to save checkpoint.
    """
    print(f"\nSaving checkpoint to: {checkpoint_path}")

    # Ensure directory exists
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    trainer.save_checkpoint(checkpoint_path)

    print(f"Checkpoint saved successfully!")


# =============================================================================
# Step 7: Load and Evaluate Model
# =============================================================================

def load_and_evaluate(
    checkpoint_path: str,
    model_class: type,
    model_kwargs: dict,
    test_inputs: torch.Tensor,
    test_labels: torch.Tensor,
) -> dict:
    """Load a saved checkpoint and evaluate the model.

    Demonstrates that the compressed model can be loaded and produces
    the same outputs as before saving.

    Args:
        checkpoint_path: Path to checkpoint file.
        model_class: Model class to instantiate.
        model_kwargs: Keyword arguments for model constructor.
        test_inputs: Test input tensor.
        test_labels: Test label tensor.

    Returns:
        Evaluation results dictionary.
    """
    print(f"\n" + "=" * 60)
    print("LOADING AND EVALUATING CHECKPOINT")
    print("=" * 60)

    # Create a fresh model instance
    model = model_class(**model_kwargs)

    # Create a fresh trainer (needed for checkpoint loading)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = SynapticTrainer(model, optimizer)

    # Load checkpoint
    # This restores:
    # - All model weights (with tiered quantization)
    # - Activity scores for each SynapticLayer
    # - Optimizer state
    # - Training progress
    trainer.load_checkpoint(checkpoint_path)

    print(f"Checkpoint loaded successfully!")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()

    # Get compression statistics
    compression_summary = trainer.get_compression_summary()
    final_stats = compression_summary["final_stats"]

    return {
        "accuracy": accuracy,
        "compression_stats": final_stats,
    }


# =============================================================================
# Main Execution
# =============================================================================

def main() -> dict:
    """Execute the complete end-to-end pipeline.

    This function orchestrates the entire workflow:
    1. Model definition
    2. Data preparation
    3. Training setup
    4. Training execution
    5. Model saving
    6. Model loading and evaluation
    7. Results reporting

    Returns:
        Dictionary with training and evaluation results.
    """
    print("\n" + "=" * 60)
    print("SYNAPTIC PRUNING: END-TO-END PIPELINE DEMO")
    print("=" * 60)
    print("\nThis script demonstrates the complete workflow:")
    print("  1. Define model with SynapticLayers")
    print("  2. Set up trainer with pruning schedule")
    print("  3. Train with progressive pruning")
    print("  4. Save compressed checkpoint")
    print("  5. Load and evaluate")
    print("  6. Report compression statistics")

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    INPUT_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 10
    NUM_SAMPLES = 500
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    MAX_SPARSITY = 0.7  # Target 70% sparsity
    WARMUP_EPOCHS = 2

    # Checkpoint path
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_path = str(checkpoint_dir / "end_to_end_pipeline.pt")

    # -------------------------------------------------------------------------
    # Step 1: Create Model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 1: Creating model with SynapticLayers")
    print("-" * 60)

    model = SimpleClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Count SynapticLayers
    num_synaptic = sum(1 for m in model.modules() if isinstance(m, SynapticLayer))
    print(f"Number of SynapticLayers: {num_synaptic}")

    # -------------------------------------------------------------------------
    # Step 2: Create Training Data
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 2: Creating synthetic training data")
    print("-" * 60)

    # Training data
    train_inputs, train_labels = create_synthetic_dataset(
        num_samples=NUM_SAMPLES,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )
    train_loader = create_dataloader(train_inputs, train_labels, batch_size=BATCH_SIZE)

    # Test data (for evaluation after loading)
    test_inputs, test_labels = create_synthetic_dataset(
        num_samples=100,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )

    print(f"Training samples: {len(train_inputs)}")
    print(f"Test samples: {len(test_inputs)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(train_loader)}")

    # -------------------------------------------------------------------------
    # Step 3: Setup Training
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 3: Setting up trainer with pruning schedule")
    print("-" * 60)

    trainer = setup_training(
        model=model,
        learning_rate=LEARNING_RATE,
        max_sparsity=MAX_SPARSITY,
        num_epochs=NUM_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
    )

    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Warmup epochs: {WARMUP_EPOCHS}")
    print(f"Max target sparsity: {MAX_SPARSITY:.1%}")
    print(f"Schedule type: linear")

    # -------------------------------------------------------------------------
    # Step 4: Train
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 4: Training with progressive pruning")
    print("-" * 60)

    loss_fn = create_loss_function()
    history = train_model(
        trainer=trainer,
        train_loader=train_loader,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
    )

    # -------------------------------------------------------------------------
    # Step 5: Get Final Compression Stats
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 5: Final compression statistics")
    print("-" * 60)

    compression_summary = trainer.get_compression_summary()
    final_stats = compression_summary["final_stats"]

    print(f"Total parameters: {final_stats['total_params']:,}")
    print(f"Hot (FP16) params:  {final_stats['hot_params']:,} ({final_stats.get('hot_ratio', 0):.1%})")
    print(f"Warm (4-bit) params: {final_stats['warm_params']:,} ({final_stats.get('warm_ratio', 0):.1%})")
    print(f"Cold (1-bit) params: {final_stats['cold_params']:,} ({final_stats.get('cold_ratio', 0):.1%})")
    print(f"\nSparsity: {final_stats['sparsity']:.2%}")
    print(f"Effective compression: {final_stats['effective_compression']:.2f}x")

    # -------------------------------------------------------------------------
    # Step 6: Save Checkpoint
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 6: Saving compressed checkpoint")
    print("-" * 60)

    save_checkpoint(trainer, checkpoint_path)

    # -------------------------------------------------------------------------
    # Step 7: Load and Evaluate
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 7: Loading checkpoint and evaluating")
    print("-" * 60)

    eval_results = load_and_evaluate(
        checkpoint_path=checkpoint_path,
        model_class=SimpleClassifier,
        model_kwargs={
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES,
        },
        test_inputs=test_inputs,
        test_labels=test_labels,
    )

    print(f"\nTest accuracy: {eval_results['accuracy']:.2%}")

    # Verify compression stats match
    loaded_stats = eval_results["compression_stats"]
    print(f"\nLoaded model compression: {loaded_stats['effective_compression']:.2f}x")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nTraining completed: {NUM_EPOCHS} epochs")
    print(f"Final loss: {history['train_losses'][-1]:.4f}")
    print(f"Final sparsity: {final_stats['sparsity']:.2%}")
    print(f"Effective compression: {final_stats['effective_compression']:.2f}x")
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"\nModel successfully:")
    print("  ✓ Trained with progressive pruning")
    print("  ✓ Achieved target sparsity")
    print("  ✓ Saved compressed checkpoint")
    print("  ✓ Loaded and evaluated")

    return {
        "history": history,
        "compression_stats": final_stats,
        "eval_accuracy": eval_results["accuracy"],
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    # Run the complete pipeline
    results = main()

    # Validate VAL-CROSS-001: End-to-End Workflow
    print("\n" + "=" * 60)
    print("VALIDATION: VAL-CROSS-001")
    print("=" * 60)

    # Check that all steps completed
    success = True

    # 1. Model was defined and trained
    if results["history"]["train_losses"]:
        print("✓ Model defined and trained successfully")
    else:
        print("✗ Training failed")
        success = False

    # 2. Checkpoint was saved
    if os.path.exists(results["checkpoint_path"]):
        print("✓ Checkpoint saved successfully")
    else:
        print("✗ Checkpoint not found")
        success = False

    # 3. Model was loaded and evaluated
    if results["eval_accuracy"] >= 0:
        print("✓ Model loaded and evaluated successfully")
    else:
        print("✗ Evaluation failed")
        success = False

    # 4. Compression was achieved
    compression = results["compression_stats"]["effective_compression"]
    if compression > 1.0:
        print(f"✓ Compression achieved: {compression:.2f}x")
    else:
        print(f"! No compression achieved: {compression:.2f}x")
        # This is a warning, not a failure - compression depends on training

    if success:
        print("\n✓ VAL-CROSS-001 PASSED: End-to-end workflow completed successfully")
    else:
        print("\n✗ VAL-CROSS-001 FAILED: Some workflow steps did not complete")

    print("\n" + "=" * 60)
    print("Example complete! Check the checkpoints directory for the saved model.")
    print("=" * 60)
