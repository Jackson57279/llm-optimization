"""Training module for Synaptic Pruning.

This module implements the SynapticTrainer for end-to-end training with:
- Progressive pruning schedule
- Recovery loss integration
- Compression metrics tracking
"""

from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
import torch.nn as nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from synaptic_pruning.recovery import HyperNetwork


class PruningSchedule:
    """Progressive pruning schedule that increases sparsity over time.

    Implements various schedule types:
    - linear: Linear increase in sparsity over epochs
    - exponential: Exponential increase with faster pruning
    - stepped: Step function at specific epoch milestones
    - cosine: Cosine annealing schedule

    Attributes:
        max_sparsity: Target sparsity level (0.0 to 1.0)
        schedule_type: Type of schedule ("linear", "exponential", "stepped", "cosine")
        warmup_epochs: Number of epochs before pruning starts
        max_epochs: Total training epochs for schedule calculation
    """

    def __init__(
        self,
        max_sparsity: float = 0.9,
        schedule_type: Literal["linear", "exponential", "stepped", "cosine"] = "linear",
        warmup_epochs: int = 0,
        max_epochs: int = 100,
        stepped_milestones: list[tuple[int, float]] | None = None,
    ) -> None:
        """Initialize pruning schedule.

        Args:
            max_sparsity: Target sparsity level (0.0 to 1.0).
            schedule_type: Type of pruning schedule.
            warmup_epochs: Number of epochs before pruning starts.
            max_epochs: Total training epochs for schedule calculation.
            stepped_milestones: List of (epoch, sparsity) tuples for stepped schedule.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0.0 <= max_sparsity <= 1.0:
            raise ValueError(f"max_sparsity must be in [0, 1], got {max_sparsity}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {max_epochs}")

        self.max_sparsity = max_sparsity
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.stepped_milestones = stepped_milestones or [(0, 0.0), (max_epochs, max_sparsity)]

    def get_sparsity(self, epoch: int) -> float:
        """Get target sparsity for a given epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Target sparsity level for this epoch.
        """
        # During warmup, no pruning
        if epoch < self.warmup_epochs:
            return 0.0

        # Effective epoch (after warmup)
        effective_epoch = epoch - self.warmup_epochs
        effective_max = self.max_epochs - self.warmup_epochs

        if effective_max <= 0:
            return self.max_sparsity

        # Compute progress through effective epochs
        progress = effective_epoch / effective_max
        progress = max(0.0, min(1.0, progress))

        if self.schedule_type == "linear":
            return progress * self.max_sparsity

        elif self.schedule_type == "exponential":
            # Exponential: starts slow, accelerates
            return (progress ** 2) * self.max_sparsity

        elif self.schedule_type == "cosine":
            # Cosine annealing: smooth curve
            import math
            cosine_factor = 0.5 * (1 - math.cos(progress * math.pi))
            return cosine_factor * self.max_sparsity

        elif self.schedule_type == "stepped":
            # Find the current sparsity based on milestones
            current_sparsity = 0.0
            for milestone_epoch, milestone_sparsity in sorted(self.stepped_milestones):
                if epoch >= milestone_epoch:
                    current_sparsity = milestone_sparsity
            return current_sparsity

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class SynapticTrainer:
    """End-to-end trainer for Synaptic Pruning.

    Manages training of models with SynapticLayers, implementing:
    - Progressive pruning according to a schedule
    - Recovery network training
    - Compression metrics tracking
    - Gradient flow through all weight tiers

    The trainer handles both the main task loss and recovery loss,
    which encourages the recovery network to reconstruct pruned weights.

    Attributes:
        model: The model to train (should contain SynapticLayers).
        optimizer: The optimizer for training.
        pruning_schedule: Schedule for progressive pruning.
        recovery_weight: Weight for the recovery loss term.
        compression_stats: History of compression metrics.

    Example:
        >>> model = MyModel()  # With SynapticLayers
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> schedule = PruningSchedule(max_sparsity=0.9, schedule_type="linear")
        >>> trainer = SynapticTrainer(model, optimizer, pruning_schedule=schedule)
        >>> trainer.train(train_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        pruning_schedule: PruningSchedule | None = None,
        recovery_weight: float = 0.01,
        recovery_network: "HyperNetwork | None" = None,
        compression_update_freq: int = 100,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize SynapticTrainer.

        Args:
            model: The model to train. Should contain SynapticLayers for
                pruning to have effect.
            optimizer: The optimizer for training model parameters.
            pruning_schedule: Schedule for progressive pruning. If None,
                uses a default linear schedule to 90% sparsity.
            recovery_weight: Weight for the recovery loss term. Default 0.01.
            recovery_network: Optional HyperNetwork for recovery. If None,
                recovery loss is not computed.
            compression_update_freq: Update compression stats every N steps.
            device: Device to use for training. If None, uses model's device.

        Raises:
            ValueError: If parameters are invalid.
        """
        self.model = model
        self.optimizer = optimizer

        if pruning_schedule is None:
            pruning_schedule = PruningSchedule(
                max_sparsity=0.9,
                schedule_type="linear",
                max_epochs=100,
            )
        self.pruning_schedule = pruning_schedule

        if recovery_weight < 0:
            raise ValueError(f"recovery_weight must be >= 0, got {recovery_weight}")
        self.recovery_weight = recovery_weight
        self.recovery_network = recovery_network
        self.compression_update_freq = compression_update_freq

        # Determine device
        if device is None:
            # Try to infer from model parameters
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Compression metrics history
        self.compression_stats: list[dict[str, Any]] = []
        self.epoch_stats: list[dict[str, Any]] = []
        self.current_step = 0
        self.current_epoch = 0

        # Move recovery network to device if provided
        if self.recovery_network is not None:
            self.recovery_network.to(self.device)

        # Find SynapticLayers in model
        self._synaptic_layers: list[nn.Module] = []
        self._discover_synaptic_layers()

    def _discover_synaptic_layers(self) -> None:
        """Discover all SynapticLayers in the model."""
        from synaptic_pruning.layers import SynapticLayer

        self._synaptic_layers = []
        for module in self.model.modules():
            if isinstance(module, SynapticLayer):
                self._synaptic_layers.append(module)

    def _update_activity_thresholds(self, epoch: int) -> None:
        """Update activity thresholds based on pruning schedule.

        Args:
            epoch: Current epoch number.
        """
        target_sparsity = self.pruning_schedule.get_sparsity(epoch)

        # Update thresholds in all SynapticLayers
        # We adjust warm_threshold to achieve target sparsity
        # Higher warm_threshold = more cold weights = more sparsity
        for layer in self._synaptic_layers:
            # Adjust threshold to achieve target sparsity
            # This is a simple heuristic: linear interpolation
            # At 0% sparsity: use warm_threshold = 0.3 (default)
            # At 100% sparsity: use warm_threshold = 0.9
            new_threshold = 0.3 + target_sparsity * 0.6
            layer.activity_tracker.warm_threshold = new_threshold  # type: ignore

    def _compute_recovery_loss(self) -> torch.Tensor:
        """Compute recovery loss for all SynapticLayers.

        Returns:
            Recovery loss tensor. Returns 0 if no recovery network.
        """
        if self.recovery_network is None:
            return torch.tensor(0.0, device=self.device)

        from synaptic_pruning.layers import SynapticLayer

        recovery_net = self.recovery_network
        assert recovery_net is not None

        total_loss = torch.tensor(0.0, device=self.device)
        num_layers = 0

        for layer in self._synaptic_layers:
            if isinstance(layer, SynapticLayer):
                if "weight" in layer.activity_tracker.activity_scores:
                    # Get cold weights (below threshold)
                    activity = layer.activity_tracker.get_activity("weight")
                    cold_mask = activity <= layer.activity_tracker.warm_threshold

                    if cold_mask.any():
                        # Get cold weights
                        cold_weights = layer.weight[cold_mask]

                        # Encode and reconstruct
                        try:
                            latent = recovery_net.encode(cold_weights)
                            _ = recovery_net(latent)

                            # Compute cosine similarity loss
                            loss = recovery_net.compute_recovery_loss(
                                cold_weights, latent
                            )
                            total_loss = total_loss + loss
                            num_layers += 1
                        except (ValueError, RuntimeError):
                            # Skip if shape mismatch or other error
                            pass

        if num_layers > 0:
            total_loss = total_loss / num_layers

        return total_loss

    def _compute_compression_stats(self) -> dict[str, Any]:
        """Compute compression statistics for the model.

        Returns:
            Dictionary with compression metrics:
            - total_params: Total number of parameters
            - hot_params: Number of hot (FP16) weights
            - warm_params: Number of warm (4-bit) weights
            - cold_params: Number of cold (1-bit) weights
            - sparsity: Current sparsity ratio
            - effective_compression: Effective compression ratio
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
            "effective_bytes": 0.0,
        }

        for layer in self._synaptic_layers:
            if isinstance(layer, SynapticLayer):
                layer_stats = layer.get_compression_stats()
                total = layer_stats["total_params"]

                stats["total_params"] += total
                stats["hot_params"] += layer_stats["hot_count"]
                stats["warm_params"] += layer_stats["warm_count"]
                stats["cold_params"] += layer_stats["cold_count"]

        # Calculate bytes assuming:
        # - FP16: 2 bytes per param
        # - 4-bit: 0.5 bytes per param
        # - 1-bit: 0.125 bytes per param
        if stats["total_params"] > 0:
            stats["hot_bytes"] = stats["hot_params"] * 2
            stats["warm_bytes"] = int(stats["warm_params"] * 0.5)
            stats["cold_bytes"] = int(stats["cold_params"] * 0.125)
            stats["total_bytes"] = stats["hot_bytes"] + stats["warm_bytes"] + stats["cold_bytes"]
            stats["effective_bytes"] = stats["total_bytes"]

            # Calculate sparsity (cold params / total)
            stats["sparsity"] = stats["cold_params"] / stats["total_params"]

            # Calculate compression ratio
            baseline_bytes = stats["total_params"] * 2  # FP16 baseline
            if stats["total_bytes"] > 0:
                stats["effective_compression"] = baseline_bytes / stats["total_bytes"]
            else:
                stats["effective_compression"] = 1.0
        else:
            stats["sparsity"] = 0.0
            stats["effective_compression"] = 1.0

        return stats

    def _train_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute a single training step.

        Args:
            batch: Batch of training data (inputs, targets, etc.).
            loss_fn: Loss function to compute task loss.

        Returns:
            Tuple of (task_loss, recovery_loss, total_loss).
        """
        # Move batch to device
        batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()

        # Compute task loss
        task_loss = loss_fn(*batch)

        # Compute recovery loss
        recovery_loss = self._compute_recovery_loss()

        # Total loss with recovery weight
        total_loss = task_loss + self.recovery_weight * recovery_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Update compression stats periodically
        self.current_step += 1
        if self.current_step % self.compression_update_freq == 0:
            stats = self._compute_compression_stats()
            stats["step"] = self.current_step
            stats["epoch"] = self.current_epoch
            self.compression_stats.append(stats)

        return task_loss, recovery_loss, total_loss

    def train(
        self,
        train_loader: Any,
        num_epochs: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        val_loader: Any | None = None,
        log_interval: int = 10,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Train the model for specified number of epochs.

        Args:
            train_loader: Data loader for training data.
            num_epochs: Number of epochs to train.
            loss_fn: Loss function. If None, assumes model outputs loss directly.
            val_loader: Optional validation data loader.
            log_interval: Log progress every N batches.
            callback: Optional callback function(epoch, stats) called each epoch.

        Returns:
            Training history dictionary with loss curves and compression stats.
        """
        from tqdm import tqdm

        history: dict[str, Any] = {
            "train_losses": [],
            "val_losses": [],
            "recovery_losses": [],
            "compression_stats": [],
        }

        if loss_fn is None:
            # Assume model returns loss directly
            def loss_fn(*batch):  # type: ignore
                return self.model(*batch)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Update pruning thresholds
            self._update_activity_thresholds(epoch)

            # Training loop
            epoch_losses = []
            epoch_recovery_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Ensure batch is a tuple
                if not isinstance(batch, (tuple, list)):
                    batch = (batch,)

                task_loss, recovery_loss, total_loss = self._train_step(batch, loss_fn)

                epoch_losses.append(task_loss.item())
                epoch_recovery_losses.append(recovery_loss.item())

                # Update progress bar
                if batch_idx % log_interval == 0:
                    avg_loss = (
                        sum(epoch_losses[-log_interval:]) /
                        min(log_interval, len(epoch_losses))
                    )
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "recovery": f"{recovery_loss.item():.4f}",
                    })

            # Epoch statistics
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            if epoch_recovery_losses:
                avg_recovery_loss = sum(epoch_recovery_losses) / len(epoch_recovery_losses)
            else:
                avg_recovery_loss = 0.0
            history["train_losses"].append(avg_epoch_loss)
            history["recovery_losses"].append(avg_recovery_loss)

            # Compute compression stats for this epoch
            compression_stats = self._compute_compression_stats()
            history["compression_stats"].append(compression_stats)

            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, loss_fn)
                history["val_losses"].append(val_loss)
                print(
                    f"Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"Sparsity={compression_stats.get('sparsity', 0):.2%}"
                )
            else:
                print(
                    f"Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, "
                    f"Sparsity={compression_stats.get('sparsity', 0):.2%}"
                )

            # Callback
            if callback is not None:
                epoch_stats = {
                    "train_loss": avg_epoch_loss,
                    "recovery_loss": avg_recovery_loss,
                    "compression": compression_stats,
                    "epoch": epoch,
                }
                if val_loader is not None:
                    epoch_stats["val_loss"] = history["val_losses"][-1]
                callback(epoch, epoch_stats)

        # Update current_epoch to reflect completed epochs (for reporting)
        self.current_epoch = num_epochs - 1

        return history

    def evaluate(
        self,
        val_loader: Any,
        loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> float:
        """Evaluate the model on validation data.

        Args:
            val_loader: Validation data loader.
            loss_fn: Loss function. If None, assumes model outputs loss directly.

        Returns:
            Average validation loss.
        """
        self.model.eval()

        if loss_fn is None:
            def loss_fn(*batch):  # type: ignore
                return self.model(*batch)

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if not isinstance(batch, (tuple, list)):
                    batch = (batch,)

                # Move batch to device
                batch = tuple(
                    t.to(self.device) if isinstance(t, torch.Tensor) else t
                    for t in batch
                )

                loss = loss_fn(*batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def get_compression_summary(self) -> dict[str, Any]:
        """Get summary of compression statistics.

        Returns:
            Dictionary with final compression metrics and history.
        """
        final_stats = self._compute_compression_stats()

        return {
            "final_stats": final_stats,
            "compression_history": self.compression_stats,
            "epoch_history": self.epoch_stats,
            "total_steps": self.current_step,
            "total_epochs": self.current_epoch,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pruning_schedule": {
                "max_sparsity": self.pruning_schedule.max_sparsity,
                "schedule_type": self.pruning_schedule.schedule_type,
                "warmup_epochs": self.pruning_schedule.warmup_epochs,
                "max_epochs": self.pruning_schedule.max_epochs,
            },
            "recovery_weight": self.recovery_weight,
            "compression_stats": self.compression_stats,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
        }

        if self.recovery_network is not None:
            checkpoint["recovery_network_state_dict"] = self.recovery_network.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state - use strict=False because SynapticLayer's load_state_dict
        # removes activity_tracker and quantizer keys, leaving only base nn.Module keys
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.recovery_weight = checkpoint.get("recovery_weight", self.recovery_weight)
        self.compression_stats = checkpoint.get("compression_stats", [])
        self.current_step = checkpoint.get("current_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)

        if self.recovery_network is not None and "recovery_network_state_dict" in checkpoint:
            self.recovery_network.load_state_dict(checkpoint["recovery_network_state_dict"])

        # Rediscover synaptic layers
        self._discover_synaptic_layers()
