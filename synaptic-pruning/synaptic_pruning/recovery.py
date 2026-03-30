"""Recovery mechanism module for Synaptic Pruning.

This module implements learned recovery systems for "pruned" (quantized) weights:
- HyperNetwork: Generates weights from small latent codes
- CodebookVQ: Vector quantization with learned codebook for compression
"""

import torch
import torch.nn as nn


class HyperNetwork(nn.Module):
    """HyperNetwork that generates weight matrices from latent codes.

    Enables extreme compression by storing small latent codes instead of
    full weight matrices. The hypernetwork regenerates weights on demand.

    Attributes:
        latent_dim: Dimensionality of latent codes.
        hidden_dim: Hidden layer dimension.
        target_shape: Shape of weight matrices to generate.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        target_shape: tuple[int, ...] = (768, 768),
    ) -> None:
        """Initialize HyperNetwork.

        Args:
            latent_dim: Dimension of latent code space.
            hidden_dim: Hidden layer dimension for generation network.
            target_shape: Shape of weight matrices to generate.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.target_shape = target_shape

        # Calculate output size from target shape
        self.output_size = 1
        for dim in target_shape:
            self.output_size *= dim

        # Generation network (latent -> weights)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_size),
        )

        # Encoder network (weights -> latent) for training
        self.encoder = nn.Sequential(
            nn.Linear(self.output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize weights with small values for stable training
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        """Generate weight matrix from latent code.

        Args:
            latent_code: Latent code tensor of shape [latent_dim].

        Returns:
            Generated weight matrix of target_shape.
        """
        # Handle both single latent code [latent_dim] and batched [B, latent_dim]
        if latent_code.dim() == 1:
            latent_code = latent_code.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Generate flattened weights
        weights_flat = self.generator(latent_code)

        # Reshape to target shape
        batch_size = weights_flat.shape[0]
        if batch_size == 1 and squeeze_output:
            weights = weights_flat.view(self.target_shape)
        else:
            weights = weights_flat.view(batch_size, *self.target_shape)

        return weights

    def encode(self, weights: torch.Tensor) -> torch.Tensor:
        """Encode weight matrix to latent code.

        Args:
            weights: Weight matrix to encode, shape [target_shape] or [B, target_shape].

        Returns:
            Latent code tensor of shape [latent_dim] or [B, latent_dim].
        """
        # Handle both single and batched weights
        if weights.dim() == len(self.target_shape):
            weights = weights.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Flatten weights
        batch_size = weights.shape[0]
        weights_flat = weights.view(batch_size, self.output_size)

        # Encode to latent space
        latent = self.encoder(weights_flat)

        if squeeze_output:
            latent = latent.squeeze(0)

        return latent

    def compute_recovery_loss(
        self, original_weights: torch.Tensor, latent_code: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity loss for recovery training.

        Args:
            original_weights: Original weight matrix.
            latent_code: Latent code used to generate weights.

        Returns:
            Recovery loss (1 - cosine_similarity).
        """
        recovered_weights = self.forward(latent_code)

        # Flatten for cosine similarity computation
        original_flat = original_weights.view(-1)
        recovered_flat = recovered_weights.view(-1)

        # Compute cosine similarity
        cos_sim = nn.functional.cosine_similarity(
            original_flat.unsqueeze(0),
            recovered_flat.unsqueeze(0),
            dim=1,
        )

        # Loss is 1 - cosine_similarity (we want high similarity)
        loss = 1.0 - cos_sim.mean()

        return loss


class CodebookVQ(nn.Module):
    """Vector Quantization codebook for compressing cold weights.

    Uses a learned codebook to compress weight vectors. Each vector is
    replaced by an index into the codebook, achieving high compression.

    Attributes:
        num_embeddings: Number of codebook entries (e.g., 256).
        embedding_dim: Dimension of each codebook vector.
        commitment_cost: Commitment cost for VQ training.
    """

    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        """Initialize Codebook VQ.

        Args:
            num_embeddings: Size of codebook (number of prototypes).
            embedding_dim: Dimension of each codebook vector.
            commitment_cost: Commitment cost for VQ-VAE style training.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook embeddings
        self.embeddings = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim)
        )

        # Initialize with small normal distribution for stability
        nn.init.normal_(self.embeddings, mean=0.0, std=0.1)

    def _flatten_inputs(self, weights: torch.Tensor) -> torch.Tensor:
        """Flatten all dimensions except the last (embedding dim).

        Args:
            weights: Input tensor of shape [..., embedding_dim].

        Returns:
            Flattened tensor of shape [N, embedding_dim].
        """
        # Keep the last dimension (embedding_dim), flatten all others
        return weights.view(-1, self.embedding_dim)

    def _unflatten_outputs(
        self, flat_outputs: torch.Tensor, original_shape: torch.Size
    ) -> torch.Tensor:
        """Reshape outputs back to original shape.

        Args:
            flat_outputs: Flattened tensor [N, embedding_dim].
            original_shape: Original shape of weights [..., embedding_dim].

        Returns:
            Reshaped tensor matching original shape.
        """
        return flat_outputs.view(original_shape)

    def quantize(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight vectors using codebook.

        Args:
            weights: Weight vectors to quantize [..., embedding_dim].

        Returns:
            Tuple of (quantized_weights, indices).
                - quantized_weights: [..., embedding_dim]
                - indices: [...] (codebook indices)
        """
        original_shape = weights.shape
        flat_weights = self._flatten_inputs(weights)
        
        # Compute distances to all codebook entries
        # [N, 1, embedding_dim] - [1, num_embeddings, embedding_dim]
        # -> [N, num_embeddings, embedding_dim] -> sum -> [N, num_embeddings]
        distances = torch.sum(
            (flat_weights.unsqueeze(1) - self.embeddings.unsqueeze(0)) ** 2,
            dim=2,
        )

        # Find nearest codebook entry for each vector
        indices = torch.argmin(distances, dim=1)

        # Get quantized vectors from codebook
        quantized_flat = self.embeddings[indices]

        # Reshape back to original shape
        quantized_weights = self._unflatten_outputs(quantized_flat, original_shape)
        indices_reshaped = indices.view(original_shape[:-1])

        return quantized_weights, indices_reshaped

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to weight vectors.

        Args:
            indices: Codebook indices [...].

        Returns:
            Dequantized weight vectors [..., embedding_dim].
        """
        # Flatten indices, lookup, then reshape
        flat_indices = indices.view(-1)
        flat_outputs = self.embeddings[flat_indices]

        # Reshape to [..., embedding_dim]
        output_shape = indices.shape + (self.embedding_dim,)
        return flat_outputs.view(output_shape)

    def forward(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with quantization and straight-through estimator.

        Uses straight-through estimator to allow gradients to flow through
        the quantization operation. Also computes VQ loss for codebook training.

        Args:
            weights: Input weight vectors [..., embedding_dim].

        Returns:
            Tuple of (quantized_weights, vq_loss).
                - quantized_weights: Quantized weights with STE [..., embedding_dim]
                - vq_loss: VQ loss for codebook commitment
        """
        original_shape = weights.shape
        flat_weights = self._flatten_inputs(weights)

        # Quantize to get nearest codebook entries
        with torch.no_grad():
            distances = torch.sum(
                (flat_weights.unsqueeze(1) - self.embeddings.unsqueeze(0)) ** 2,
                dim=2,
            )
            indices = torch.argmin(distances, dim=1)
            quantized_flat = self.embeddings[indices]

        # Straight-through estimator: use quantized values in forward,
        # but gradients flow to original weights
        quantized_flat_ste = flat_weights + (quantized_flat - flat_weights).detach()

        # Reshape
        quantized_weights = self._unflatten_outputs(quantized_flat_ste, original_shape)

        # Compute VQ losses
        # Commitment loss: encourages encoder to commit to codebook entries
        commitment_loss = torch.mean(
            (quantized_flat.detach() - flat_weights) ** 2
        )

        # Codebook loss: encourages codebook entries to move towards encoder outputs
        codebook_loss = torch.mean(
            (quantized_flat - flat_weights.detach()) ** 2
        )

        # Combined VQ loss (VQ-VAE style with commitment cost)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        return quantized_weights, vq_loss
