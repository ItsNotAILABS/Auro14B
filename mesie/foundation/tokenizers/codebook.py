"""Spectral codebook implementations for vector quantization.

This module provides codebook mechanisms for discretizing continuous
spectral representations into token sequences, including residual
and product quantization strategies.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SpectralCodebook:
    """Vector quantization codebook for spectral tokens.

    Maintains a codebook of prototype vectors and provides methods
    for encoding continuous vectors to discrete codes and decoding
    back. Supports EMA updates, dead code detection, and code reset.

    Attributes:
        codebook_size: Number of codebook entries.
        codebook_dim: Dimension of each code vector.
        decay: EMA decay rate for updates.
        threshold_ema_dead_code: Threshold for dead code detection.
        use_cosine: Whether to use cosine similarity.
        temperature: Softmax temperature for soft assignment.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_dim: int = 256,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2.0,
        use_cosine: bool = True,
        temperature: float = 1.0,
        orthogonal_init: bool = True,
    ):
        """Initialize spectral codebook.

        Args:
            codebook_size: Number of codes.
            codebook_dim: Code vector dimension.
            decay: EMA decay for codebook updates.
            threshold_ema_dead_code: Dead code threshold.
            use_cosine: Use cosine similarity for assignment.
            temperature: Softmax temperature.
            orthogonal_init: Initialize codes orthogonally.
        """
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.use_cosine = use_cosine
        self.temperature = temperature

        # Initialize codebook
        if orthogonal_init and codebook_size <= codebook_dim:
            # Use orthogonal initialization
            random_matrix = np.random.randn(codebook_dim, codebook_size)
            u, _, vt = np.linalg.svd(random_matrix, full_matrices=False)
            self.embeddings = u[:, :codebook_size].T.copy()
        else:
            self.embeddings = np.random.randn(codebook_size, codebook_dim)
            # Normalize
            norms = np.sqrt(np.sum(self.embeddings ** 2, axis=-1, keepdims=True))
            self.embeddings = self.embeddings / (norms + 1e-10)

        # EMA tracking
        self.ema_count = np.zeros(codebook_size)
        self.ema_weight = self.embeddings.copy()

        # Usage statistics
        self.usage_count = np.zeros(codebook_size, dtype=np.int64)
        self.total_updates = 0

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Encode continuous vectors to discrete codes.

        Args:
            x: Input vectors [..., codebook_dim].

        Returns:
            Tuple of:
            - indices: Code indices [...].
            - quantized: Quantized vectors [..., codebook_dim].
            - info: Encoding information.
        """
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.codebook_dim)
        num_vectors = flat_x.shape[0]

        if self.use_cosine:
            # Cosine similarity
            x_norm = flat_x / (np.sqrt(np.sum(flat_x ** 2, axis=-1, keepdims=True)) + 1e-10)
            emb_norm = self.embeddings / (
                np.sqrt(np.sum(self.embeddings ** 2, axis=-1, keepdims=True)) + 1e-10
            )
            similarities = np.einsum("nd,kd->nk", x_norm, emb_norm)
            indices = np.argmax(similarities, axis=-1)
        else:
            # Euclidean distance
            # ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x·e
            x_sq = np.sum(flat_x ** 2, axis=-1, keepdims=True)
            emb_sq = np.sum(self.embeddings ** 2, axis=-1)
            cross = np.einsum("nd,kd->nk", flat_x, self.embeddings)
            distances = x_sq + emb_sq - 2 * cross
            indices = np.argmin(distances, axis=-1)

        # Quantize
        quantized = self.embeddings[indices]
        quantized = quantized.reshape(*original_shape, self.codebook_dim)
        indices = indices.reshape(original_shape)

        # Update usage statistics
        for idx in indices.flatten():
            self.usage_count[idx] += 1

        # Compute info
        info: Dict[str, Any] = {
            "codebook_utilization": float(
                np.sum(self.usage_count > 0) / self.codebook_size
            ),
            "perplexity": self._compute_perplexity(indices.flatten()),
            "quantization_error": float(
                np.mean(np.sum((flat_x - self.embeddings[indices.flatten()]) ** 2, axis=-1))
            ),
        }

        return indices, quantized, info

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode code indices to vectors.

        Args:
            indices: Code indices [...].

        Returns:
            Decoded vectors [..., codebook_dim].
        """
        flat_indices = indices.flatten().astype(int)
        decoded = self.embeddings[flat_indices]
        return decoded.reshape(*indices.shape, self.codebook_dim)

    def update_ema(self, x: np.ndarray, indices: np.ndarray) -> None:
        """Update codebook with EMA.

        Args:
            x: Input vectors that were quantized [N, codebook_dim].
            indices: Assigned code indices [N].
        """
        flat_x = x.reshape(-1, self.codebook_dim)
        flat_indices = indices.flatten().astype(int)

        # Count assignments per code
        counts = np.bincount(flat_indices, minlength=self.codebook_size)

        # EMA count update
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * counts

        # EMA weight update (sum of assigned vectors per code)
        for k in range(self.codebook_size):
            mask = flat_indices == k
            if np.any(mask):
                assigned_sum = np.sum(flat_x[mask], axis=0)
                self.ema_weight[k] = (
                    self.decay * self.ema_weight[k] + (1 - self.decay) * assigned_sum
                )

        # Update embeddings
        n = self.ema_count + 1e-5
        self.embeddings = self.ema_weight / n[:, np.newaxis]

        # Reset dead codes
        dead_mask = self.ema_count < self.threshold_ema_dead_code
        if np.any(dead_mask):
            num_dead = int(np.sum(dead_mask))
            # Replace with random active vectors
            active_indices = np.where(~dead_mask)[0]
            if len(active_indices) > 0:
                replacement_indices = np.random.choice(
                    active_indices, size=num_dead, replace=True
                )
                self.embeddings[dead_mask] = self.embeddings[replacement_indices]
                # Add noise
                self.embeddings[dead_mask] += np.random.randn(num_dead, self.codebook_dim) * 0.01

        self.total_updates += 1

    def _compute_perplexity(self, indices: np.ndarray) -> float:
        """Compute codebook perplexity (effective codebook usage)."""
        counts = np.bincount(indices.astype(int), minlength=self.codebook_size)
        probs = counts / (np.sum(counts) + 1e-10)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(np.exp(entropy))

    def get_statistics(self) -> Dict[str, Any]:
        """Get codebook statistics."""
        return {
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "utilization": float(np.sum(self.usage_count > 0) / self.codebook_size),
            "dead_codes": int(np.sum(self.usage_count == 0)),
            "total_updates": self.total_updates,
            "max_usage": int(np.max(self.usage_count)),
            "min_usage": int(np.min(self.usage_count)),
            "usage_std": float(np.std(self.usage_count)),
        }


class ResidualQuantizer:
    """Residual Vector Quantization (RVQ) for high-fidelity encoding.

    Uses multiple codebook stages where each stage quantizes the
    residual from the previous stage, enabling progressively finer
    approximations of the input.

    Attributes:
        num_codebooks: Number of residual stages.
        codebook_size: Size of each codebook.
        codebook_dim: Dimension of code vectors.
        codebooks: List of codebook stages.
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 8192,
        codebook_dim: int = 256,
        decay: float = 0.99,
        use_cosine: bool = True,
    ):
        """Initialize residual quantizer.

        Args:
            num_codebooks: Number of residual stages.
            codebook_size: Size per codebook.
            codebook_dim: Code vector dimension.
            decay: EMA decay rate.
            use_cosine: Whether to use cosine similarity.
        """
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Create codebook stages
        self.codebooks = [
            SpectralCodebook(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                decay=decay,
                use_cosine=use_cosine,
            )
            for _ in range(num_codebooks)
        ]

        # Per-stage scaling factors (learned importance)
        self.stage_scales = np.ones(num_codebooks)
        # Exponentially decreasing: each stage contributes less
        for i in range(num_codebooks):
            self.stage_scales[i] = 0.5 ** i

    def encode(
        self, x: np.ndarray, num_stages: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Encode with residual quantization.

        Args:
            x: Input vectors [..., codebook_dim].
            num_stages: Number of stages to use (default: all).

        Returns:
            Tuple of:
            - all_indices: Code indices [..., num_stages].
            - quantized: Final quantized vectors [..., codebook_dim].
            - info: Quantization information.
        """
        stages = num_stages or self.num_codebooks
        original_shape = x.shape[:-1]

        residual = x.copy()
        quantized = np.zeros_like(x)
        all_indices = []
        stage_infos = []

        for i in range(stages):
            # Quantize residual
            indices, stage_quantized, info = self.codebooks[i].encode(residual)
            all_indices.append(indices)
            stage_infos.append(info)

            # Accumulate quantized output
            quantized = quantized + stage_quantized * self.stage_scales[i]

            # Update residual
            residual = residual - stage_quantized

        # Stack indices
        indices_array = np.stack(all_indices, axis=-1)

        # Aggregate info
        total_info: Dict[str, Any] = {
            "num_stages": stages,
            "per_stage_info": stage_infos,
            "total_quantization_error": float(np.mean(np.sum(residual ** 2, axis=-1))),
            "reconstruction_quality": float(
                1.0 - np.mean(np.sum(residual ** 2, axis=-1)) /
                (np.mean(np.sum(x ** 2, axis=-1)) + 1e-10)
            ),
        }

        return indices_array, quantized, total_info

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode from residual quantization codes.

        Args:
            indices: Code indices [..., num_stages].

        Returns:
            Reconstructed vectors [..., codebook_dim].
        """
        num_stages = indices.shape[-1]
        quantized = np.zeros((*indices.shape[:-1], self.codebook_dim))

        for i in range(num_stages):
            stage_indices = indices[..., i]
            stage_decoded = self.codebooks[i].decode(stage_indices)
            quantized = quantized + stage_decoded * self.stage_scales[i]

        return quantized

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        """Update all codebook stages with new data.

        Args:
            x: Training vectors [..., codebook_dim].

        Returns:
            Update statistics.
        """
        residual = x.copy()
        stats: Dict[str, Any] = {}

        for i in range(self.num_codebooks):
            indices, quantized, _ = self.codebooks[i].encode(residual)
            self.codebooks[i].update_ema(residual, indices)
            residual = residual - quantized
            stats[f"stage_{i}_error"] = float(np.mean(np.sum(residual ** 2, axis=-1)))

        return stats


class ProductQuantizer:
    """Product Quantization for efficient high-dimensional encoding.

    Splits the input vector into sub-vectors and quantizes each
    independently, allowing exponentially large effective codebook
    sizes with linear storage.

    Attributes:
        num_subspaces: Number of sub-vector spaces.
        codebook_size: Size of each sub-codebook.
        input_dim: Total input dimension.
        sub_dim: Dimension per sub-space.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_subspaces: int = 8,
        codebook_size: int = 256,
        decay: float = 0.99,
    ):
        """Initialize product quantizer.

        Args:
            input_dim: Total input dimension.
            num_subspaces: Number of sub-spaces to split into.
            codebook_size: Size of each sub-codebook.
            decay: EMA decay rate.
        """
        self.input_dim = input_dim
        self.num_subspaces = num_subspaces
        self.codebook_size = codebook_size
        self.sub_dim = input_dim // num_subspaces

        if input_dim % num_subspaces != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible by "
                f"num_subspaces ({num_subspaces})"
            )

        # Create sub-codebooks
        self.sub_codebooks = [
            SpectralCodebook(
                codebook_size=codebook_size,
                codebook_dim=self.sub_dim,
                decay=decay,
                use_cosine=False,  # Euclidean for PQ
            )
            for _ in range(num_subspaces)
        ]

        # Effective codebook size
        self.effective_size = codebook_size ** num_subspaces

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Encode with product quantization.

        Args:
            x: Input vectors [..., input_dim].

        Returns:
            Tuple of:
            - indices: Sub-code indices [..., num_subspaces].
            - quantized: Reconstructed vectors [..., input_dim].
            - info: Encoding information.
        """
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.input_dim)

        # Split into sub-vectors
        sub_vectors = np.split(flat_x, self.num_subspaces, axis=-1)

        all_indices = []
        all_quantized = []
        sub_infos = []

        for i, (sub_vec, sub_codebook) in enumerate(zip(sub_vectors, self.sub_codebooks)):
            indices, quantized, info = sub_codebook.encode(sub_vec)
            all_indices.append(indices)
            all_quantized.append(quantized)
            sub_infos.append(info)

        # Concatenate
        indices_array = np.stack(all_indices, axis=-1)
        quantized_full = np.concatenate(all_quantized, axis=-1)

        # Reshape
        indices_array = indices_array.reshape(*original_shape, self.num_subspaces)
        quantized_full = quantized_full.reshape(*original_shape, self.input_dim)

        info: Dict[str, Any] = {
            "effective_codebook_size": self.effective_size,
            "bits_per_vector": self.num_subspaces * math.log2(self.codebook_size),
            "sub_space_infos": sub_infos,
            "total_error": float(np.mean(np.sum((flat_x - quantized_full.reshape(-1, self.input_dim)) ** 2, axis=-1))),
        }

        return indices_array, quantized_full, info

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode from product quantization codes.

        Args:
            indices: Sub-code indices [..., num_subspaces].

        Returns:
            Reconstructed vectors [..., input_dim].
        """
        original_shape = indices.shape[:-1]
        flat_indices = indices.reshape(-1, self.num_subspaces)

        sub_decoded = []
        for i in range(self.num_subspaces):
            decoded = self.sub_codebooks[i].decode(flat_indices[:, i])
            sub_decoded.append(decoded)

        full_decoded = np.concatenate(sub_decoded, axis=-1)
        return full_decoded.reshape(*original_shape, self.input_dim)
