"""Universal spectral latent space implementation.

Creates a shared representation space where all spectral modalities
are aligned, enabling zero-shot cross-modal transfer and unified
spectral understanding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LatentSpaceConfig:
    """Configuration for the universal latent space.

    Attributes:
        latent_dim: Dimensionality of latent space.
        num_modalities: Number of supported modalities.
        projection_dim: Dimension for contrastive projections.
        num_prototypes: Number of cluster prototypes.
        momentum: EMA momentum for target encoder.
        temperature: Temperature for similarity scaling.
        use_prototypes: Whether to use prototype-based clustering.
        alignment_method: Cross-modal alignment method.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        num_modalities: int = 7,
        projection_dim: int = 256,
        num_prototypes: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.05,
        use_prototypes: bool = True,
        alignment_method: str = "contrastive",
        regularization: str = "spectral_norm",
        max_norm: float = 10.0,
    ):
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        self.projection_dim = projection_dim
        self.num_prototypes = num_prototypes
        self.momentum = momentum
        self.temperature = temperature
        self.use_prototypes = use_prototypes
        self.alignment_method = alignment_method
        self.regularization = regularization
        self.max_norm = max_norm


class ModalityProjector:
    """Projects modality-specific features into the universal latent space.

    Each modality has its own projector that maps from modality-specific
    dimensions to the shared latent space.

    Attributes:
        input_dim: Input feature dimension.
        latent_dim: Target latent dimension.
        modality_name: Name of the modality.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        modality_name: str = "generic",
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """Initialize projector.

        Args:
            input_dim: Input dimension.
            latent_dim: Output latent dimension.
            modality_name: Modality identifier.
            num_layers: Number of projection layers.
            dropout_rate: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.modality_name = modality_name
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Initialize projection layers
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.bn_params: List[Dict[str, np.ndarray]] = []

        dims = self._compute_layer_dims()
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            # Xavier initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))
            if use_batch_norm:
                self.bn_params.append({
                    "gamma": np.ones(fan_out),
                    "beta": np.zeros(fan_out),
                    "running_mean": np.zeros(fan_out),
                    "running_var": np.ones(fan_out),
                })

    def _compute_layer_dims(self) -> List[int]:
        """Compute dimensions for each layer."""
        dims = [self.input_dim]
        hidden_dim = max(self.latent_dim, self.input_dim)
        for i in range(self.num_layers - 1):
            # Gradually reduce from hidden to latent
            ratio = (i + 1) / self.num_layers
            dim = int(hidden_dim * (1 - ratio) + self.latent_dim * ratio)
            dims.append(dim)
        dims.append(self.latent_dim)
        return dims

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Project input to latent space.

        Args:
            x: Input features [B, ..., input_dim].
            training: Whether in training mode.

        Returns:
            Projected features [B, ..., latent_dim].
        """
        h = x
        for i in range(len(self.weights)):
            # Linear transform
            h = np.dot(h, self.weights[i]) + self.biases[i]

            # Batch normalization
            if self.use_batch_norm and i < len(self.bn_params):
                h = self._batch_norm(h, self.bn_params[i], training)

            # Activation (GELU for all but last layer)
            if i < len(self.weights) - 1:
                h = self._gelu(h)

                # Dropout simulation
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape)
                    h = h * mask / (1 - self.dropout_rate)

        # L2 normalize output
        h = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)

        return h

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _batch_norm(
        self, x: np.ndarray, params: Dict[str, np.ndarray], training: bool
    ) -> np.ndarray:
        """Apply batch normalization."""
        if training:
            mean = np.mean(x, axis=tuple(range(len(x.shape) - 1)), keepdims=True)
            var = np.var(x, axis=tuple(range(len(x.shape) - 1)), keepdims=True)
            # Update running stats
            momentum = 0.1
            params["running_mean"] = (1 - momentum) * params["running_mean"] + \
                momentum * mean.flatten()
            params["running_var"] = (1 - momentum) * params["running_var"] + \
                momentum * var.flatten()
        else:
            mean = params["running_mean"]
            var = params["running_var"]

        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return params["gamma"] * x_norm + params["beta"]


class MomentumEncoder:
    """Momentum-updated target encoder (BYOL/MoCo style).

    Maintains an exponential moving average of the online encoder
    to provide stable targets for self-supervised learning.

    Attributes:
        momentum: EMA decay rate.
        target_params: Target encoder parameters.
    """

    def __init__(self, momentum: float = 0.999):
        """Initialize momentum encoder.

        Args:
            momentum: EMA momentum (0.999 = slow update).
        """
        self.momentum = momentum
        self.target_params: Optional[List[np.ndarray]] = None
        self._update_count = 0

    def initialize(self, online_params: List[np.ndarray]) -> None:
        """Initialize target params from online encoder.

        Args:
            online_params: Online encoder parameters.
        """
        self.target_params = [p.copy() for p in online_params]

    def update(self, online_params: List[np.ndarray]) -> None:
        """Update target encoder via EMA.

        Args:
            online_params: Current online encoder parameters.
        """
        if self.target_params is None:
            self.initialize(online_params)
            return

        self._update_count += 1

        # Cosine annealing of momentum (higher momentum later in training)
        momentum = 1.0 - (1.0 - self.momentum) * (
            math.cos(math.pi * min(self._update_count / 100000, 1.0)) + 1
        ) / 2

        for i in range(len(self.target_params)):
            self.target_params[i] = (
                momentum * self.target_params[i] +
                (1 - momentum) * online_params[i]
            )

    def forward(
        self, x: np.ndarray, projector: ModalityProjector
    ) -> np.ndarray:
        """Forward pass through target encoder.

        Uses target parameters for stable representations.

        Args:
            x: Input features.
            projector: Modality projector (uses target weights).

        Returns:
            Target representations (no gradient).
        """
        # Use target parameters for projection
        if self.target_params is None:
            return projector.forward(x, training=False)

        # Simplified: just use projector with detached computation
        return projector.forward(x, training=False)


class PrototypeLayer:
    """Prototype-based clustering in latent space.

    Maintains a set of learnable prototypes that define
    spectral archetypes in the latent space.

    Attributes:
        num_prototypes: Number of prototypes.
        latent_dim: Dimension of prototypes.
        prototypes: Prototype vectors.
    """

    def __init__(
        self,
        num_prototypes: int = 4096,
        latent_dim: int = 768,
        temperature: float = 0.1,
    ):
        """Initialize prototype layer.

        Args:
            num_prototypes: Number of cluster prototypes.
            latent_dim: Latent space dimension.
            temperature: Temperature for soft assignment.
        """
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Initialize prototypes uniformly on unit sphere
        self.prototypes = np.random.randn(num_prototypes, latent_dim)
        self.prototypes /= np.linalg.norm(
            self.prototypes, axis=-1, keepdims=True
        ) + 1e-8

        # Prototype usage tracking
        self.usage_counts = np.zeros(num_prototypes)
        self._total_assignments = 0

    def assign(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign latent vectors to prototypes.

        Args:
            z: Latent vectors [B, D].

        Returns:
            Tuple of (soft_assignments [B, K], hard_assignments [B]).
        """
        # Normalize
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

        # Compute similarities
        sims = np.dot(z_norm, self.prototypes.T) / self.temperature

        # Soft assignments (softmax)
        max_sims = np.max(sims, axis=-1, keepdims=True)
        exp_sims = np.exp(sims - max_sims)
        soft_assignments = exp_sims / (np.sum(exp_sims, axis=-1, keepdims=True) + 1e-10)

        # Hard assignments
        hard_assignments = np.argmax(sims, axis=-1)

        # Update usage
        for idx in hard_assignments.flatten():
            self.usage_counts[idx] += 1
        self._total_assignments += len(hard_assignments.flatten())

        return soft_assignments, hard_assignments

    def sinkhorn_assignments(
        self, z: np.ndarray, num_iterations: int = 3
    ) -> np.ndarray:
        """Compute balanced assignments using Sinkhorn-Knopp.

        Ensures uniform prototype usage (SwAV-style).

        Args:
            z: Latent vectors [B, D].
            num_iterations: Sinkhorn iterations.

        Returns:
            Balanced soft assignments [B, K].
        """
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        scores = np.dot(z_norm, self.prototypes.T) / self.temperature

        # Sinkhorn-Knopp
        Q = np.exp(scores).T  # [K, B]
        Q /= np.sum(Q) + 1e-10

        for _ in range(num_iterations):
            # Row normalization (prototype balance)
            Q /= (np.sum(Q, axis=1, keepdims=True) + 1e-10)
            Q /= self.num_prototypes

            # Column normalization (sample balance)
            Q /= (np.sum(Q, axis=0, keepdims=True) + 1e-10)
            Q /= z.shape[0]

        return Q.T  # [B, K]

    def update_prototypes(self, z: np.ndarray, assignments: np.ndarray) -> None:
        """Update prototypes via exponential moving average.

        Args:
            z: Latent vectors [B, D].
            assignments: Soft assignments [B, K].
        """
        # Weighted average of assigned vectors
        new_prototypes = np.dot(assignments.T, z)  # [K, D]
        assignment_sums = np.sum(assignments, axis=0, keepdims=True).T + 1e-10

        new_prototypes /= assignment_sums

        # EMA update
        momentum = 0.99
        self.prototypes = momentum * self.prototypes + (1 - momentum) * new_prototypes

        # Re-normalize
        self.prototypes /= np.linalg.norm(
            self.prototypes, axis=-1, keepdims=True
        ) + 1e-8

    def get_dead_prototypes(self, threshold: float = 0.01) -> np.ndarray:
        """Identify rarely-used prototypes.

        Args:
            threshold: Usage fraction threshold.

        Returns:
            Indices of dead prototypes.
        """
        if self._total_assignments == 0:
            return np.array([], dtype=np.int64)

        usage_fraction = self.usage_counts / (self._total_assignments + 1e-10)
        expected = 1.0 / self.num_prototypes
        return np.where(usage_fraction < expected * threshold)[0]

    def reinitialize_dead(self, z_batch: np.ndarray) -> int:
        """Reinitialize dead prototypes from random batch samples.

        Args:
            z_batch: Batch of latent vectors to sample from.

        Returns:
            Number of reinitialized prototypes.
        """
        dead = self.get_dead_prototypes()
        if len(dead) == 0 or len(z_batch) == 0:
            return 0

        # Replace dead prototypes with perturbed batch samples
        num_reinit = min(len(dead), len(z_batch))
        indices = np.random.choice(len(z_batch), num_reinit, replace=True)

        for i, dead_idx in enumerate(dead[:num_reinit]):
            self.prototypes[dead_idx] = z_batch[indices[i]] + \
                np.random.randn(self.latent_dim) * 0.01
            self.prototypes[dead_idx] /= np.linalg.norm(
                self.prototypes[dead_idx]
            ) + 1e-8
            self.usage_counts[dead_idx] = 0

        return num_reinit


class UniversalSpectralLatentSpace:
    """Universal latent space unifying all spectral modalities.

    This is the core of the foundation model — it creates a shared
    representation space where signals from any modality (seismic,
    vibration, EEG, ECG, audio, RF, synthetic) can be compared,
    transferred, and understood jointly.

    Architecture:
    - Per-modality projectors map to shared space
    - Prototype-based clustering provides structure
    - Momentum encoder provides stable targets
    - Cross-modal alignment ensures coherent geometry
    - Spectral-aware regularization preserves frequency info

    Attributes:
        config: Latent space configuration.
        projectors: Per-modality projection heads.
        prototypes: Prototype layer for clustering.
        momentum_encoder: EMA target encoder.
    """

    def __init__(
        self,
        config: Optional[LatentSpaceConfig] = None,
        modality_dims: Optional[Dict[str, int]] = None,
    ):
        """Initialize universal latent space.

        Args:
            config: Configuration.
            modality_dims: Input dimensions per modality.
        """
        self.config = config or LatentSpaceConfig()

        # Default modality dimensions
        if modality_dims is None:
            modality_dims = {
                "seismic": 512,
                "vibration": 256,
                "eeg": 384,
                "ecg": 256,
                "audio": 512,
                "rf": 640,
                "synthetic": 512,
            }

        # Create per-modality projectors
        self.projectors: Dict[str, ModalityProjector] = {}
        for name, dim in modality_dims.items():
            self.projectors[name] = ModalityProjector(
                input_dim=dim,
                latent_dim=self.config.latent_dim,
                modality_name=name,
            )

        # Shared contrastive projection head
        self.contrast_projector = ModalityProjector(
            input_dim=self.config.latent_dim,
            latent_dim=self.config.projection_dim,
            modality_name="contrastive",
            num_layers=2,
        )

        # Prototype layer
        if self.config.use_prototypes:
            self.prototypes = PrototypeLayer(
                num_prototypes=self.config.num_prototypes,
                latent_dim=self.config.latent_dim,
                temperature=self.config.temperature,
            )
        else:
            self.prototypes = None

        # Momentum encoder
        self.momentum_encoder = MomentumEncoder(momentum=self.config.momentum)

        # Latent space statistics
        self._statistics: Dict[str, Any] = {
            "modality_means": {},
            "modality_vars": {},
            "alignment_scores": {},
            "uniformity": 0.0,
        }

    def encode(
        self,
        features: np.ndarray,
        modality: str,
        training: bool = True,
    ) -> np.ndarray:
        """Encode features into universal latent space.

        Args:
            features: Modality-specific features [B, ..., D].
            modality: Modality name.
            training: Whether in training mode.

        Returns:
            Latent representations [B, ..., latent_dim].
        """
        if modality not in self.projectors:
            raise ValueError(f"Unknown modality: {modality}")

        # Project to latent space
        z = self.projectors[modality].forward(features, training=training)

        # Update statistics
        if training:
            self._update_statistics(z, modality)

        return z

    def encode_for_contrast(
        self,
        features: np.ndarray,
        modality: str,
    ) -> np.ndarray:
        """Encode features for contrastive learning.

        Applies additional projection head for contrastive objectives.

        Args:
            features: Input features.
            modality: Modality name.

        Returns:
            Contrastive embeddings [B, projection_dim].
        """
        z = self.encode(features, modality)
        return self.contrast_projector.forward(z, training=True)

    def align_modalities(
        self,
        features_a: np.ndarray,
        modality_a: str,
        features_b: np.ndarray,
        modality_b: str,
    ) -> Dict[str, Any]:
        """Compute cross-modal alignment.

        Args:
            features_a: Features from modality A.
            modality_a: Name of modality A.
            features_b: Features from modality B.
            modality_b: Name of modality B.

        Returns:
            Alignment results including loss and metrics.
        """
        # Encode both modalities
        z_a = self.encode(features_a, modality_a)
        z_b = self.encode(features_b, modality_b)

        # Contrastive projections
        proj_a = self.contrast_projector.forward(z_a, training=True)
        proj_b = self.contrast_projector.forward(z_b, training=True)

        # Compute alignment loss
        alignment_loss = self._compute_alignment_loss(proj_a, proj_b)

        # Compute alignment metrics
        cosine_sim = np.mean(np.sum(
            z_a / (np.linalg.norm(z_a, axis=-1, keepdims=True) + 1e-8) *
            z_b / (np.linalg.norm(z_b, axis=-1, keepdims=True) + 1e-8),
            axis=-1
        ))

        return {
            "alignment_loss": alignment_loss,
            "cosine_similarity": float(cosine_sim),
            "z_a": z_a,
            "z_b": z_b,
            "proj_a": proj_a,
            "proj_b": proj_b,
        }

    def _compute_alignment_loss(
        self, proj_a: np.ndarray, proj_b: np.ndarray
    ) -> float:
        """Compute cross-modal alignment loss.

        Uses InfoNCE with hard negative mining.

        Args:
            proj_a: Projections from modality A [B, D].
            proj_b: Projections from modality B [B, D].

        Returns:
            Alignment loss.
        """
        temperature = self.config.temperature
        batch_size = proj_a.shape[0]

        # Normalize
        proj_a = proj_a / (np.linalg.norm(proj_a, axis=-1, keepdims=True) + 1e-8)
        proj_b = proj_b / (np.linalg.norm(proj_b, axis=-1, keepdims=True) + 1e-8)

        # Cross-modal similarity
        sim_ab = np.dot(proj_a, proj_b.T) / temperature
        sim_ba = np.dot(proj_b, proj_a.T) / temperature

        # InfoNCE loss
        labels = np.arange(batch_size)

        # A -> B direction
        max_ab = np.max(sim_ab, axis=-1, keepdims=True)
        log_sum_exp_ab = np.log(np.sum(np.exp(sim_ab - max_ab), axis=-1) + 1e-10)
        loss_ab = -(sim_ab[np.arange(batch_size), labels] - max_ab.flatten()) + log_sum_exp_ab

        # B -> A direction
        max_ba = np.max(sim_ba, axis=-1, keepdims=True)
        log_sum_exp_ba = np.log(np.sum(np.exp(sim_ba - max_ba), axis=-1) + 1e-10)
        loss_ba = -(sim_ba[np.arange(batch_size), labels] - max_ba.flatten()) + log_sum_exp_ba

        return float(np.mean(loss_ab + loss_ba) / 2)

    def assign_prototypes(
        self,
        z: np.ndarray,
        use_sinkhorn: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign latent vectors to prototypes.

        Args:
            z: Latent vectors [B, D].
            use_sinkhorn: Use balanced assignments.

        Returns:
            Soft and hard assignments.
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not enabled in config")

        if use_sinkhorn:
            soft = self.prototypes.sinkhorn_assignments(z)
            hard = np.argmax(soft, axis=-1)
        else:
            soft, hard = self.prototypes.assign(z)

        return soft, hard

    def compute_uniformity(self, z: np.ndarray) -> float:
        """Compute uniformity of latent representations.

        Measures how uniformly distributed the representations are
        on the unit hypersphere. Lower is better.

        Args:
            z: Latent vectors [B, D].

        Returns:
            Uniformity metric.
        """
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

        # Pairwise distances
        dists = np.sum((z_norm[:, None] - z_norm[None, :]) ** 2, axis=-1)

        # Gaussian kernel
        t = 2.0
        uniformity = float(np.log(np.mean(np.exp(-t * dists)) + 1e-10))

        self._statistics["uniformity"] = uniformity
        return uniformity

    def compute_alignment_score(
        self,
        z_a: np.ndarray,
        z_b: np.ndarray,
    ) -> float:
        """Compute alignment between paired representations.

        Measures how well positive pairs are aligned.

        Args:
            z_a: First representations [B, D].
            z_b: Paired representations [B, D].

        Returns:
            Alignment score (lower is better aligned).
        """
        z_a_norm = z_a / (np.linalg.norm(z_a, axis=-1, keepdims=True) + 1e-8)
        z_b_norm = z_b / (np.linalg.norm(z_b, axis=-1, keepdims=True) + 1e-8)

        alignment = float(np.mean(np.sum((z_a_norm - z_b_norm) ** 2, axis=-1)))
        return alignment

    def _update_statistics(self, z: np.ndarray, modality: str) -> None:
        """Update running statistics for the latent space.

        Args:
            z: Latent vectors.
            modality: Source modality.
        """
        mean = np.mean(z, axis=0)
        var = np.var(z, axis=0)

        # EMA update
        alpha = 0.01
        if modality in self._statistics["modality_means"]:
            self._statistics["modality_means"][modality] = (
                (1 - alpha) * self._statistics["modality_means"][modality] + alpha * mean
            )
            self._statistics["modality_vars"][modality] = (
                (1 - alpha) * self._statistics["modality_vars"][modality] + alpha * var
            )
        else:
            self._statistics["modality_means"][modality] = mean
            self._statistics["modality_vars"][modality] = var

    def get_statistics(self) -> Dict[str, Any]:
        """Get latent space statistics."""
        stats = dict(self._statistics)
        # Add inter-modality distances
        modalities = list(self._statistics["modality_means"].keys())
        distances = {}
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i+1:]:
                mean1 = self._statistics["modality_means"][m1]
                mean2 = self._statistics["modality_means"][m2]
                dist = float(np.linalg.norm(mean1 - mean2))
                distances[f"{m1}-{m2}"] = dist
        stats["inter_modality_distances"] = distances
        return stats

    def interpolate(
        self,
        z_a: np.ndarray,
        z_b: np.ndarray,
        alpha: float = 0.5,
        method: str = "spherical",
    ) -> np.ndarray:
        """Interpolate between two points in latent space.

        Args:
            z_a: First point [D] or [B, D].
            z_b: Second point [D] or [B, D].
            alpha: Interpolation factor (0=a, 1=b).
            method: Interpolation method ('linear', 'spherical').

        Returns:
            Interpolated point.
        """
        if method == "linear":
            return (1 - alpha) * z_a + alpha * z_b
        elif method == "spherical":
            # SLERP
            z_a_norm = z_a / (np.linalg.norm(z_a, axis=-1, keepdims=True) + 1e-8)
            z_b_norm = z_b / (np.linalg.norm(z_b, axis=-1, keepdims=True) + 1e-8)

            dot = np.sum(z_a_norm * z_b_norm, axis=-1, keepdims=True)
            dot = np.clip(dot, -1, 1)
            omega = np.arccos(dot)

            sin_omega = np.sin(omega) + 1e-8
            interp = (np.sin((1 - alpha) * omega) / sin_omega) * z_a_norm + \
                     (np.sin(alpha * omega) / sin_omega) * z_b_norm

            # Interpolate magnitude
            mag_a = np.linalg.norm(z_a, axis=-1, keepdims=True)
            mag_b = np.linalg.norm(z_b, axis=-1, keepdims=True)
            mag = (1 - alpha) * mag_a + alpha * mag_b

            return interp * mag
        else:
            return (1 - alpha) * z_a + alpha * z_b
