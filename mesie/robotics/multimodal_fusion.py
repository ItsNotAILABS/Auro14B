"""Multi-modal signal fusion for embodied AI.

Treats spectral signals as a 'virtual chip' layer that combines with
vision, audio, tactile, and other modalities for richer AI perception.
Enables cross-modal attention and joint spectral-spatial representations.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class Modality(Enum):
    """Supported signal modalities for fusion."""

    SPECTRAL = "spectral"
    VISION = "vision"
    AUDIO = "audio"
    TACTILE = "tactile"
    IMU = "imu"
    LIDAR = "lidar"
    THERMAL = "thermal"
    PROPRIOCEPTION = "proprioception"


class FusionStrategy(Enum):
    """Strategies for combining multi-modal representations."""

    CONCATENATE = "concatenate"
    ATTENTION_WEIGHTED = "attention_weighted"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    GATED_FUSION = "gated_fusion"
    SPECTRAL_ANCHORED = "spectral_anchored"


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion.

    Args:
        strategy: Fusion strategy to use.
        output_dim: Dimension of fused representation.
        spectral_weight: Base weight for spectral modality (anchor).
        temporal_alignment: Whether to align modalities temporally.
        cross_attention_heads: Number of cross-attention heads.
        dropout_rate: Dropout for regularization.
    """

    strategy: FusionStrategy = FusionStrategy.SPECTRAL_ANCHORED
    output_dim: int = 256
    spectral_weight: float = 0.4
    temporal_alignment: bool = True
    cross_attention_heads: int = 4
    dropout_rate: float = 0.1


@dataclass
class ModalityStream:
    """A single modality input stream.

    Attributes:
        modality: Type of modality.
        data: Raw signal data (shape depends on modality).
        embedding: Projected embedding vector.
        timestamp: Acquisition timestamp.
        confidence: Confidence in this modality's data quality.
        stream_id: Unique stream identifier.
    """

    modality: Modality
    data: np.ndarray
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    stream_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class FusedRepresentation:
    """Result of multi-modal fusion.

    Attributes:
        vector: Fused representation vector.
        modality_contributions: Per-modality contribution weights.
        attention_map: Cross-modal attention weights (if applicable).
        timestamp: When fusion was computed.
        fusion_id: Unique identifier for this fusion result.
        quality_score: Overall quality/confidence of the fusion.
    """

    vector: np.ndarray
    modality_contributions: Dict[str, float] = field(default_factory=dict)
    attention_map: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    fusion_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    quality_score: float = 1.0


class MultiModalFusion:
    """Multi-modal signal fusion engine for embodied AI.

    Combines spectral intelligence with other sensor modalities
    (vision, audio, tactile, etc.) into a unified representation
    suitable for robotic control and embodied reasoning.

    Uses spectral signals as the anchor modality — treating them
    as a 'virtual chip' layer that other modalities project into.

    Args:
        config: Fusion configuration.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        self._streams: Dict[Modality, ModalityStream] = {}
        self._projectors: Dict[Modality, np.ndarray] = {}
        self._fusion_count = 0
        self._initialized = False

    def register_modality(
        self,
        modality: Modality,
        input_dim: int,
    ) -> None:
        """Register a modality with its input dimension.

        Creates a random projection matrix for this modality to map
        into the shared fusion space.

        Args:
            modality: Modality type to register.
            input_dim: Dimension of raw modality vectors.
        """
        # Initialize random projection (Xavier-like)
        scale = np.sqrt(2.0 / (input_dim + self.config.output_dim))
        self._projectors[modality] = np.random.randn(
            input_dim, self.config.output_dim
        ) * scale
        self._initialized = True

    def feed(self, stream: ModalityStream) -> None:
        """Feed a modality stream for fusion.

        Projects the raw data into the shared space and stores
        it for the next fusion step.

        Args:
            stream: Modality stream with raw data.
        """
        modality = stream.modality
        if modality not in self._projectors:
            # Auto-register with inferred dimension
            data_flat = stream.data.ravel()
            self.register_modality(modality, len(data_flat))

        # Project into shared space
        data_flat = stream.data.ravel().astype(np.float64)
        proj = self._projectors[modality]

        # Handle dimension mismatch
        if len(data_flat) != proj.shape[0]:
            # Resize via interpolation
            x_old = np.linspace(0, 1, len(data_flat))
            x_new = np.linspace(0, 1, proj.shape[0])
            data_flat = np.interp(x_new, x_old, data_flat)

        stream.embedding = data_flat @ proj
        self._streams[modality] = stream

    def fuse(self) -> FusedRepresentation:
        """Fuse all current modality streams into unified representation.

        Applies the configured fusion strategy to combine all
        registered modality embeddings.

        Returns:
            FusedRepresentation with the combined vector.
        """
        self._fusion_count += 1

        if not self._streams:
            return FusedRepresentation(
                vector=np.zeros(self.config.output_dim),
                quality_score=0.0,
            )

        if self.config.strategy == FusionStrategy.CONCATENATE:
            return self._fuse_concatenate()
        elif self.config.strategy == FusionStrategy.ATTENTION_WEIGHTED:
            return self._fuse_attention_weighted()
        elif self.config.strategy == FusionStrategy.SPECTRAL_ANCHORED:
            return self._fuse_spectral_anchored()
        elif self.config.strategy == FusionStrategy.GATED_FUSION:
            return self._fuse_gated()
        else:
            return self._fuse_attention_weighted()

    def _fuse_concatenate(self) -> FusedRepresentation:
        """Simple concatenation with truncation/padding to output_dim."""
        embeddings = []
        contributions: Dict[str, float] = {}

        for modality, stream in self._streams.items():
            if stream.embedding is not None:
                embeddings.append(stream.embedding)
                contributions[modality.value] = 1.0 / len(self._streams)

        if not embeddings:
            return FusedRepresentation(
                vector=np.zeros(self.config.output_dim),
                quality_score=0.0,
            )

        concat = np.concatenate(embeddings)
        # Resize to output_dim
        if len(concat) != self.config.output_dim:
            x_old = np.linspace(0, 1, len(concat))
            x_new = np.linspace(0, 1, self.config.output_dim)
            concat = np.interp(x_new, x_old, concat)

        norm = np.linalg.norm(concat)
        if norm > 0:
            concat = concat / norm

        return FusedRepresentation(
            vector=concat,
            modality_contributions=contributions,
            quality_score=min(s.confidence for s in self._streams.values()),
        )

    def _fuse_attention_weighted(self) -> FusedRepresentation:
        """Attention-weighted fusion across modalities."""
        embeddings: List[np.ndarray] = []
        weights: List[float] = []
        contributions: Dict[str, float] = {}

        for modality, stream in self._streams.items():
            if stream.embedding is not None:
                embeddings.append(stream.embedding)
                w = stream.confidence
                weights.append(w)

        if not embeddings:
            return FusedRepresentation(
                vector=np.zeros(self.config.output_dim),
                quality_score=0.0,
            )

        # Softmax weights
        weights_arr = np.array(weights)
        weights_arr = np.exp(weights_arr) / np.sum(np.exp(weights_arr))

        fused = np.zeros(self.config.output_dim)
        for emb, w, (modality, _) in zip(
            embeddings, weights_arr, self._streams.items()
        ):
            fused += w * emb
            contributions[modality.value] = float(w)

        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return FusedRepresentation(
            vector=fused,
            modality_contributions=contributions,
            attention_map=weights_arr,
            quality_score=float(np.mean([s.confidence for s in self._streams.values()])),
        )

    def _fuse_spectral_anchored(self) -> FusedRepresentation:
        """Spectral-anchored fusion — spectral is the primary modality."""
        contributions: Dict[str, float] = {}

        # Start with spectral embedding if available
        spectral_stream = self._streams.get(Modality.SPECTRAL)
        if spectral_stream is not None and spectral_stream.embedding is not None:
            base = spectral_stream.embedding * self.config.spectral_weight
            contributions[Modality.SPECTRAL.value] = self.config.spectral_weight
        else:
            base = np.zeros(self.config.output_dim)

        # Add other modalities with remaining weight
        other_streams = {
            k: v for k, v in self._streams.items()
            if k != Modality.SPECTRAL and v.embedding is not None
        }

        if other_streams:
            remaining_weight = 1.0 - self.config.spectral_weight
            per_modal_weight = remaining_weight / len(other_streams)

            for modality, stream in other_streams.items():
                base += stream.embedding * per_modal_weight * stream.confidence
                contributions[modality.value] = float(
                    per_modal_weight * stream.confidence
                )

        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm

        return FusedRepresentation(
            vector=base,
            modality_contributions=contributions,
            quality_score=float(
                np.mean([s.confidence for s in self._streams.values()])
            ),
        )

    def _fuse_gated(self) -> FusedRepresentation:
        """Gated fusion with learned-like gating per modality."""
        embeddings: List[np.ndarray] = []
        gates: List[float] = []
        contributions: Dict[str, float] = {}

        for modality, stream in self._streams.items():
            if stream.embedding is not None:
                embeddings.append(stream.embedding)
                # Gate based on signal energy and confidence
                energy = float(np.mean(stream.embedding ** 2))
                gate = stream.confidence * (1.0 / (1.0 + np.exp(-energy)))
                gates.append(gate)

        if not embeddings:
            return FusedRepresentation(
                vector=np.zeros(self.config.output_dim),
                quality_score=0.0,
            )

        # Normalize gates
        total_gate = sum(gates)
        if total_gate > 0:
            gates = [g / total_gate for g in gates]

        fused = np.zeros(self.config.output_dim)
        for emb, g, (modality, _) in zip(
            embeddings, gates, self._streams.items()
        ):
            fused += g * emb
            contributions[modality.value] = float(g)

        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return FusedRepresentation(
            vector=fused,
            modality_contributions=contributions,
            quality_score=float(np.mean([s.confidence for s in self._streams.values()])),
        )

    @property
    def active_modalities(self) -> List[Modality]:
        """List of currently active modalities."""
        return list(self._streams.keys())

    @property
    def fusion_count(self) -> int:
        """Total number of fusion operations performed."""
        return self._fusion_count

    def clear(self) -> None:
        """Clear all current modality streams."""
        self._streams.clear()
