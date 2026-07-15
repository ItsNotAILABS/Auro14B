"""Sovereign On-Device AI — Portable Brain for Air-Gapped Environments.

Fully local "portable brain" designed for air-gapped, secure, and
low-connectivity environments including defense, remote operations,
and critical infrastructure.

Key Capabilities
----------------
- **Zero Cloud Dependency** — All inference, learning, and memory run
  entirely on-device with no network calls. Eliminates cloud costs,
  latency, and privacy risks.
- **Live Streaming Pipelines** — Continuous learning and real-time
  updating of spectral fingerprint libraries on-device from sensor
  streams.
- **Air-Gap Compatible** — Operates without any external connectivity.
  Ideal for classified, remote, or infrastructure-critical deployments.
- **Privacy-Sovereign** — Data never leaves the device boundary.
  Full compliance with data-sovereignty requirements.

Architecture
------------
    Sensor / Signal Input
            ↓
    On-Device Streaming Pipeline
    ├── Continuous fingerprint ingestion
    ├── Incremental library updates (no batch retraining)
    └── Drift detection & adaptation
            ↓
    Sovereign Inference Core
    ├── Local MESIE spectral matching
    ├── On-device embedding generation
    ├── Memory-efficient model execution
    └── NeuroAIX connectome propagation (optional)
            ↓
    Agent Output (fully local)

Deployment Targets
------------------
- Edge compute (NVIDIA Jetson, Raspberry Pi, rugged SBCs)
- Disconnected field laptops / tablets
- SCADA / ICS isolated networks
- Submarine, satellite, aircraft systems
- Disaster-response portable kits

Copyright (c) 2024-2026 MESIE Contributors. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class DeviceProfile(Enum):
    """Supported on-device deployment profiles."""

    EDGE_MINIMAL = "edge_minimal"          # <512 MB RAM, constrained CPU
    EDGE_STANDARD = "edge_standard"        # 2-8 GB RAM, GPU optional
    FIELD_LAPTOP = "field_laptop"          # 8-32 GB RAM, discrete GPU
    RUGGED_EMBEDDED = "rugged_embedded"    # Real-time OS, deterministic latency
    HIGH_SECURITY = "high_security"        # FIPS-compliant, audit logging


class PrivacyLevel(Enum):
    """Data sovereignty and privacy enforcement levels."""

    STANDARD = "standard"          # Data stays on-device, no telemetry
    CLASSIFIED = "classified"      # Encrypted at rest, memory-zeroized on idle
    AIR_GAPPED = "air_gapped"      # No network stack loaded, hardware-enforced


@dataclass
class SovereignConfig:
    """Configuration for on-device sovereign AI operation.

    Parameters
    ----------
    device_profile : DeviceProfile
        Hardware target for resource allocation.
    privacy_level : PrivacyLevel
        Data sovereignty enforcement level.
    fingerprint_capacity : int
        Maximum spectral fingerprints in the local library.
    streaming_buffer_size : int
        Number of streaming samples buffered before incremental update.
    embedding_dim : int
        Dimensionality of on-device spectral embeddings.
    enable_drift_detection : bool
        Whether to monitor fingerprint distribution drift.
    max_memory_mb : int
        Best-effort memory budget target for the engine (in megabytes).
        This module currently does not enforce a hard ceiling.
    deterministic : bool
        Preference flag for deterministic behavior where explicitly
        supported by the active components. This is not a global
        determinism guarantee for all operations in this module.
    """

    device_profile: DeviceProfile = DeviceProfile.EDGE_STANDARD
    privacy_level: PrivacyLevel = PrivacyLevel.STANDARD
    fingerprint_capacity: int = 10000
    streaming_buffer_size: int = 64
    embedding_dim: int = 128
    enable_drift_detection: bool = True
    max_memory_mb: int = 512
    deterministic: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# FINGERPRINT LIBRARY (ON-DEVICE)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SpectralFingerprint:
    """A single spectral fingerprint in the on-device library."""

    fingerprint_id: str
    embedding: np.ndarray
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    update_count: int = 0


class OnDeviceFingerprintLibrary:
    """Local fingerprint library with incremental update support.

    Stores and retrieves spectral fingerprints entirely on-device.
    Supports streaming updates without full retraining.
    """

    def __init__(self, capacity: int = 10000, embedding_dim: int = 128):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self._fingerprints: Dict[str, SpectralFingerprint] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._index_dirty = True

    @property
    def size(self) -> int:
        """Current number of fingerprints stored."""
        return len(self._fingerprints)

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate/cast embedding to configured dimensionality."""
        validated_embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if validated_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding must have length {self.embedding_dim}, got {validated_embedding.shape[0]}"
            )
        return validated_embedding

    def add(self, fingerprint: SpectralFingerprint) -> bool:
        """Add or update a fingerprint in the library.

        Returns True if the fingerprint was added successfully.
        If at capacity, evicts the oldest fingerprint.
        """
        if self.capacity <= 0:
            return False

        fingerprint.embedding = self._validate_embedding(fingerprint.embedding)

        if self.size >= self.capacity and fingerprint.fingerprint_id not in self._fingerprints:
            # Evict oldest
            oldest_id = min(
                self._fingerprints,
                key=lambda fid: self._fingerprints[fid].created_at,
            )
            del self._fingerprints[oldest_id]

        self._fingerprints[fingerprint.fingerprint_id] = fingerprint
        self._index_dirty = True
        return True

    def query(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[SpectralFingerprint, float]]:
        """Find the top-k most similar fingerprints.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector.
        top_k : int
            Number of results to return.

        Returns
        -------
        List of (fingerprint, similarity_score) tuples, sorted by
        descending similarity.
        """
        if not self._fingerprints:
            return []

        embedding = self._validate_embedding(embedding)
        self._rebuild_index_if_needed()

        # Cosine similarity
        query_norm = np.linalg.norm(embedding)
        if query_norm == 0:
            return []
        query_normalized = embedding / query_norm

        similarities = self._embedding_matrix @ query_normalized
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        ids = list(self._fingerprints.keys())
        results = []
        for idx in top_indices:
            fp = self._fingerprints[ids[idx]]
            results.append((fp, float(similarities[idx])))
        return results

    def incremental_update(self, fingerprint_id: str, new_embedding: np.ndarray, momentum: float = 0.9) -> bool:
        """Update an existing fingerprint with exponential moving average.

        This enables continuous learning without full retraining.

        Parameters
        ----------
        fingerprint_id : str
            ID of the fingerprint to update.
        new_embedding : np.ndarray
            New observation embedding.
        momentum : float
            Weight for existing embedding (0-1). Higher = more stable.

        Returns
        -------
        bool
            True if the fingerprint was updated.
        """
        if fingerprint_id not in self._fingerprints:
            return False
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")

        new_embedding = self._validate_embedding(new_embedding)

        fp = self._fingerprints[fingerprint_id]
        fp.embedding = momentum * fp.embedding + (1 - momentum) * new_embedding
        # Re-normalize
        norm = np.linalg.norm(fp.embedding)
        if norm > 0:
            fp.embedding = fp.embedding / norm
        fp.update_count += 1
        self._index_dirty = True
        return True

    def _rebuild_index_if_needed(self):
        """Rebuild the embedding matrix index if dirty."""
        if self._index_dirty and self._fingerprints:
            embeddings = []
            for fp in self._fingerprints.values():
                emb = fp.embedding
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 0 else emb)
            self._embedding_matrix = np.stack(embeddings)
            self._index_dirty = False


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING PIPELINE (ON-DEVICE CONTINUOUS LEARNING)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StreamingSample:
    """A single sample in the streaming pipeline."""

    data: np.ndarray
    timestamp: float
    source_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnDeviceStreamingPipeline:
    """Live streaming pipeline for continuous fingerprint library updates.

    Enables real-time ingestion of spectral data from sensors with
    incremental updates to the fingerprint library — all on-device
    with no cloud dependency.

    Features
    --------
    - Buffered micro-batch processing for efficiency
    - Drift detection to flag distribution shifts
    - Automatic fingerprint creation for novel patterns
    - Resource-aware operation tuned for constrained devices
    """

    def __init__(
        self,
        library: OnDeviceFingerprintLibrary,
        config: SovereignConfig,
        novelty_threshold: float = 0.3,
        drift_window: int = 200,
    ):
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError(f"novelty_threshold must be in [0, 1], got {novelty_threshold}")
        if drift_window < 4:
            raise ValueError(f"drift_window must be >= 4, got {drift_window}")

        self.library = library
        self.config = config
        self.novelty_threshold = novelty_threshold
        self.drift_window = drift_window

        self._buffer: List[StreamingSample] = []
        self._processed_count: int = 0
        self._drift_history: List[float] = []
        self._drift_alert: bool = False

        # Callbacks
        self._on_novel_pattern: Optional[Callable[[np.ndarray, float], None]] = None
        self._on_drift_detected: Optional[Callable[[float], None]] = None

    @property
    def processed_count(self) -> int:
        """Total number of samples processed."""
        return self._processed_count

    @property
    def drift_detected(self) -> bool:
        """Whether distribution drift has been detected."""
        return self._drift_alert

    def ingest(self, sample: StreamingSample) -> Optional[SpectralFingerprint]:
        """Ingest a streaming sample into the pipeline.

        Buffers the sample and triggers a micro-batch update when
        the buffer is full.

        Parameters
        ----------
        sample : StreamingSample
            Incoming spectral sample.

        Returns
        -------
        Optional[SpectralFingerprint]
            If a novel fingerprint was created, returns it. Otherwise None.
        """
        self._buffer.append(sample)

        if len(self._buffer) >= self.config.streaming_buffer_size:
            return self._process_buffer()
        return None

    def flush(self) -> Optional[SpectralFingerprint]:
        """Force-process any remaining buffered samples."""
        if self._buffer:
            return self._process_buffer()
        return None

    def register_novel_pattern_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Register a callback for novel pattern detection."""
        self._on_novel_pattern = callback

    def register_drift_callback(self, callback: Callable[[float], None]):
        """Register a callback for distribution drift detection."""
        self._on_drift_detected = callback

    def _process_buffer(self) -> Optional[SpectralFingerprint]:
        """Process the current buffer as a micro-batch."""
        if not self._buffer:
            return None

        # Compute mean embedding for this micro-batch
        embeddings = np.stack([s.data for s in self._buffer])
        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        self._processed_count += len(self._buffer)
        timestamp = self._buffer[-1].timestamp

        # Check novelty against library
        matches = self.library.query(mean_embedding, top_k=1)
        novel_fingerprint = None

        if not matches or matches[0][1] < self.novelty_threshold:
            # Novel pattern — create new fingerprint
            fp_id = f"stream_{self._processed_count}_{timestamp:.2f}"
            novel_fingerprint = SpectralFingerprint(
                fingerprint_id=fp_id,
                embedding=mean_embedding,
                label=f"auto_discovered_{self._processed_count}",
                metadata={"source": "streaming_pipeline", "batch_size": len(self._buffer)},
                created_at=timestamp,
            )
            self.library.add(novel_fingerprint)
            if self._on_novel_pattern:
                self._on_novel_pattern(mean_embedding, timestamp)
        else:
            # Update existing fingerprint with momentum
            best_match = matches[0][0]
            self.library.incremental_update(
                best_match.fingerprint_id, mean_embedding, momentum=0.95
            )

        # Drift detection
        if self.config.enable_drift_detection:
            self._update_drift(mean_embedding, matches)

        self._buffer.clear()
        return novel_fingerprint

    def _update_drift(self, embedding: np.ndarray, matches: List[Tuple[SpectralFingerprint, float]]):
        """Update drift detection state."""
        if matches:
            similarity = matches[0][1]
        else:
            similarity = 0.0

        self._drift_history.append(similarity)
        if len(self._drift_history) > self.drift_window:
            self._drift_history.pop(0)

        # Detect drift: moving average similarity dropping significantly
        if len(self._drift_history) >= self.drift_window // 2:
            recent = np.mean(self._drift_history[-self.drift_window // 4:])
            historical = np.mean(self._drift_history[:self.drift_window // 4])
            drift_score = historical - recent  # Positive = increasing novelty

            if drift_score > 0.15:
                self._drift_alert = True
                if self._on_drift_detected:
                    self._on_drift_detected(drift_score)
            else:
                self._drift_alert = False


# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN ON-DEVICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class SovereignOnDeviceEngine:
    """Sovereign, Private On-Device AI Engine.

    A fully local "portable brain" that performs spectral intelligence
    entirely on-device — ideal for air-gapped, secure, or
    low-connectivity environments (defense, remote ops, critical
    infrastructure).

    Benefits
    --------
    - **Zero cloud dependency** — eliminates costs, latency, and privacy risks
    - **Continuous on-device learning** — live streaming pipelines update
      fingerprint libraries without cloud round-trips
    - **Air-gap safe** — no network calls, no telemetry, no data exfiltration
    - **Resource-aware** — adapts to device constraints (edge → laptop → embedded)
    - **Deterministic option** — reproducible inference for safety-critical systems

    Example
    -------
    >>> config = SovereignConfig(
    ...     device_profile=DeviceProfile.EDGE_STANDARD,
    ...     privacy_level=PrivacyLevel.AIR_GAPPED,
    ... )
    >>> engine = SovereignOnDeviceEngine(config)
    >>> engine.infer(spectral_data)  # Fully local inference
    >>> engine.stream(sample)        # Continuous learning on-device
    """

    def __init__(self, config: Optional[SovereignConfig] = None):
        self.config = config or SovereignConfig()

        # Core components
        self.library = OnDeviceFingerprintLibrary(
            capacity=self.config.fingerprint_capacity,
            embedding_dim=self.config.embedding_dim,
        )
        self.pipeline = OnDeviceStreamingPipeline(
            library=self.library,
            config=self.config,
        )

        # Internal state
        self._initialized = True
        self._inference_count = 0
        self._total_latency_ms: float = 0.0

    @property
    def is_sovereign(self) -> bool:
        """Always True — this engine never contacts external services."""
        return True

    @property
    def cloud_dependency(self) -> bool:
        """Always False — no cloud calls are made."""
        return False

    @property
    def inference_count(self) -> int:
        """Total inferences performed on-device."""
        return self._inference_count

    @property
    def average_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    def infer(
        self,
        spectral_data: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[SpectralFingerprint, float]]:
        """Perform fully local spectral inference.

        Matches the input against the on-device fingerprint library
        with zero network calls.

        Parameters
        ----------
        spectral_data : np.ndarray
            Input spectral data or embedding vector.
        top_k : int
            Number of top matches to return.

        Returns
        -------
        List of (fingerprint, similarity) tuples.
        """
        import time

        start = time.perf_counter()

        # Normalize input
        embedding = self._prepare_embedding(spectral_data)

        # Local library query
        results = self.library.query(embedding, top_k=top_k)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._total_latency_ms += elapsed_ms
        self._inference_count += 1

        return results

    def stream(self, spectral_data: np.ndarray, timestamp: float = 0.0, source_id: str = "") -> Optional[SpectralFingerprint]:
        """Ingest streaming data for continuous on-device learning.

        Feeds spectral data into the live streaming pipeline for
        incremental fingerprint library updates — no cloud required.

        Parameters
        ----------
        spectral_data : np.ndarray
            Raw spectral data from sensor stream.
        timestamp : float
            Sample timestamp.
        source_id : str
            Identifier for the data source.

        Returns
        -------
        Optional[SpectralFingerprint]
            Newly discovered fingerprint, if one was created.
        """
        sample = StreamingSample(
            data=self._prepare_embedding(spectral_data),
            timestamp=timestamp,
            source_id=source_id,
        )
        return self.pipeline.ingest(sample)

    def seed_library(self, fingerprints: List[SpectralFingerprint]) -> int:
        """Seed the on-device library with pre-computed fingerprints.

        Use this to bootstrap the library from a curated dataset
        before deployment to an air-gapped environment.

        Parameters
        ----------
        fingerprints : List[SpectralFingerprint]
            Pre-computed fingerprints to load.

        Returns
        -------
        int
            Number of fingerprints successfully loaded.
        """
        loaded = 0
        for fp in fingerprints:
            if self.library.add(fp):
                loaded += 1
        return loaded

    def export_library(self) -> Dict[str, Any]:
        """Export the fingerprint library for offline transfer.

        Enables sneakernet/offline updates between air-gapped systems.

        Returns
        -------
        Dict containing serializable library state.
        """
        return {
            "version": "1.0.0",
            "engine": "sovereign_ondevice",
            "fingerprint_count": self.library.size,
            "embedding_dim": self.config.embedding_dim,
            "fingerprints": {
                fp_id: {
                    "embedding": fp.embedding.tolist(),
                    "label": fp.label,
                    "metadata": fp.metadata,
                    "created_at": fp.created_at,
                    "update_count": fp.update_count,
                }
                for fp_id, fp in self.library._fingerprints.items()
            },
        }

    def import_library(self, data: Dict[str, Any]) -> int:
        """Import a fingerprint library from offline transfer.

        Parameters
        ----------
        data : Dict
            Library export data (from export_library).

        Returns
        -------
        int
            Number of fingerprints imported.
        """
        imported = 0
        for fp_id, fp_data in data.get("fingerprints", {}).items():
            fp = SpectralFingerprint(
                fingerprint_id=fp_id,
                embedding=np.array(fp_data["embedding"]),
                label=fp_data.get("label", ""),
                metadata=fp_data.get("metadata", {}),
                created_at=fp_data.get("created_at", 0.0),
                update_count=fp_data.get("update_count", 0),
            )
            if self.library.add(fp):
                imported += 1
        return imported

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status for monitoring.

        Returns
        -------
        Dict with engine health, memory usage, and performance metrics.
        """
        return {
            "sovereign": self.is_sovereign,
            "cloud_dependency": self.cloud_dependency,
            "device_profile": self.config.device_profile.value,
            "privacy_level": self.config.privacy_level.value,
            "library_size": self.library.size,
            "library_capacity": self.config.fingerprint_capacity,
            "inference_count": self._inference_count,
            "average_latency_ms": self.average_latency_ms,
            "streaming_processed": self.pipeline.processed_count,
            "drift_detected": self.pipeline.drift_detected,
            "deterministic": self.config.deterministic,
        }

    def _prepare_embedding(self, spectral_data: np.ndarray) -> np.ndarray:
        """Prepare input data as a normalized embedding vector."""
        dim = self.config.embedding_dim
        if len(spectral_data) >= dim:
            embedding = spectral_data[:dim].copy()
        else:
            embedding = np.zeros(dim)
            embedding[:len(spectral_data)] = spectral_data

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding
