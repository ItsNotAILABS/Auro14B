"""NeuroCores — Spectral Neural Processing Cores.

Provides specialized neural processing units for spectral intelligence.
Each NeuroCore is a self-contained processing unit that combines
attention analysis, memory integration (via TAURUS), and multi-scale
spectral reasoning into a unified execution model.

NeuroCores enable:
- Spectral understanding through attention-weighted embeddings
- Long-range dependency capture across frequency bands
- Multi-scale simultaneous analysis at multiple resolutions
- Interpretable attention maps for scientific validation
- Foundation model potential through composable core architecture
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.cognitive.taurus_memory import (
    TaurusMemoryStore,
    TaurusWorkingMemory,
    RetrievalResult,
)


@dataclass
class NeuroCoreConfig:
    """Configuration for a NeuroCore processing unit.

    Args:
        core_id: Unique identifier for this core.
        d_model: Internal representation dimension.
        n_attention_heads: Number of parallel attention heads.
        memory_capacity: Long-term memory capacity.
        working_memory_slots: Working memory slot count.
        attention_temperature: Softmax temperature for attention.
        multi_scale_levels: Number of frequency resolution levels.
        enable_cross_band: Enable cross-band relationship detection.
        enable_harmonics: Enable harmonic resonance detection.
    """

    core_id: str = "core_0"
    d_model: int = 128
    n_attention_heads: int = 8
    memory_capacity: int = 500
    working_memory_slots: int = 7
    attention_temperature: float = 1.0
    multi_scale_levels: int = 4
    enable_cross_band: bool = True
    enable_harmonics: bool = True


@dataclass
class CoreProcessingResult:
    """Result from a NeuroCore processing operation.

    Args:
        embedding: Processed spectral embedding.
        attention_map: Attention weight distribution.
        attention_analysis: Interpretability metrics.
        multi_scale_features: Features at each resolution level.
        cross_band_scores: Cross-band relationship scores.
        harmonic_peaks: Detected harmonic frequencies.
        memory_matches: Related memories from TAURUS.
        metadata: Processing metadata.
    """

    embedding: np.ndarray
    attention_map: np.ndarray
    attention_analysis: Dict[str, float]
    multi_scale_features: List[np.ndarray]
    cross_band_scores: Optional[np.ndarray] = None
    harmonic_peaks: Optional[List[float]] = None
    memory_matches: List[RetrievalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpectralNeuroCore:
    """A neural processing core for spectral intelligence.

    Combines multi-head attention, multi-scale analysis, TAURUS memory,
    and cross-band dependency detection into a unified spectral
    processing unit.

    Args:
        config: NeuroCore configuration.
    """

    def __init__(self, config: Optional[NeuroCoreConfig] = None) -> None:
        self.config = config or NeuroCoreConfig()

        # Attention weights per head
        self._attention_weights = np.random.randn(
            self.config.n_attention_heads, self.config.d_model
        ) * 0.02

        # Multi-scale projection matrices
        self._scale_projections = [
            np.random.randn(self.config.d_model, self.config.d_model) * 0.02
            for _ in range(self.config.multi_scale_levels)
        ]

        # TAURUS memory integration
        self._long_term_memory = TaurusMemoryStore(
            capacity=self.config.memory_capacity,
            attention_temperature=self.config.attention_temperature,
        )
        self._working_memory = TaurusWorkingMemory(
            capacity=self.config.working_memory_slots,
            long_term_store=self._long_term_memory,
        )

        # Processing state
        self._processing_count = 0
        self._attention_history: List[np.ndarray] = []

    def process(
        self,
        spectrum: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        store_in_memory: bool = True,
    ) -> CoreProcessingResult:
        """Process a spectrum through the full NeuroCore pipeline.

        Steps:
        1. Multi-scale feature extraction at multiple resolutions
        2. Multi-head attention computation
        3. Cross-band relationship detection
        4. Harmonic peak identification
        5. TAURUS memory storage and retrieval
        6. Attention analysis for interpretability

        Args:
            spectrum: Input spectral data (any length).
            context: Optional contextual metadata.
            store_in_memory: Whether to store in TAURUS.

        Returns:
            CoreProcessingResult with all computed features.
        """
        self._processing_count += 1
        spectrum = np.atleast_1d(spectrum).flatten().astype(float)

        # Normalize to d_model length for processing
        if len(spectrum) != self.config.d_model:
            processed = np.interp(
                np.linspace(0, 1, self.config.d_model),
                np.linspace(0, 1, len(spectrum)),
                spectrum,
            )
        else:
            processed = spectrum.copy()

        # 1. Multi-scale feature extraction
        multi_scale_features = self._extract_multi_scale(processed)

        # 2. Multi-head attention
        attention_map, attended_output = self._compute_attention(processed)
        self._attention_history.append(attention_map)

        # 3. Cross-band relationships
        cross_band_scores = None
        if self.config.enable_cross_band:
            cross_band_scores = self._detect_cross_band(processed)

        # 4. Harmonic detection
        harmonic_peaks = None
        if self.config.enable_harmonics:
            harmonic_peaks = self._detect_harmonics(spectrum)

        # 5. TAURUS memory
        memory_matches: List[RetrievalResult] = []
        if store_in_memory:
            self._working_memory.hold(
                embedding=attended_output,
                context=context or {},
                importance=float(np.max(attention_map)),
                semantic_tag=context.get("tag", "") if context else "",
            )
            self._long_term_memory.store(
                embedding=attended_output,
                context=context or {},
                importance=float(np.max(attention_map)),
                semantic_tag=context.get("tag", "") if context else "",
            )

        # Retrieve related memories
        memory_matches = self._long_term_memory.retrieve(
            attended_output, top_k=3
        )

        # 6. Attention analysis
        attention_analysis = self._analyze_attention(attention_map)

        return CoreProcessingResult(
            embedding=attended_output,
            attention_map=attention_map,
            attention_analysis=attention_analysis,
            multi_scale_features=multi_scale_features,
            cross_band_scores=cross_band_scores,
            harmonic_peaks=harmonic_peaks,
            memory_matches=memory_matches,
            metadata={
                "core_id": self.config.core_id,
                "processing_id": self._processing_count,
                "spectrum_length": len(spectrum),
                "d_model": self.config.d_model,
            },
        )

    def _extract_multi_scale(self, spectrum: np.ndarray) -> List[np.ndarray]:
        """Extract features at multiple frequency resolutions."""
        features = []
        for level in range(self.config.multi_scale_levels):
            # Downsample at each scale
            scale_factor = 2 ** level
            downsampled = spectrum[::scale_factor]
            # Pad or truncate to d_model
            if len(downsampled) < self.config.d_model:
                padded = np.zeros(self.config.d_model)
                padded[: len(downsampled)] = downsampled
                feature = padded
            else:
                feature = downsampled[: self.config.d_model]

            # Apply scale-specific projection
            projected = feature @ self._scale_projections[level]
            features.append(projected)

        return features

    def _compute_attention(
        self, spectrum: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute multi-head attention over the spectrum.

        Returns:
            Tuple of (attention_map, attended_output).
        """
        n_heads = self.config.n_attention_heads
        head_outputs = np.zeros((n_heads, self.config.d_model))
        attention_scores = np.zeros((n_heads, self.config.d_model))

        for h in range(n_heads):
            # Score each dimension
            scores = spectrum * self._attention_weights[h]
            # Softmax
            exp_scores = np.exp(scores - np.max(scores))
            attn = exp_scores / (np.sum(exp_scores) + 1e-12)
            attention_scores[h] = attn
            # Apply attention
            head_outputs[h] = spectrum * attn * self.config.d_model

        # Average across heads
        attention_map = np.mean(attention_scores, axis=0)
        attended_output = np.mean(head_outputs, axis=0)

        return attention_map, attended_output

    def _detect_cross_band(self, spectrum: np.ndarray) -> np.ndarray:
        """Detect cross-band relationships (correlations between frequency regions)."""
        n_bands = min(8, self.config.d_model // 16)
        band_size = self.config.d_model // n_bands
        bands = [
            spectrum[i * band_size: (i + 1) * band_size]
            for i in range(n_bands)
        ]

        # Cross-correlation matrix
        cross_scores = np.zeros((n_bands, n_bands))
        for i in range(n_bands):
            for j in range(n_bands):
                norm_i = np.linalg.norm(bands[i]) + 1e-12
                norm_j = np.linalg.norm(bands[j]) + 1e-12
                cross_scores[i, j] = float(
                    np.dot(bands[i], bands[j]) / (norm_i * norm_j)
                )

        return cross_scores

    def _detect_harmonics(self, spectrum: np.ndarray) -> List[float]:
        """Detect harmonic peaks in the spectrum."""
        # Find peaks using simple local maximum detection
        peaks = []
        padded = np.concatenate([[0], spectrum, [0]])
        for i in range(1, len(padded) - 1):
            if padded[i] > padded[i - 1] and padded[i] > padded[i + 1]:
                if padded[i] > np.mean(spectrum) + np.std(spectrum):
                    peaks.append(float(i - 1))  # Original index

        # Filter for harmonic relationships (integer ratios)
        harmonics = []
        if peaks:
            fundamental = peaks[0] if peaks[0] > 0 else 1.0
            for p in peaks:
                ratio = p / fundamental
                if abs(ratio - round(ratio)) < 0.1:
                    harmonics.append(p)

        return harmonics[:10]  # Limit to top 10

    def _analyze_attention(self, attention_map: np.ndarray) -> Dict[str, float]:
        """Compute attention interpretability metrics.

        Returns:
            Dictionary with:
            - attention_entropy: Distribution spread of attention.
            - maximum_attention: Peak attention value.
            - attention_sparsity: Fraction of near-zero weights.
            - focus_ratio: Ratio of top-10% to bottom-90% attention.
        """
        # Normalize
        total = np.sum(attention_map) + 1e-12
        probs = attention_map / total

        # Entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        # Max attention
        max_attn = float(np.max(attention_map))
        # Sparsity
        sparsity = float(np.mean(attention_map < 0.01 * max_attn))
        # Focus ratio
        sorted_attn = np.sort(attention_map)[::-1]
        top_10_idx = max(1, len(sorted_attn) // 10)
        top_sum = np.sum(sorted_attn[:top_10_idx])
        bottom_sum = np.sum(sorted_attn[top_10_idx:]) + 1e-12
        focus_ratio = float(top_sum / bottom_sum)

        return {
            "attention_entropy": entropy,
            "maximum_attention": max_attn,
            "attention_sparsity": sparsity,
            "focus_ratio": focus_ratio,
        }

    def get_attention_analysis(self) -> Dict[str, Any]:
        """Get comprehensive attention analysis across all processed spectra.

        Provides interpretability through:
        - Attention entropy: How distributed vs. focused the attention is
        - Maximum attention: Strength of the strongest attended-to token
        - Attention sparsity: Fraction of near-zero attention weights

        Returns:
            Dictionary with aggregate attention metrics.
        """
        if not self._attention_history:
            return {
                "n_processed": 0,
                "mean_entropy": 0.0,
                "mean_max_attention": 0.0,
                "mean_sparsity": 0.0,
                "memory_analysis": self._long_term_memory.get_attention_analysis(),
            }

        analyses = [self._analyze_attention(a) for a in self._attention_history]

        return {
            "n_processed": self._processing_count,
            "mean_entropy": float(np.mean([a["attention_entropy"] for a in analyses])),
            "mean_max_attention": float(np.mean([a["maximum_attention"] for a in analyses])),
            "mean_sparsity": float(np.mean([a["attention_sparsity"] for a in analyses])),
            "focus_ratio_trend": [a["focus_ratio"] for a in analyses[-10:]],
            "memory_analysis": self._long_term_memory.get_attention_analysis(),
        }

    @property
    def memory_store(self) -> TaurusMemoryStore:
        """Access the TAURUS long-term memory store."""
        return self._long_term_memory

    @property
    def working_memory(self) -> TaurusWorkingMemory:
        """Access the working memory buffer."""
        return self._working_memory

    @property
    def processing_count(self) -> int:
        """Total number of spectra processed by this core."""
        return self._processing_count


class NeuroCoreCluster:
    """A cluster of NeuroCores for parallel multi-domain spectral processing.

    Orchestrates multiple NeuroCores that specialize in different
    aspects of spectral analysis (e.g., low-frequency, high-frequency,
    transient detection, harmonic analysis).

    Args:
        n_cores: Number of cores in the cluster.
        config: Base configuration (core_id will be overridden per core).
    """

    def __init__(
        self,
        n_cores: int = 4,
        config: Optional[NeuroCoreConfig] = None,
    ) -> None:
        base_config = config or NeuroCoreConfig()
        self.cores: List[SpectralNeuroCore] = []

        for i in range(n_cores):
            core_config = NeuroCoreConfig(
                core_id=f"core_{i}",
                d_model=base_config.d_model,
                n_attention_heads=base_config.n_attention_heads,
                memory_capacity=base_config.memory_capacity // n_cores,
                working_memory_slots=base_config.working_memory_slots,
                attention_temperature=base_config.attention_temperature,
                multi_scale_levels=base_config.multi_scale_levels,
                enable_cross_band=base_config.enable_cross_band,
                enable_harmonics=base_config.enable_harmonics,
            )
            self.cores.append(SpectralNeuroCore(core_config))

    def process_distributed(
        self,
        spectrum: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CoreProcessingResult]:
        """Process a spectrum across all cores in the cluster.

        Each core processes the full spectrum but focuses on different
        aspects due to random initialization of attention weights.

        Args:
            spectrum: Input spectral data.
            context: Optional contextual metadata.

        Returns:
            List of results, one from each core.
        """
        results = []
        for core in self.cores:
            result = core.process(spectrum, context=context)
            results.append(result)
        return results

    def get_ensemble_embedding(self, spectrum: np.ndarray) -> np.ndarray:
        """Get an ensemble embedding by averaging across all cores.

        Args:
            spectrum: Input spectral data.

        Returns:
            Averaged embedding vector.
        """
        results = self.process_distributed(spectrum)
        embeddings = np.vstack([r.embedding for r in results])
        return np.mean(embeddings, axis=0)

    def get_cluster_attention_analysis(self) -> Dict[str, Any]:
        """Get attention analysis aggregated across all cores.

        Returns:
            Dictionary with per-core and cluster-wide metrics.
        """
        per_core = {}
        for core in self.cores:
            per_core[core.config.core_id] = core.get_attention_analysis()

        # Aggregate
        all_entropies = [
            v["mean_entropy"] for v in per_core.values() if v["n_processed"] > 0
        ]

        return {
            "n_cores": len(self.cores),
            "per_core_analysis": per_core,
            "cluster_mean_entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "total_processed": sum(c.processing_count for c in self.cores),
        }

    @property
    def total_memory_size(self) -> int:
        """Total memory traces across all cores."""
        return sum(c.memory_store.size for c in self.cores)
