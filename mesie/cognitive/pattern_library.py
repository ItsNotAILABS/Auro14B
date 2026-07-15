"""Spectral Pattern Library and Template Matching.

Provides a comprehensive library of spectral patterns,
templates for matching, and pattern-based classification.

Key Components:
    - SpectralTemplate: Individual spectral pattern template
    - PatternLibrary: Collection of reference patterns
    - TemplateMatching: Match spectra against templates
    - PatternGenerator: Generate synthetic training patterns
    - SpectralFingerprinter: Create unique spectral fingerprints
    - PatternEvolution: Track pattern changes over time
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class MatchingMetric(Enum):
    """Distance metrics for template matching."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    CORRELATION = "correlation"
    DTW = "dtw"
    SPECTRAL_ANGLE = "spectral_angle"
    EARTH_MOVERS = "earth_movers"


class PatternCategory(Enum):
    """Categories of spectral patterns."""
    HARMONIC = "harmonic"
    NOISE = "noise"
    TRANSIENT = "transient"
    RESONANCE = "resonance"
    MODULATION = "modulation"
    DECAY = "decay"
    BROADBAND = "broadband"
    NARROWBAND = "narrowband"
    CHIRP = "chirp"
    IMPULSE = "impulse"


class EvolutionType(Enum):
    """Types of pattern evolution."""
    STABLE = "stable"
    DRIFTING = "drifting"
    DEGRADING = "degrading"
    EMERGING = "emerging"
    OSCILLATING = "oscillating"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SpectralTemplate:
    """A reference spectral pattern template.

    Args:
        template_id: Unique identifier.
        name: Human-readable name.
        pattern: Reference spectrum.
        category: Pattern category.
        tolerance: Matching tolerance.
        metadata: Additional info.
        created_at: Creation time.
    """
    template_id: str
    name: str
    pattern: np.ndarray
    category: PatternCategory = PatternCategory.HARMONIC
    tolerance: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        """Template length."""
        return len(self.pattern)

    @property
    def energy(self) -> float:
        """Template energy."""
        return float(np.sum(self.pattern ** 2))


@dataclass
class MatchResult:
    """Result of template matching.

    Args:
        template_id: Matched template.
        similarity: Similarity score (0-1).
        offset: Best alignment offset.
        scale_factor: Amplitude scale.
        residual: Matching residual.
    """
    template_id: str
    similarity: float
    offset: int = 0
    scale_factor: float = 1.0
    residual: float = 0.0


@dataclass
class SpectralFingerprint:
    """Unique spectral fingerprint.

    Args:
        fingerprint_id: Unique ID.
        features: Fingerprint feature vector.
        hash_code: Compact hash representation.
        source: Origin of fingerprint.
    """
    fingerprint_id: str
    features: np.ndarray
    hash_code: int = 0
    source: str = ""


@dataclass
class EvolutionSnapshot:
    """Snapshot of pattern state at a point in time.

    Args:
        timestamp: Snapshot time.
        pattern: Pattern state.
        metrics: Pattern metrics at this time.
    """
    timestamp: float
    pattern: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Pattern Library
# =============================================================================


class PatternLibrary:
    """Library of reference spectral patterns.

    Manages a collection of spectral templates for classification,
    monitoring, and reference lookup.

    Args:
        max_templates: Maximum number of templates.
        default_metric: Default matching metric.
    """

    def __init__(
        self,
        max_templates: int = 1000,
        default_metric: MatchingMetric = MatchingMetric.COSINE,
    ) -> None:
        self.max_templates = max_templates
        self.default_metric = default_metric
        self._templates: Dict[str, SpectralTemplate] = {}
        self._category_index: Dict[PatternCategory, List[str]] = {}

    def add_template(self, template: SpectralTemplate) -> None:
        """Add a template to the library.

        Args:
            template: SpectralTemplate to add.
        """
        if len(self._templates) >= self.max_templates:
            # Remove oldest
            oldest_id = min(self._templates, key=lambda k: self._templates[k].created_at)
            self.remove_template(oldest_id)

        self._templates[template.template_id] = template
        if template.category not in self._category_index:
            self._category_index[template.category] = []
        self._category_index[template.category].append(template.template_id)

    def remove_template(self, template_id: str) -> None:
        """Remove a template from the library."""
        if template_id in self._templates:
            cat = self._templates[template_id].category
            self._templates.pop(template_id)
            if cat in self._category_index:
                self._category_index[cat] = [
                    tid for tid in self._category_index[cat] if tid != template_id
                ]

    def get_template(self, template_id: str) -> Optional[SpectralTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def search(
        self,
        spectrum: np.ndarray,
        top_k: int = 5,
        category: Optional[PatternCategory] = None,
        metric: Optional[MatchingMetric] = None,
    ) -> List[MatchResult]:
        """Search library for matching templates.

        Args:
            spectrum: Query spectrum.
            top_k: Number of results.
            category: Optional category filter.
            metric: Matching metric.

        Returns:
            Sorted list of MatchResult.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        metric = metric or self.default_metric

        # Get candidate templates
        if category:
            candidate_ids = self._category_index.get(category, [])
        else:
            candidate_ids = list(self._templates.keys())

        results = []
        for tid in candidate_ids:
            template = self._templates[tid]
            similarity = self._compute_similarity(
                spectrum, template.pattern, metric
            )
            results.append(MatchResult(
                template_id=tid,
                similarity=similarity,
                scale_factor=self._compute_scale(spectrum, template.pattern),
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def get_by_category(self, category: PatternCategory) -> List[SpectralTemplate]:
        """Get all templates in a category."""
        ids = self._category_index.get(category, [])
        return [self._templates[tid] for tid in ids if tid in self._templates]

    def _compute_similarity(
        self,
        spectrum: np.ndarray,
        template: np.ndarray,
        metric: MatchingMetric,
    ) -> float:
        """Compute similarity between spectrum and template."""
        # Match lengths
        n = min(len(spectrum), len(template))
        s = spectrum[:n]
        t = template[:n]

        if metric == MatchingMetric.COSINE:
            dot = np.dot(s, t)
            norm_s = np.linalg.norm(s) + 1e-12
            norm_t = np.linalg.norm(t) + 1e-12
            return float(dot / (norm_s * norm_t))

        elif metric == MatchingMetric.CORRELATION:
            if np.std(s) < 1e-12 or np.std(t) < 1e-12:
                return 0.0
            return float(np.corrcoef(s, t)[0, 1])

        elif metric == MatchingMetric.EUCLIDEAN:
            dist = np.linalg.norm(s - t)
            max_dist = np.sqrt(n) * (np.max(np.abs(s)) + np.max(np.abs(t)))
            return float(1.0 - dist / (max_dist + 1e-12))

        elif metric == MatchingMetric.SPECTRAL_ANGLE:
            dot = np.dot(s, t)
            norms = np.linalg.norm(s) * np.linalg.norm(t) + 1e-12
            cos_angle = np.clip(dot / norms, -1, 1)
            angle = np.arccos(cos_angle)
            return float(1.0 - angle / np.pi)

        elif metric == MatchingMetric.DTW:
            return self._dtw_similarity(s, t)

        return 0.0

    def _dtw_similarity(self, s: np.ndarray, t: np.ndarray) -> float:
        """Simplified DTW-based similarity."""
        n, m = len(s), len(t)
        # Downsample for efficiency
        max_len = 100
        if n > max_len:
            s = s[np.linspace(0, n-1, max_len, dtype=int)]
            n = max_len
        if m > max_len:
            t = t[np.linspace(0, m-1, max_len, dtype=int)]
            m = max_len

        # DTW cost matrix
        cost = np.full((n + 1, m + 1), float("inf"))
        cost[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = (s[i-1] - t[j-1]) ** 2
                cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

        dtw_dist = np.sqrt(cost[n, m] / max(n, m))
        max_possible = np.sqrt(np.max(s ** 2) + np.max(t ** 2)) + 1e-12
        return float(max(0.0, 1.0 - dtw_dist / max_possible))

    def _compute_scale(self, spectrum: np.ndarray, template: np.ndarray) -> float:
        """Compute amplitude scale factor."""
        n = min(len(spectrum), len(template))
        t_norm = np.linalg.norm(template[:n])
        if t_norm < 1e-12:
            return 1.0
        return float(np.linalg.norm(spectrum[:n]) / t_norm)

    @property
    def n_templates(self) -> int:
        """Number of templates."""
        return len(self._templates)

    @property
    def categories(self) -> List[PatternCategory]:
        """Available categories."""
        return list(self._category_index.keys())


# =============================================================================
# Pattern Generator
# =============================================================================


class PatternGenerator:
    """Generate synthetic spectral patterns.

    Creates training patterns with controlled characteristics
    for testing, augmentation, and simulation.

    Args:
        length: Default pattern length.
        sample_rate: Sample rate for frequency calculations.
    """

    def __init__(
        self,
        length: int = 512,
        sample_rate: float = 44100.0,
    ) -> None:
        self.length = length
        self.sample_rate = sample_rate

    def generate(
        self,
        category: PatternCategory,
        n_patterns: int = 1,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Generate patterns of a specific category.

        Args:
            category: Pattern category.
            n_patterns: Number to generate.
            **kwargs: Category-specific parameters.

        Returns:
            List of generated patterns.
        """
        generators = {
            PatternCategory.HARMONIC: self._generate_harmonic,
            PatternCategory.NOISE: self._generate_noise,
            PatternCategory.TRANSIENT: self._generate_transient,
            PatternCategory.RESONANCE: self._generate_resonance,
            PatternCategory.MODULATION: self._generate_modulation,
            PatternCategory.DECAY: self._generate_decay,
            PatternCategory.BROADBAND: self._generate_broadband,
            PatternCategory.NARROWBAND: self._generate_narrowband,
            PatternCategory.CHIRP: self._generate_chirp,
            PatternCategory.IMPULSE: self._generate_impulse,
        }

        gen_fn = generators.get(category, self._generate_harmonic)
        return [gen_fn(**kwargs) for _ in range(n_patterns)]

    def _generate_harmonic(self, **kwargs: Any) -> np.ndarray:
        """Generate harmonic pattern."""
        fundamental = kwargs.get("fundamental", np.random.uniform(100, 1000))
        n_harmonics = kwargs.get("n_harmonics", np.random.randint(3, 8))
        t = np.arange(self.length) / self.sample_rate

        signal = np.zeros(self.length)
        for k in range(1, n_harmonics + 1):
            amp = 1.0 / k
            signal += amp * np.sin(2 * np.pi * fundamental * k * t)

        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_noise(self, **kwargs: Any) -> np.ndarray:
        """Generate noise pattern."""
        noise_type = kwargs.get("noise_type", "white")
        if noise_type == "pink":
            freqs = np.fft.rfftfreq(self.length, 1.0 / self.sample_rate)
            freqs[0] = 1e-10
            magnitudes = 1.0 / np.sqrt(freqs)
            phases = np.random.uniform(0, 2 * np.pi, len(freqs))
            spectrum = magnitudes * np.exp(1j * phases)
            signal = np.fft.irfft(spectrum, n=self.length)
        else:
            signal = np.random.randn(self.length)
        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_transient(self, **kwargs: Any) -> np.ndarray:
        """Generate transient pattern."""
        position = kwargs.get("position", np.random.randint(self.length // 4, 3 * self.length // 4))
        width = kwargs.get("width", np.random.randint(5, self.length // 10))
        freq = kwargs.get("frequency", np.random.uniform(500, 5000))

        t = np.arange(self.length) / self.sample_rate
        envelope = np.exp(-0.5 * ((np.arange(self.length) - position) / width) ** 2)
        signal = envelope * np.sin(2 * np.pi * freq * t)
        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_resonance(self, **kwargs: Any) -> np.ndarray:
        """Generate resonance pattern."""
        freq = kwargs.get("frequency", np.random.uniform(200, 2000))
        damping = kwargs.get("damping", np.random.uniform(0.001, 0.05))

        t = np.arange(self.length) / self.sample_rate
        signal = np.exp(-damping * 2 * np.pi * freq * t) * np.sin(2 * np.pi * freq * t)
        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_modulation(self, **kwargs: Any) -> np.ndarray:
        """Generate amplitude-modulated pattern."""
        carrier = kwargs.get("carrier", np.random.uniform(1000, 5000))
        modulator = kwargs.get("modulator", np.random.uniform(10, 100))
        depth = kwargs.get("depth", np.random.uniform(0.3, 0.9))

        t = np.arange(self.length) / self.sample_rate
        envelope = 1.0 + depth * np.sin(2 * np.pi * modulator * t)
        signal = envelope * np.sin(2 * np.pi * carrier * t)
        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_decay(self, **kwargs: Any) -> np.ndarray:
        """Generate exponential decay pattern."""
        freq = kwargs.get("frequency", np.random.uniform(200, 2000))
        decay_rate = kwargs.get("decay_rate", np.random.uniform(1, 20))

        t = np.arange(self.length) / self.sample_rate
        signal = np.exp(-decay_rate * t) * np.sin(2 * np.pi * freq * t)
        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_broadband(self, **kwargs: Any) -> np.ndarray:
        """Generate broadband excitation."""
        n_components = kwargs.get("n_components", 50)
        signal = np.zeros(self.length)
        t = np.arange(self.length) / self.sample_rate

        for _ in range(n_components):
            freq = np.random.uniform(20, self.sample_rate / 2)
            amp = np.random.uniform(0.1, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)

        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_narrowband(self, **kwargs: Any) -> np.ndarray:
        """Generate narrowband signal."""
        center_freq = kwargs.get("center_freq", np.random.uniform(500, 5000))
        bandwidth_hz = kwargs.get("bandwidth", np.random.uniform(10, 100))

        signal = np.zeros(self.length)
        t = np.arange(self.length) / self.sample_rate
        n_components = 10

        for i in range(n_components):
            freq = center_freq + np.random.uniform(-bandwidth_hz/2, bandwidth_hz/2)
            signal += np.sin(2 * np.pi * freq * t)

        return signal / (np.max(np.abs(signal)) + 1e-12)

    def _generate_chirp(self, **kwargs: Any) -> np.ndarray:
        """Generate chirp (frequency sweep)."""
        f_start = kwargs.get("f_start", np.random.uniform(100, 1000))
        f_end = kwargs.get("f_end", np.random.uniform(2000, 10000))

        t = np.arange(self.length) / self.sample_rate
        T = self.length / self.sample_rate
        phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t ** 2 / (2 * T))
        return np.sin(phase)

    def _generate_impulse(self, **kwargs: Any) -> np.ndarray:
        """Generate impulse pattern."""
        position = kwargs.get("position", self.length // 2)
        signal = np.zeros(self.length)
        signal[position] = 1.0

        # Add some ring-down
        freq = kwargs.get("ring_freq", np.random.uniform(1000, 5000))
        damping = kwargs.get("damping", 0.1)
        t = np.arange(self.length) / self.sample_rate
        ring = np.zeros(self.length)
        for i in range(position, self.length):
            dt = (i - position) / self.sample_rate
            ring[i] = np.exp(-damping * 2 * np.pi * freq * dt) * np.sin(2 * np.pi * freq * dt)

        signal += ring * 0.5
        return signal / (np.max(np.abs(signal)) + 1e-12)

    def augment(
        self,
        pattern: np.ndarray,
        n_augmented: int = 5,
        noise_level: float = 0.05,
        shift_range: int = 10,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> List[np.ndarray]:
        """Create augmented versions of a pattern.

        Args:
            pattern: Original pattern.
            n_augmented: Number of augmentations.
            noise_level: Noise amplitude.
            shift_range: Maximum circular shift.
            scale_range: Amplitude scale range.

        Returns:
            List of augmented patterns.
        """
        pattern = np.atleast_1d(pattern).flatten()
        augmented = []

        for _ in range(n_augmented):
            p = pattern.copy()

            # Random noise
            p += noise_level * np.random.randn(len(p))

            # Random shift
            shift = np.random.randint(-shift_range, shift_range + 1)
            p = np.roll(p, shift)

            # Random scale
            scale = np.random.uniform(*scale_range)
            p *= scale

            augmented.append(p)

        return augmented


# =============================================================================
# Spectral Fingerprinter
# =============================================================================


class SpectralFingerprinter:
    """Create unique spectral fingerprints.

    Generates compact, distinctive representations of spectral
    signals for identification and deduplication.

    Args:
        fingerprint_dim: Fingerprint vector dimension.
        n_hash_bits: Number of bits for hash code.
    """

    def __init__(
        self,
        fingerprint_dim: int = 32,
        n_hash_bits: int = 64,
    ) -> None:
        self.fingerprint_dim = fingerprint_dim
        self.n_hash_bits = n_hash_bits

        # Random projection matrix for fingerprinting
        self._projection = np.random.randn(fingerprint_dim, 256) / np.sqrt(256)
        # Hash planes for LSH
        self._hash_planes = np.random.randn(n_hash_bits, fingerprint_dim)
        self._fingerprint_count: int = 0

    def fingerprint(self, spectrum: np.ndarray, source: str = "") -> SpectralFingerprint:
        """Create a fingerprint for a spectrum.

        Args:
            spectrum: Input spectrum.
            source: Source identifier.

        Returns:
            SpectralFingerprint object.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        self._fingerprint_count += 1

        # Extract features
        features = self._extract_fingerprint_features(spectrum)

        # Project to fingerprint space
        fp_vector = self._projection @ features

        # Normalize
        norm = np.linalg.norm(fp_vector) + 1e-12
        fp_vector = fp_vector / norm

        # Compute hash
        hash_code = self._compute_hash(fp_vector)

        return SpectralFingerprint(
            fingerprint_id=f"fp_{self._fingerprint_count}",
            features=fp_vector,
            hash_code=hash_code,
            source=source,
        )

    def compare(self, fp1: SpectralFingerprint, fp2: SpectralFingerprint) -> float:
        """Compare two fingerprints.

        Args:
            fp1: First fingerprint.
            fp2: Second fingerprint.

        Returns:
            Similarity score (0-1).
        """
        n = min(len(fp1.features), len(fp2.features))
        cos_sim = np.dot(fp1.features[:n], fp2.features[:n])
        return float((cos_sim + 1) / 2)  # Map [-1,1] to [0,1]

    def _extract_fingerprint_features(self, spectrum: np.ndarray) -> np.ndarray:
        """Extract features for fingerprinting."""
        # Normalize to fixed size
        if len(spectrum) > 256:
            # Subsample
            indices = np.linspace(0, len(spectrum) - 1, 256, dtype=int)
            features = spectrum[indices]
        elif len(spectrum) < 256:
            features = np.pad(spectrum, (0, 256 - len(spectrum)))
        else:
            features = spectrum.copy()

        # Statistical features overlay
        abs_spec = np.abs(features)
        features[:8] = [
            np.mean(abs_spec), np.std(abs_spec),
            np.max(abs_spec), np.min(abs_spec),
            float(np.argmax(abs_spec)) / 256,
            float(np.sum(abs_spec ** 2)),
            float(np.median(abs_spec)),
            float(np.percentile(abs_spec, 75) - np.percentile(abs_spec, 25)),
        ]

        return features

    def _compute_hash(self, fp_vector: np.ndarray) -> int:
        """Compute locality-sensitive hash."""
        projections = self._hash_planes @ fp_vector
        bits = (projections > 0).astype(int)
        # Convert to integer (first 32 bits for int safety)
        hash_code = 0
        for i in range(min(32, len(bits))):
            hash_code |= int(bits[i]) << i
        return hash_code

    @property
    def n_fingerprints(self) -> int:
        """Total fingerprints created."""
        return self._fingerprint_count


# =============================================================================
# Pattern Evolution Tracker
# =============================================================================


class PatternEvolutionTracker:
    """Track how spectral patterns evolve over time.

    Monitors pattern changes, detects trends, and classifies
    evolution behavior.

    Args:
        max_history: Maximum snapshots per pattern.
        change_threshold: Threshold for significant change.
    """

    def __init__(
        self,
        max_history: int = 100,
        change_threshold: float = 0.1,
    ) -> None:
        self.max_history = max_history
        self.change_threshold = change_threshold
        self._histories: Dict[str, List[EvolutionSnapshot]] = {}

    def record(self, pattern_id: str, pattern: np.ndarray, metrics: Optional[Dict[str, float]] = None) -> None:
        """Record a new snapshot of a pattern.

        Args:
            pattern_id: Pattern identifier.
            pattern: Current pattern state.
            metrics: Optional associated metrics.
        """
        if pattern_id not in self._histories:
            self._histories[pattern_id] = []

        snapshot = EvolutionSnapshot(
            timestamp=time.time(),
            pattern=np.atleast_1d(pattern).flatten().copy(),
            metrics=metrics or {},
        )

        self._histories[pattern_id].append(snapshot)
        if len(self._histories[pattern_id]) > self.max_history:
            self._histories[pattern_id].pop(0)

    def classify_evolution(self, pattern_id: str) -> EvolutionType:
        """Classify the evolution behavior of a pattern.

        Args:
            pattern_id: Pattern to analyze.

        Returns:
            EvolutionType classification.
        """
        history = self._histories.get(pattern_id, [])
        if len(history) < 3:
            return EvolutionType.STABLE

        # Compute change rates
        changes = []
        for i in range(1, len(history)):
            n = min(len(history[i].pattern), len(history[i-1].pattern))
            diff = np.mean(np.abs(history[i].pattern[:n] - history[i-1].pattern[:n]))
            changes.append(diff)

        changes = np.array(changes)
        mean_change = np.mean(changes)
        change_trend = np.polyfit(np.arange(len(changes)), changes, 1)[0] if len(changes) > 2 else 0

        # Classify
        if mean_change < self.change_threshold * 0.1:
            return EvolutionType.STABLE
        elif change_trend > self.change_threshold:
            return EvolutionType.DEGRADING
        elif change_trend < -self.change_threshold:
            return EvolutionType.EMERGING
        elif np.std(changes) > mean_change:
            return EvolutionType.OSCILLATING
        else:
            return EvolutionType.DRIFTING

    def get_trend(self, pattern_id: str) -> Dict[str, float]:
        """Get evolution trend metrics.

        Args:
            pattern_id: Pattern to analyze.

        Returns:
            Dictionary of trend metrics.
        """
        history = self._histories.get(pattern_id, [])
        if len(history) < 2:
            return {"change_rate": 0.0, "stability": 1.0}

        # Compute per-step changes
        changes = []
        for i in range(1, len(history)):
            n = min(len(history[i].pattern), len(history[i-1].pattern))
            diff = float(np.mean(np.abs(history[i].pattern[:n] - history[i-1].pattern[:n])))
            changes.append(diff)

        # Energy trend
        energies = [float(np.sum(s.pattern ** 2)) for s in history]
        energy_trend = np.polyfit(np.arange(len(energies)), energies, 1)[0] if len(energies) > 2 else 0

        return {
            "change_rate": float(np.mean(changes)),
            "stability": float(1.0 - min(1.0, np.std(changes) / (np.mean(changes) + 1e-12))),
            "energy_trend": float(energy_trend),
            "n_snapshots": len(history),
            "evolution_type": self.classify_evolution(pattern_id).value,
        }

    @property
    def n_patterns_tracked(self) -> int:
        """Number of patterns being tracked."""
        return len(self._histories)
