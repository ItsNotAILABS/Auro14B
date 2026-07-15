"""Intelligence AI protocols for spectral reasoning.

Provides high-level intelligence protocols that embed AI reasoning
capabilities into spectral data pipelines. These protocols enable
autonomous spectral analysis, anomaly reasoning, and adaptive
decision-making across the MESIE system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class IntelligenceLevel(Enum):
    """Levels of intelligence protocol engagement."""

    PASSIVE = "passive"         # Observe and log only
    REACTIVE = "reactive"       # Respond to threshold events
    ADAPTIVE = "adaptive"       # Learn and adjust parameters
    PREDICTIVE = "predictive"   # Forecast spectral behavior
    AUTONOMOUS = "autonomous"   # Full self-directed reasoning


class ReasoningStrategy(Enum):
    """Strategies for spectral reasoning."""

    STATISTICAL = "statistical"
    PATTERN_MATCHING = "pattern_matching"
    ANOMALY_DETECTION = "anomaly_detection"
    CAUSAL_INFERENCE = "causal_inference"
    ENSEMBLE = "ensemble"


@dataclass
class IntelligenceConfig:
    """Configuration for intelligence protocol behavior.

    Args:
        level: Intelligence engagement level.
        strategy: Primary reasoning strategy.
        confidence_threshold: Minimum confidence for autonomous actions.
        memory_window: Number of past observations to retain.
        adaptation_rate: Rate of parameter adaptation (0-1).
        enable_attention: Whether to use attention-based focus.
        enable_memory: Whether to enable episodic memory.
    """

    level: IntelligenceLevel = IntelligenceLevel.ADAPTIVE
    strategy: ReasoningStrategy = ReasoningStrategy.ENSEMBLE
    confidence_threshold: float = 0.7
    memory_window: int = 100
    adaptation_rate: float = 0.1
    enable_attention: bool = True
    enable_memory: bool = True


@dataclass
class ReasoningResult:
    """Result from an intelligence protocol reasoning step.

    Args:
        conclusion: Primary conclusion or decision.
        confidence: Confidence in the conclusion (0-1).
        evidence: Supporting evidence for the conclusion.
        alternative_hypotheses: Other considered explanations.
        recommended_actions: Suggested follow-up actions.
        metadata: Additional reasoning metadata.
    """

    conclusion: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    alternative_hypotheses: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SpectralMemoryBuffer:
    """Episodic memory buffer for spectral observations.

    Stores past spectral observations with associated context
    to enable pattern recognition across time.

    Args:
        capacity: Maximum number of observations to store.
    """

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self._observations: list[dict[str, Any]] = []
        self._timestamps: list[float] = []
        self._importance_scores: list[float] = []

    def store(
        self,
        observation: np.ndarray,
        context: Optional[dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        importance: float = 1.0,
    ) -> None:
        """Store an observation in memory.

        Args:
            observation: Spectral data observation.
            context: Contextual metadata.
            timestamp: Time of observation.
            importance: Importance weight for retention.
        """
        entry = {
            "observation": observation.copy(),
            "context": context or {},
        }
        self._observations.append(entry)
        self._timestamps.append(timestamp or 0.0)
        self._importance_scores.append(importance)

        # Evict least important if over capacity
        if len(self._observations) > self.capacity:
            min_idx = int(np.argmin(self._importance_scores))
            self._observations.pop(min_idx)
            self._timestamps.pop(min_idx)
            self._importance_scores.pop(min_idx)

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve most relevant observations by similarity.

        Args:
            query: Query spectral vector.
            top_k: Number of results to return.

        Returns:
            List of relevant observation entries.
        """
        if not self._observations:
            return []

        # Compute cosine similarities
        query_flat = query.flatten()
        query_norm = np.linalg.norm(query_flat) + 1e-10

        similarities = []
        for entry in self._observations:
            obs_flat = entry["observation"].flatten()
            # Ensure same length for comparison
            min_len = min(len(query_flat), len(obs_flat))
            sim = np.dot(query_flat[:min_len], obs_flat[:min_len]) / (
                query_norm * (np.linalg.norm(obs_flat[:min_len]) + 1e-10)
            )
            similarities.append(sim)

        # Get top-k indices
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self._observations[i] for i in top_indices]

    @property
    def size(self) -> int:
        """Current number of stored observations."""
        return len(self._observations)

    def clear(self) -> None:
        """Clear all stored observations."""
        self._observations.clear()
        self._timestamps.clear()
        self._importance_scores.clear()


class AttentionFocusModule:
    """Attention-based focus mechanism for spectral analysis.

    Directs analytical focus to the most informative frequency
    regions based on learned attention patterns.

    Args:
        n_frequency_bins: Number of frequency bins to attend over.
        n_attention_heads: Number of parallel attention heads.
    """

    def __init__(
        self,
        n_frequency_bins: int = 128,
        n_attention_heads: int = 4,
    ) -> None:
        self.n_frequency_bins = n_frequency_bins
        self.n_attention_heads = n_attention_heads
        self._attention_weights = np.ones((n_attention_heads, n_frequency_bins)) / n_frequency_bins
        self._focus_history: list[np.ndarray] = []

    def compute_focus(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute attention focus scores across frequency bins.

        Args:
            spectrum: Input spectrum of shape (n_frequency_bins,) or compatible.

        Returns:
            Attention-weighted focus scores.
        """
        spectrum = np.atleast_1d(spectrum).flatten()

        # Adapt spectrum length if needed
        if len(spectrum) != self.n_frequency_bins:
            spectrum = np.interp(
                np.linspace(0, 1, self.n_frequency_bins),
                np.linspace(0, 1, len(spectrum)),
                spectrum,
            )

        # Multi-head attention scoring
        head_scores = np.zeros((self.n_attention_heads, self.n_frequency_bins))
        for h in range(self.n_attention_heads):
            weighted = spectrum * self._attention_weights[h]
            # Softmax normalization
            exp_weighted = np.exp(weighted - np.max(weighted))
            head_scores[h] = exp_weighted / (np.sum(exp_weighted) + 1e-10)

        # Average across heads
        focus = np.mean(head_scores, axis=0)
        self._focus_history.append(focus)
        return focus

    def update_attention(self, feedback: np.ndarray, learning_rate: float = 0.01) -> None:
        """Update attention weights based on feedback.

        Args:
            feedback: Feedback signal indicating informative regions.
            learning_rate: Weight update rate.
        """
        feedback = np.atleast_1d(feedback).flatten()
        if len(feedback) != self.n_frequency_bins:
            feedback = np.interp(
                np.linspace(0, 1, self.n_frequency_bins),
                np.linspace(0, 1, len(feedback)),
                feedback,
            )

        for h in range(self.n_attention_heads):
            self._attention_weights[h] += learning_rate * feedback
            # Normalize to keep attention as distribution
            self._attention_weights[h] = np.abs(self._attention_weights[h])
            total = np.sum(self._attention_weights[h])
            if total > 0:
                self._attention_weights[h] /= total

    @property
    def current_focus(self) -> Optional[np.ndarray]:
        """Most recent focus pattern."""
        return self._focus_history[-1] if self._focus_history else None


class IntelligenceProtocol:
    """Core intelligence protocol for MESIE spectral reasoning.

    Orchestrates memory, attention, and reasoning to provide
    intelligent analysis of spectral data streams.

    Args:
        config: Intelligence protocol configuration.
    """

    def __init__(self, config: Optional[IntelligenceConfig] = None) -> None:
        self.config = config or IntelligenceConfig()
        self._memory = SpectralMemoryBuffer(capacity=self.config.memory_window)
        self._attention = AttentionFocusModule()
        self._reasoning_history: list[ReasoningResult] = []
        self._adaptation_state: dict[str, float] = {}
        self._observation_count = 0

    def observe(self, spectrum: np.ndarray, context: Optional[dict[str, Any]] = None) -> None:
        """Submit a spectral observation to the protocol.

        Args:
            spectrum: Spectral data to observe.
            context: Optional contextual metadata.
        """
        self._observation_count += 1

        # Compute attention focus
        if self.config.enable_attention:
            focus = self._attention.compute_focus(spectrum)
            importance = float(np.max(focus))
        else:
            importance = 1.0

        # Store in memory
        if self.config.enable_memory:
            self._memory.store(
                observation=spectrum,
                context=context,
                importance=importance,
            )

    def reason(self, query_spectrum: np.ndarray) -> ReasoningResult:
        """Perform intelligent reasoning about a spectral query.

        Args:
            query_spectrum: Spectrum to reason about.

        Returns:
            ReasoningResult with conclusions and confidence.
        """
        query_spectrum = np.atleast_1d(query_spectrum).flatten()
        evidence = []
        confidence = 0.5

        # Attention-based analysis
        if self.config.enable_attention:
            focus = self._attention.compute_focus(query_spectrum)
            peak_regions = np.where(focus > np.mean(focus) + np.std(focus))[0]
            if len(peak_regions) > 0:
                evidence.append(
                    f"High-attention regions detected at bins: {peak_regions[:5].tolist()}"
                )
                confidence += 0.1

        # Memory-based pattern matching
        if self.config.enable_memory and self._memory.size > 0:
            similar = self._memory.retrieve(query_spectrum, top_k=3)
            if similar:
                evidence.append(f"Found {len(similar)} similar past observations")
                confidence += 0.1

        # Statistical analysis
        spectrum_stats = {
            "mean": float(np.mean(query_spectrum)),
            "std": float(np.std(query_spectrum)),
            "max": float(np.max(query_spectrum)),
            "energy": float(np.sum(query_spectrum**2)),
        }

        # Anomaly check
        if spectrum_stats["std"] > 2.0 * spectrum_stats["mean"]:
            evidence.append("High variability detected - possible anomaly")
            conclusion = "anomaly_detected"
            confidence = min(confidence + 0.2, 1.0)
        elif spectrum_stats["energy"] < 1e-6:
            evidence.append("Very low energy signal detected")
            conclusion = "low_signal"
            confidence = min(confidence + 0.15, 1.0)
        else:
            conclusion = "normal_operation"
            confidence = min(confidence + 0.1, 1.0)

        result = ReasoningResult(
            conclusion=conclusion,
            confidence=min(confidence, 1.0),
            evidence=evidence,
            alternative_hypotheses=["noise_artifact", "sensor_drift"],
            recommended_actions=self._determine_actions(conclusion),
            metadata={
                "level": self.config.level.value,
                "strategy": self.config.strategy.value,
                "observation_count": self._observation_count,
                "statistics": spectrum_stats,
            },
        )
        self._reasoning_history.append(result)
        return result

    def _determine_actions(self, conclusion: str) -> list[str]:
        """Determine recommended actions based on conclusion."""
        actions = {
            "anomaly_detected": [
                "flag_for_review",
                "increase_monitoring_frequency",
                "cross_reference_with_historical_data",
            ],
            "low_signal": [
                "check_sensor_connectivity",
                "verify_acquisition_parameters",
            ],
            "normal_operation": [
                "continue_monitoring",
                "update_baseline_statistics",
            ],
        }
        return actions.get(conclusion, ["log_observation"])

    def adapt(self, feedback: np.ndarray) -> None:
        """Adapt the protocol based on external feedback.

        Args:
            feedback: Feedback signal for adaptation.
        """
        if self.config.level in (IntelligenceLevel.ADAPTIVE, IntelligenceLevel.AUTONOMOUS):
            self._attention.update_attention(
                feedback, learning_rate=self.config.adaptation_rate
            )

    @property
    def observation_count(self) -> int:
        """Total number of observations processed."""
        return self._observation_count

    @property
    def memory_utilization(self) -> float:
        """Fraction of memory capacity used."""
        return self._memory.size / self._memory.capacity

    @property
    def reasoning_history(self) -> list[ReasoningResult]:
        """History of all reasoning results."""
        return self._reasoning_history
