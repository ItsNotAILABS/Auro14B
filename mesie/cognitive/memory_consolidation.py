"""Memory Consolidation and Replay System.

Implements biologically-inspired memory consolidation algorithms for
the TAURUS memory system. Provides experience replay, memory
prioritization, dream-like consolidation, and synaptic homeostasis
for spectral intelligence systems.

Key Components:
    - ExperienceReplayBuffer: Priority-based replay buffer
    - MemoryConsolidator: Offline consolidation with importance weighting
    - DreamReplayEngine: Generative replay for memory augmentation
    - SynapticHomeostasis: Automatic memory pruning and strengthening
    - ConsolidationPipeline: Full consolidation orchestrator
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesie.cognitive.taurus_memory import (
    MemoryTrace,
    TaurusMemoryStore,
    RetrievalResult,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation.

    Args:
        replay_buffer_size: Maximum size of replay buffer.
        consolidation_interval: Steps between consolidation events.
        priority_exponent: Exponent for priority-based sampling.
        dream_replay_ratio: Fraction of replay from generated samples.
        homeostasis_target: Target average memory strength.
        strengthening_rate: Rate of memory strengthening.
        weakening_rate: Rate of memory weakening.
        similarity_threshold: Threshold for merging similar memories.
        max_consolidation_batch: Maximum batch size per consolidation.
    """

    replay_buffer_size: int = 1000
    consolidation_interval: int = 50
    priority_exponent: float = 0.6
    dream_replay_ratio: float = 0.2
    homeostasis_target: float = 0.5
    strengthening_rate: float = 0.1
    weakening_rate: float = 0.05
    similarity_threshold: float = 0.9
    max_consolidation_batch: int = 32


@dataclass
class ConsolidationEvent:
    """Record of a consolidation event.

    Args:
        timestamp: When consolidation occurred.
        n_replayed: Number of memories replayed.
        n_strengthened: Number of memories strengthened.
        n_weakened: Number of memories weakened.
        n_merged: Number of memories merged.
        n_pruned: Number of memories pruned.
        mean_priority: Average priority of replayed items.
        duration: Duration of consolidation in seconds.
    """

    timestamp: float = field(default_factory=time.time)
    n_replayed: int = 0
    n_strengthened: int = 0
    n_weakened: int = 0
    n_merged: int = 0
    n_pruned: int = 0
    mean_priority: float = 0.0
    duration: float = 0.0


# =============================================================================
# Experience Replay Buffer
# =============================================================================


class ExperienceReplayBuffer:
    """Priority-based experience replay buffer.

    Stores experiences with associated priorities and supports
    weighted sampling for replay during consolidation.
    Uses proportional prioritization with importance sampling
    corrections.

    Args:
        capacity: Maximum buffer capacity.
        priority_exponent: Exponent for priority weighting (alpha).
        importance_sampling_exponent: IS correction exponent (beta).
    """

    def __init__(
        self,
        capacity: int = 1000,
        priority_exponent: float = 0.6,
        importance_sampling_exponent: float = 0.4,
    ) -> None:
        self.capacity = capacity
        self.priority_exponent = priority_exponent
        self.importance_sampling_exponent = importance_sampling_exponent

        self._buffer: List[Dict[str, Any]] = []
        self._priorities: List[float] = []
        self._max_priority: float = 1.0
        self._total_added: int = 0

    def add(
        self,
        experience: Dict[str, Any],
        priority: Optional[float] = None,
    ) -> None:
        """Add an experience to the buffer.

        Args:
            experience: Experience dictionary with at minimum an 'embedding' key.
            priority: Initial priority (defaults to max priority).
        """
        p = priority if priority is not None else self._max_priority
        self._total_added += 1

        if len(self._buffer) >= self.capacity:
            # Replace lowest priority
            min_idx = int(np.argmin(self._priorities))
            self._buffer[min_idx] = experience
            self._priorities[min_idx] = p
        else:
            self._buffer.append(experience)
            self._priorities.append(p)

        self._max_priority = max(self._max_priority, p)

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Sample a batch with priority-based weighting.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Tuple of (experiences, indices, importance_weights).
        """
        n = len(self._buffer)
        if n == 0:
            return [], np.array([]), np.array([])

        batch_size = min(batch_size, n)

        # Compute sampling probabilities
        priorities = np.array(self._priorities)
        probs = priorities ** self.priority_exponent
        probs /= np.sum(probs) + 1e-12

        # Sample indices
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.importance_sampling_exponent)
        weights /= np.max(weights) + 1e-12  # Normalize

        experiences = [self._buffer[i] for i in indices]
        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update.
            priorities: New priority values.
        """
        for idx, p in zip(indices, priorities):
            self._priorities[int(idx)] = float(p)
            self._max_priority = max(self._max_priority, float(p))

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

    @property
    def total_added(self) -> int:
        """Total experiences ever added."""
        return self._total_added

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self._priorities:
            return {"size": 0, "mean_priority": 0.0}

        priorities = np.array(self._priorities)
        return {
            "size": len(self._buffer),
            "mean_priority": float(np.mean(priorities)),
            "max_priority": float(np.max(priorities)),
            "min_priority": float(np.min(priorities)),
            "std_priority": float(np.std(priorities)),
            "total_added": self._total_added,
        }


# =============================================================================
# Memory Consolidator
# =============================================================================


class MemoryConsolidator:
    """Consolidate memories through importance-weighted replay.

    Performs offline consolidation by replaying important memories,
    strengthening frequently accessed ones, and weakening rarely
    used ones. Merges similar memories to reduce redundancy.

    Args:
        memory_store: TAURUS memory store to consolidate.
        config: Consolidation configuration.
    """

    def __init__(
        self,
        memory_store: TaurusMemoryStore,
        config: Optional[ConsolidationConfig] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or ConsolidationConfig()
        self._consolidation_events: List[ConsolidationEvent] = []
        self._steps_since_consolidation: int = 0

    def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered.

        Returns:
            True if enough steps have elapsed since last consolidation.
        """
        return self._steps_since_consolidation >= self.config.consolidation_interval

    def step(self) -> None:
        """Advance the consolidation counter."""
        self._steps_since_consolidation += 1

    def consolidate(self) -> ConsolidationEvent:
        """Perform a consolidation pass over the memory store.

        Steps:
        1. Identify memories to strengthen (high access, high importance)
        2. Identify memories to weaken (low access, old)
        3. Merge highly similar memories
        4. Prune memories below threshold

        Returns:
            ConsolidationEvent with statistics.
        """
        start_time = time.time()
        self._steps_since_consolidation = 0

        n_strengthened = 0
        n_weakened = 0
        n_merged = 0
        n_pruned = 0

        current_time = time.time()
        traces = self.memory_store._traces

        if not traces:
            event = ConsolidationEvent(duration=time.time() - start_time)
            self._consolidation_events.append(event)
            return event

        # 1. Strengthen/Weaken based on access patterns
        for trace in traces:
            strength = trace.effective_strength(current_time)

            if trace.access_count > 3 and strength > self.config.homeostasis_target:
                # Strengthen frequently accessed memories
                trace.importance *= (1.0 + self.config.strengthening_rate)
                trace.importance = min(trace.importance, 10.0)
                n_strengthened += 1
            elif trace.access_count == 0 and strength < self.config.homeostasis_target * 0.5:
                # Weaken unused memories
                trace.importance *= (1.0 - self.config.weakening_rate)
                n_weakened += 1

        # 2. Merge similar memories
        n_merged = self._merge_similar(traces)

        # 3. Prune very weak memories
        initial_count = len(traces)
        self.memory_store._traces = [
            t for t in traces
            if t.effective_strength(current_time) > self.memory_store.consolidation_threshold * 0.1
        ]
        n_pruned = initial_count - len(self.memory_store._traces)

        duration = time.time() - start_time
        event = ConsolidationEvent(
            n_strengthened=n_strengthened,
            n_weakened=n_weakened,
            n_merged=n_merged,
            n_pruned=n_pruned,
            duration=duration,
        )
        self._consolidation_events.append(event)
        return event

    def _merge_similar(self, traces: List[MemoryTrace]) -> int:
        """Merge highly similar memory traces."""
        if len(traces) < 2:
            return 0

        merged_count = 0
        to_remove = set()

        for i in range(len(traces)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(traces)):
                if j in to_remove:
                    continue

                # Compute similarity
                min_len = min(len(traces[i].embedding), len(traces[j].embedding))
                if min_len == 0:
                    continue

                norm_i = np.linalg.norm(traces[i].embedding[:min_len]) + 1e-12
                norm_j = np.linalg.norm(traces[j].embedding[:min_len]) + 1e-12
                similarity = float(
                    np.dot(traces[i].embedding[:min_len], traces[j].embedding[:min_len])
                    / (norm_i * norm_j)
                )

                if similarity > self.config.similarity_threshold:
                    # Merge j into i
                    traces[i].importance = max(traces[i].importance, traces[j].importance)
                    traces[i].access_count += traces[j].access_count
                    # Average embeddings
                    traces[i].embedding = (
                        traces[i].embedding * traces[i].importance
                        + traces[j].embedding * traces[j].importance
                    ) / (traces[i].importance + traces[j].importance)
                    to_remove.add(j)
                    merged_count += 1

        # Remove merged traces
        if to_remove:
            self.memory_store._traces = [
                t for idx, t in enumerate(traces) if idx not in to_remove
            ]

        return merged_count

    @property
    def consolidation_history(self) -> List[ConsolidationEvent]:
        """History of consolidation events."""
        return self._consolidation_events

    @property
    def n_consolidations(self) -> int:
        """Total number of consolidation events."""
        return len(self._consolidation_events)


# =============================================================================
# Dream Replay Engine
# =============================================================================


class DreamReplayEngine:
    """Generative replay for memory augmentation.

    Generates synthetic spectral experiences by interpolating and
    perturbing existing memories. Inspired by memory replay during
    sleep/dreaming in biological neural systems.

    Args:
        noise_scale: Scale of noise added during dream generation.
        interpolation_points: Number of interpolation points per pair.
        creativity: How much to deviate from stored patterns (0-1).
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        interpolation_points: int = 3,
        creativity: float = 0.3,
    ) -> None:
        self.noise_scale = noise_scale
        self.interpolation_points = interpolation_points
        self.creativity = creativity
        self._dream_count: int = 0

    def generate_dreams(
        self,
        memory_store: TaurusMemoryStore,
        n_dreams: int = 10,
    ) -> List[np.ndarray]:
        """Generate dream-like synthetic spectral experiences.

        Creates new experiences by:
        1. Randomly selecting pairs of memories
        2. Interpolating between them (mixup)
        3. Adding creative noise perturbations
        4. Applying spectral transformations

        Args:
            memory_store: Source memory store for dream generation.
            n_dreams: Number of dreams to generate.

        Returns:
            List of generated spectral embeddings.
        """
        traces = memory_store._traces
        if len(traces) < 2:
            return []

        dreams = []
        for _ in range(n_dreams):
            self._dream_count += 1
            dream = self._generate_single_dream(traces)
            if dream is not None:
                dreams.append(dream)

        return dreams

    def _generate_single_dream(self, traces: List[MemoryTrace]) -> Optional[np.ndarray]:
        """Generate a single dream experience."""
        if len(traces) < 2:
            return None

        # Select two random memories weighted by importance
        importances = np.array([t.importance for t in traces])
        probs = importances / (np.sum(importances) + 1e-12)

        idx_a, idx_b = np.random.choice(len(traces), size=2, replace=False, p=probs)
        trace_a = traces[idx_a]
        trace_b = traces[idx_b]

        # Interpolation
        alpha = np.random.beta(2, 2)  # Centered around 0.5
        min_len = min(len(trace_a.embedding), len(trace_b.embedding))
        interpolated = (
            alpha * trace_a.embedding[:min_len]
            + (1 - alpha) * trace_b.embedding[:min_len]
        )

        # Creative noise
        noise = np.random.randn(min_len) * self.noise_scale * self.creativity
        dream = interpolated + noise

        # Random spectral transformation
        transform = np.random.choice(["shift", "scale", "warp", "none"])
        if transform == "shift":
            shift = int(np.random.randint(-5, 6))
            dream = np.roll(dream, shift)
        elif transform == "scale":
            scale = np.random.uniform(0.8, 1.2)
            dream *= scale
        elif transform == "warp":
            # Non-linear warping
            x = np.linspace(0, 1, min_len)
            warp = x + self.creativity * 0.1 * np.sin(2 * np.pi * x * np.random.randint(1, 4))
            warp = np.clip(warp, 0, 1)
            dream = np.interp(np.linspace(0, 1, min_len), warp, dream)

        return dream

    def replay_with_dreams(
        self,
        memory_store: TaurusMemoryStore,
        replay_buffer: ExperienceReplayBuffer,
        n_replays: int = 20,
        dream_ratio: float = 0.2,
    ) -> Dict[str, int]:
        """Perform replay mixing real memories and dreams.

        Args:
            memory_store: Source memory store.
            replay_buffer: Buffer to store replay experiences.
            n_replays: Total number of replay items.
            dream_ratio: Fraction that should be dreams.

        Returns:
            Dictionary with replay statistics.
        """
        n_dreams = int(n_replays * dream_ratio)
        n_real = n_replays - n_dreams

        # Generate dreams
        dreams = self.generate_dreams(memory_store, n_dreams)
        for dream in dreams:
            replay_buffer.add(
                {"embedding": dream, "is_dream": True},
                priority=0.5,  # Lower priority for dreams
            )

        # Sample real memories
        traces = memory_store._traces
        if traces:
            indices = np.random.choice(
                len(traces),
                size=min(n_real, len(traces)),
                replace=False,
            )
            for idx in indices:
                replay_buffer.add(
                    {"embedding": traces[idx].embedding, "is_dream": False},
                    priority=traces[idx].importance,
                )

        return {
            "n_dreams_generated": len(dreams),
            "n_real_replayed": min(n_real, len(traces)),
            "total_dream_count": self._dream_count,
        }

    @property
    def dream_count(self) -> int:
        """Total dreams generated."""
        return self._dream_count


# =============================================================================
# Synaptic Homeostasis
# =============================================================================


class SynapticHomeostasis:
    """Automatic memory strength regulation.

    Implements synaptic homeostasis scaling (SHS) to maintain
    stable overall memory activity. Prevents runaway strengthening
    or catastrophic forgetting by normalizing memory strengths
    toward a target distribution.

    Args:
        target_mean_strength: Target average memory strength.
        target_std_strength: Target strength standard deviation.
        scaling_rate: Rate of homeostatic adjustment.
        min_strength: Minimum allowed memory strength.
        max_strength: Maximum allowed memory strength.
    """

    def __init__(
        self,
        target_mean_strength: float = 0.5,
        target_std_strength: float = 0.2,
        scaling_rate: float = 0.05,
        min_strength: float = 0.01,
        max_strength: float = 5.0,
    ) -> None:
        self.target_mean_strength = target_mean_strength
        self.target_std_strength = target_std_strength
        self.scaling_rate = scaling_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        self._adjustment_history: List[Dict[str, float]] = []

    def regulate(self, memory_store: TaurusMemoryStore) -> Dict[str, Any]:
        """Apply homeostatic regulation to memory store.

        Adjusts all memory importances to maintain stable overall
        activity levels. Strong memories are slightly weakened,
        weak memories are slightly strengthened.

        Args:
            memory_store: Memory store to regulate.

        Returns:
            Dictionary with regulation statistics.
        """
        traces = memory_store._traces
        if not traces:
            return {"n_regulated": 0, "adjustment": 0.0}

        # Compute current statistics
        current_time = time.time()
        strengths = np.array([t.effective_strength(current_time) for t in traces])
        current_mean = float(np.mean(strengths))
        current_std = float(np.std(strengths))

        # Compute scaling factor to bring mean toward target
        if current_mean > 0:
            mean_scale = self.target_mean_strength / current_mean
        else:
            mean_scale = 1.0

        # Apply gradual scaling
        scale = 1.0 + self.scaling_rate * (mean_scale - 1.0)
        n_regulated = 0

        for trace in traces:
            old_importance = trace.importance
            trace.importance *= scale

            # Clamp to bounds
            trace.importance = np.clip(trace.importance, self.min_strength, self.max_strength)

            if trace.importance != old_importance:
                n_regulated += 1

        # Record adjustment
        adjustment = {
            "scale_applied": float(scale),
            "mean_before": current_mean,
            "mean_after": float(np.mean([t.effective_strength(current_time) for t in traces])),
            "std_before": current_std,
            "n_regulated": n_regulated,
        }
        self._adjustment_history.append(adjustment)

        return adjustment

    def get_health_metrics(self, memory_store: TaurusMemoryStore) -> Dict[str, Any]:
        """Compute memory health metrics.

        Args:
            memory_store: Memory store to assess.

        Returns:
            Health metrics dictionary.
        """
        traces = memory_store._traces
        if not traces:
            return {
                "health_score": 1.0,
                "n_memories": 0,
                "mean_strength": 0.0,
            }

        current_time = time.time()
        strengths = np.array([t.effective_strength(current_time) for t in traces])
        importances = np.array([t.importance for t in traces])
        access_counts = np.array([t.access_count for t in traces])

        # Health score: how close to target distribution
        mean_deviation = abs(np.mean(strengths) - self.target_mean_strength)
        std_deviation = abs(np.std(strengths) - self.target_std_strength)
        health_score = float(np.exp(-(mean_deviation + std_deviation)))

        return {
            "health_score": health_score,
            "n_memories": len(traces),
            "mean_strength": float(np.mean(strengths)),
            "std_strength": float(np.std(strengths)),
            "mean_importance": float(np.mean(importances)),
            "mean_access_count": float(np.mean(access_counts)),
            "utilization": len(traces) / memory_store.capacity,
            "target_mean": self.target_mean_strength,
            "deviation_from_target": float(mean_deviation),
        }

    @property
    def n_adjustments(self) -> int:
        """Number of homeostatic adjustments performed."""
        return len(self._adjustment_history)


# =============================================================================
# Consolidation Pipeline
# =============================================================================


class ConsolidationPipeline:
    """Full memory consolidation pipeline.

    Orchestrates experience replay, consolidation, dream replay,
    and homeostatic regulation into a unified memory maintenance
    system for the TAURUS memory store.

    Args:
        memory_store: TAURUS memory store to maintain.
        config: Consolidation configuration.
    """

    def __init__(
        self,
        memory_store: TaurusMemoryStore,
        config: Optional[ConsolidationConfig] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or ConsolidationConfig()

        # Components
        self._replay_buffer = ExperienceReplayBuffer(
            capacity=self.config.replay_buffer_size,
            priority_exponent=self.config.priority_exponent,
        )
        self._consolidator = MemoryConsolidator(memory_store, self.config)
        self._dream_engine = DreamReplayEngine(
            creativity=self.config.dream_replay_ratio,
        )
        self._homeostasis = SynapticHomeostasis(
            target_mean_strength=self.config.homeostasis_target,
            scaling_rate=self.config.strengthening_rate,
        )

        # State
        self._total_steps: int = 0
        self._auto_consolidation: bool = True

    def step(self, observation: Optional[np.ndarray] = None) -> Optional[ConsolidationEvent]:
        """Advance one step, optionally with a new observation.

        Automatically triggers consolidation when the interval is reached.

        Args:
            observation: Optional new spectral observation to buffer.

        Returns:
            ConsolidationEvent if consolidation was triggered, None otherwise.
        """
        self._total_steps += 1
        self._consolidator.step()

        # Buffer the observation
        if observation is not None:
            self._replay_buffer.add(
                {"embedding": np.atleast_1d(observation).flatten(), "is_dream": False},
                priority=float(np.linalg.norm(observation)),
            )

        # Auto-consolidation
        if self._auto_consolidation and self._consolidator.should_consolidate():
            return self.run_consolidation()

        return None

    def run_consolidation(self) -> ConsolidationEvent:
        """Manually trigger a full consolidation cycle.

        Steps:
        1. Memory consolidation (strengthen/weaken/merge/prune)
        2. Dream replay (generate and store synthetic experiences)
        3. Homeostatic regulation (maintain stable activity levels)

        Returns:
            ConsolidationEvent with combined statistics.
        """
        start_time = time.time()

        # 1. Core consolidation
        event = self._consolidator.consolidate()

        # 2. Dream replay
        dream_stats = self._dream_engine.replay_with_dreams(
            self.memory_store,
            self._replay_buffer,
            n_replays=self.config.max_consolidation_batch,
            dream_ratio=self.config.dream_replay_ratio,
        )

        # 3. Homeostatic regulation
        regulation = self._homeostasis.regulate(self.memory_store)

        # Update event with full stats
        event.n_replayed = dream_stats.get("n_real_replayed", 0) + dream_stats.get("n_dreams_generated", 0)
        event.duration = time.time() - start_time

        return event

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive memory system health metrics.

        Returns:
            Dictionary with health metrics from all components.
        """
        return {
            "memory_store": self.memory_store.get_attention_analysis(),
            "replay_buffer": self._replay_buffer.get_statistics(),
            "homeostasis": self._homeostasis.get_health_metrics(self.memory_store),
            "consolidation": {
                "n_consolidations": self._consolidator.n_consolidations,
                "steps_since_last": self._consolidator._steps_since_consolidation,
                "total_steps": self._total_steps,
            },
            "dreams": {
                "total_generated": self._dream_engine.dream_count,
            },
        }

    @property
    def replay_buffer(self) -> ExperienceReplayBuffer:
        """Access the replay buffer."""
        return self._replay_buffer

    @property
    def total_steps(self) -> int:
        """Total steps processed."""
        return self._total_steps

    def set_auto_consolidation(self, enabled: bool) -> None:
        """Enable or disable automatic consolidation.

        Args:
            enabled: Whether to auto-consolidate.
        """
        self._auto_consolidation = enabled
