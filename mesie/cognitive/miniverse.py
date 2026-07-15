"""Miniverse nesting — recursive containment, scale-bridging, and downward attention.

Implements the three-layer nesting architecture:
1. Recursive Containment: Outer memory objects contain inner MESIE engines
   that re-activate on query (memories that re-resonate).
2. Scale-Bridging Protocol: A formal interface promoting MICRO-layer MatchResults
   into MemoryEntries at the next scale, with resonance score as importance weight.
3. Downward Attention: The outer layer selects which inner micro-patterns to
   amplify based on cognitive context via the SpectralAttentionAdapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.cognitive.attention_adapter import SpectralAttentionAdapter
from mesie.cognitive.memory_adapter import SpectralMemoryAdapter
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.features.resonance import compute_resonance_score
from mesie.io.loaders import RecordInput, load_record
from mesie.matching.matcher import MatchResult, SpectralMatcher
from mesie.pretraining.spectral_memory import MemoryEntry, SpectralMemoryStore


# ---------------------------------------------------------------------------
# Scale-Bridging Protocol
# ---------------------------------------------------------------------------


@dataclass
class ScaleBridgeConfig:
    """Configuration for scale-bridging promotion.

    Attributes:
        resonance_weight: How much resonance score influences importance.
        score_weight: How much the composite match score influences importance.
        min_importance: Floor importance for promoted entries.
        event_type_map: Mapping from score ranges to event types.
    """

    resonance_weight: float = 0.6
    score_weight: float = 0.4
    min_importance: float = 0.1
    event_type_map: Dict[str, float] = field(default_factory=lambda: {
        "anomaly": 0.9,
        "resonance": 0.7,
        "drift": 0.5,
        "normal": 0.0,
    })


class ScaleBridge:
    """Promote MICRO-layer MatchResults into MemoryEntries at the next scale.

    The bridge converts low-level pattern matching results into higher-order
    memory objects, using the resonance score as the importance weight.

    Args:
        vectorizer: Vectorizer for generating embeddings from records.
        config: Scale-bridging configuration.
    """

    def __init__(
        self,
        vectorizer: Optional[SpectralVectorizer] = None,
        config: Optional[ScaleBridgeConfig] = None,
    ) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()
        self.config = config or ScaleBridgeConfig()

    def promote(
        self,
        match_result: MatchResult,
        candidate_record: RecordInput,
        timestamp: Optional[float] = None,
    ) -> MemoryEntry:
        """Promote a MatchResult into a MemoryEntry at the outer scale.

        The resonance score of the candidate becomes the importance weight,
        blended with the composite match score.

        Args:
            match_result: Result from MICRO-layer spectral matching.
            candidate_record: The candidate record that produced the match.
            timestamp: Optional timestamp; defaults to current time.

        Returns:
            MemoryEntry suitable for storage in the outer-scale memory store.
        """
        rec = load_record(candidate_record)
        embedding = self.vectorizer.transform(rec)

        # Compute resonance-derived importance
        resonance_scores = [
            compute_resonance_score(comp) for comp in rec.components
        ]
        avg_resonance = float(np.mean(resonance_scores)) if resonance_scores else 1.0
        # Normalize resonance to [0, 1] range using sigmoid-like mapping
        normalized_resonance = 1.0 - 1.0 / (1.0 + avg_resonance)

        importance = (
            self.config.resonance_weight * normalized_resonance
            + self.config.score_weight * match_result.score
        )
        importance = max(importance, self.config.min_importance)

        # Determine event type from score thresholds
        event_type = "normal"
        for etype, threshold in sorted(
            self.config.event_type_map.items(), key=lambda x: x[1], reverse=True
        ):
            if match_result.score >= threshold:
                event_type = etype
                break

        return MemoryEntry(
            timestamp=timestamp if timestamp is not None else time.time(),
            embedding=embedding,
            event_type=event_type,
            metadata={
                "source_reference_id": match_result.reference_id,
                "source_candidate_id": match_result.candidate_id,
                "match_score": match_result.score,
                "resonance_score": avg_resonance,
                "metrics": match_result.metrics,
                "component_scores": match_result.component_scores,
            },
            importance=importance,
        )

    def batch_promote(
        self,
        results: Sequence[MatchResult],
        records: Sequence[RecordInput],
        base_timestamp: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """Promote a batch of MatchResults into MemoryEntries.

        Args:
            results: Sequence of match results.
            records: Corresponding candidate records.
            base_timestamp: Starting timestamp; entries are offset by index.

        Returns:
            List of promoted MemoryEntries.
        """
        base_ts = base_timestamp if base_timestamp is not None else time.time()
        entries = []
        for i, (result, record) in enumerate(zip(results, records)):
            entries.append(self.promote(result, record, timestamp=base_ts + i))
        return entries


# ---------------------------------------------------------------------------
# Recursive Containment
# ---------------------------------------------------------------------------


@dataclass
class ContainedEngine:
    """An inner MESIE engine contained within a memory object.

    When the outer layer queries this memory and it re-resonates,
    the inner engine activates to perform micro-level pattern matching.

    Attributes:
        engine: The inner SpectralMatcher configured with reference patterns.
        memory_entry: The outer-scale memory entry wrapping this engine.
        activation_threshold: Minimum similarity to trigger re-resonance.
        activation_count: Number of times this engine has been activated.
    """

    engine: SpectralMatcher
    memory_entry: MemoryEntry
    activation_threshold: float = 0.5
    activation_count: int = 0

    def activate(self, candidate: RecordInput) -> Optional[MatchResult]:
        """Activate the inner engine if the candidate re-resonates.

        Args:
            candidate: Incoming candidate record.

        Returns:
            MatchResult if the inner engine produces a match above threshold,
            None otherwise.
        """
        try:
            result = self.engine.match(candidate)
        except RuntimeError:
            return None

        if result.score >= self.activation_threshold:
            self.activation_count += 1
            return result
        return None


class RecursiveMemoryContainer:
    """Outer-layer container holding memory objects with inner MESIE engines.

    Each stored memory can itself contain a fitted SpectralMatcher that
    re-activates when a query resonates with its embedding — implementing
    the concept of "a memory that re-resonates."

    Args:
        memory_store: The outer-scale spectral memory store.
        scale_bridge: Bridge for promoting inner results to outer entries.
        activation_threshold: Default threshold for inner engine activation.
    """

    def __init__(
        self,
        memory_store: Optional[SpectralMemoryStore] = None,
        scale_bridge: Optional[ScaleBridge] = None,
        activation_threshold: float = 0.5,
    ) -> None:
        self.memory_store = memory_store or SpectralMemoryStore()
        self.scale_bridge = scale_bridge or ScaleBridge()
        self.activation_threshold = activation_threshold
        self._contained: List[ContainedEngine] = []
        self._vectorizer = self.scale_bridge.vectorizer

    @property
    def contained_engines(self) -> List[ContainedEngine]:
        """List of contained inner engines."""
        return list(self._contained)

    def contain(
        self,
        references: Sequence[RecordInput],
        timestamp: Optional[float] = None,
        activation_threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContainedEngine:
        """Create and store a contained inner engine from reference records.

        Fits a SpectralMatcher on the references and wraps it in a memory
        entry stored in the outer memory store.

        Args:
            references: Reference records to fit the inner engine with.
            timestamp: Optional timestamp for the memory entry.
            activation_threshold: Override default activation threshold.
            metadata: Additional metadata for the memory entry.

        Returns:
            The ContainedEngine wrapper.
        """
        # Fit an inner matcher
        inner_matcher = SpectralMatcher()
        inner_matcher.fit(references)

        # Create an aggregate embedding for the outer memory entry
        embeddings = [self._vectorizer.transform(r) for r in references]
        aggregate_embedding = np.mean(embeddings, axis=0)

        ts = timestamp if timestamp is not None else time.time()
        threshold = activation_threshold or self.activation_threshold

        entry = MemoryEntry(
            timestamp=ts,
            embedding=aggregate_embedding,
            event_type="containment",
            metadata={
                "n_references": len(references),
                "activation_threshold": threshold,
                **(metadata or {}),
            },
            importance=2.0,  # Contained engines are high-importance
        )

        # Store in the outer memory
        self.memory_store.store(
            timestamp=entry.timestamp,
            embedding=entry.embedding,
            event_type=entry.event_type,
            metadata=entry.metadata,
            importance=entry.importance,
        )

        contained = ContainedEngine(
            engine=inner_matcher,
            memory_entry=entry,
            activation_threshold=threshold,
        )
        self._contained.append(contained)
        return contained

    def query_and_resonate(
        self,
        candidate: RecordInput,
        top_k: int = 5,
    ) -> List[MatchResult]:
        """Query contained engines and collect re-resonance results.

        First finds which contained engines are similar to the candidate
        (via embedding similarity), then activates those engines for
        detailed micro-level matching.

        Args:
            candidate: Incoming candidate record to match.
            top_k: Maximum number of contained engines to probe.

        Returns:
            List of MatchResults from re-resonating inner engines.
        """
        if not self._contained:
            return []

        # Compute candidate embedding for similarity check
        candidate_embedding = self._vectorizer.transform(candidate)

        # Rank contained engines by embedding similarity
        similarities = []
        for contained in self._contained:
            entry_emb = contained.memory_entry.embedding
            sim = float(
                np.dot(candidate_embedding, entry_emb)
                / (np.linalg.norm(candidate_embedding) * np.linalg.norm(entry_emb) + 1e-12)
            )
            similarities.append(sim)

        # Select top-k most similar
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        # Activate selected engines
        results = []
        for idx in ranked_indices:
            contained = self._contained[idx]
            result = contained.activate(candidate)
            if result is not None:
                results.append(result)

        return results

    def query_promote_and_store(
        self,
        candidate: RecordInput,
        top_k: int = 5,
        timestamp: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """Query, resonate, and promote results into the outer memory.

        Combines recursive query with scale-bridging: inner match results
        are promoted to outer-layer MemoryEntries and stored.

        Args:
            candidate: Incoming record.
            top_k: Number of engines to probe.
            timestamp: Optional base timestamp for new entries.

        Returns:
            List of newly promoted and stored MemoryEntries.
        """
        match_results = self.query_and_resonate(candidate, top_k=top_k)
        if not match_results:
            return []

        base_ts = timestamp if timestamp is not None else time.time()
        promoted = []
        for i, result in enumerate(match_results):
            entry = self.scale_bridge.promote(
                result, candidate, timestamp=base_ts + i
            )
            self.memory_store.store(
                timestamp=entry.timestamp,
                embedding=entry.embedding,
                event_type=entry.event_type,
                metadata=entry.metadata,
                importance=entry.importance,
            )
            promoted.append(entry)

        return promoted


# ---------------------------------------------------------------------------
# Downward Attention
# ---------------------------------------------------------------------------


class DownwardAttention:
    """Outer-layer attention selecting which inner micro-patterns to amplify.

    Uses SpectralAttentionAdapter to compute context-driven weights over
    contained engines, determining which memories receive amplified focus.

    Args:
        attention_adapter: The spectral attention adapter.
        container: The recursive memory container holding inner engines.
        amplification_factor: How much to boost importance of attended engines.
    """

    def __init__(
        self,
        attention_adapter: Optional[SpectralAttentionAdapter] = None,
        container: Optional[RecursiveMemoryContainer] = None,
        amplification_factor: float = 2.0,
    ) -> None:
        self.attention = attention_adapter or SpectralAttentionAdapter()
        self.container = container or RecursiveMemoryContainer()
        self.amplification_factor = amplification_factor

    def compute_attention_over_engines(
        self,
        query: Optional[RecordInput] = None,
    ) -> Dict[int, float]:
        """Compute attention weights over contained inner engines.

        Uses the spectral attention adapter to determine which engines
        should receive amplified focus based on cognitive context.

        Args:
            query: Optional query record for similarity-based attention.
                If None, uses energy-based attention.

        Returns:
            Dictionary mapping engine index to attention weight.
        """
        contained = self.container.contained_engines
        if not contained:
            return {}

        # Use memory entry embeddings as proxy records for attention
        # We compute attention over the aggregate embeddings
        embeddings = np.array([c.memory_entry.embedding for c in contained])
        norms = np.linalg.norm(embeddings, axis=1)

        if query is not None:
            query_embedding = self.attention.vectorizer.transform(query)
            # Similarity-based attention
            similarities = np.array([
                float(
                    np.dot(query_embedding, emb)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-12)
                )
                for emb in embeddings
            ])
            # Softmax
            exp_sim = np.exp(similarities - np.max(similarities))
            weights = exp_sim / (np.sum(exp_sim) + 1e-12)
        else:
            # Energy-based attention
            weights = norms / (np.sum(norms) + 1e-12)

        return {i: float(w) for i, w in enumerate(weights)}

    def amplify(
        self,
        query: Optional[RecordInput] = None,
        threshold: float = 0.0,
    ) -> List[ContainedEngine]:
        """Amplify inner engines based on attention weights.

        Boosts the importance of contained engines that receive high
        attention, modifying their memory entries in place.

        Args:
            query: Optional query for context-driven attention.
            threshold: Minimum attention weight to receive amplification.

        Returns:
            List of amplified ContainedEngine instances (weight > threshold).
        """
        weights = self.compute_attention_over_engines(query)
        contained = self.container.contained_engines
        amplified = []

        for idx, weight in weights.items():
            if weight > threshold:
                engine = contained[idx]
                engine.memory_entry.importance *= (
                    1.0 + (self.amplification_factor - 1.0) * weight
                )
                amplified.append(engine)

        return amplified

    def focused_resonate(
        self,
        candidate: RecordInput,
        context_query: Optional[RecordInput] = None,
        top_k: int = 3,
        attention_threshold: float = 0.1,
    ) -> List[MatchResult]:
        """Query with downward attention focus — only activate attended engines.

        Unlike the container's query_and_resonate which uses embedding
        similarity, this method uses cognitive attention to select engines,
        then activates only the attended subset.

        Args:
            candidate: Incoming record to match.
            context_query: Cognitive context determining attention focus.
            top_k: Maximum engines to activate.
            attention_threshold: Minimum attention weight to activate.

        Returns:
            List of MatchResults from attention-selected engines.
        """
        weights = self.compute_attention_over_engines(context_query or candidate)
        contained = self.container.contained_engines

        # Filter and sort by attention weight
        attended = [
            (idx, w) for idx, w in weights.items() if w >= attention_threshold
        ]
        attended.sort(key=lambda x: x[1], reverse=True)
        attended = attended[:top_k]

        results = []
        for idx, _weight in attended:
            engine = contained[idx]
            result = engine.activate(candidate)
            if result is not None:
                results.append(result)

        return results
