"""AI system connector — bridges MESIE libraries into a unified AI backbone.

Connects embeddings, cognitive adapters, topology, and feature layers
into an integrated AI processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.ai.intelligence_protocols import (
    AttentionFocusModule,
    IntelligenceConfig,
    IntelligenceProtocol,
    ReasoningResult,
    SpectralMemoryBuffer,
)
from mesie.ai.inference import InferenceEngine, PredictionResult
from mesie.cognitive.memory_adapter import SpectralMemoryAdapter
from mesie.cognitive.attention_adapter import SpectralAttentionAdapter
from mesie.cognitive.agent_state_adapter import AgentStateSpectralAdapter, SpectralAnomalyAdapter
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.embeddings.retrieval import SpectralRetriever
from mesie.features.electro_spectral import ElectroSpectralLayer
from mesie.topology.node_mapping import NodeTopologyMapper
from mesie.topology.lineage import RecordLineageTracker
from mesie.io.loaders import RecordInput, load_record


@dataclass
class ConnectorConfig:
    """Configuration for the AI system connector.

    Args:
        intelligence_config: Configuration for reasoning protocols.
        embedding_bands: Number of frequency bands for vectorization.
        anomaly_threshold: Threshold for anomaly detection.
        enable_topology: Whether to use topology-aware weighting.
        enable_lineage_tracking: Whether to track record provenance.
        memory_capacity: Capacity for internal memory buffers.
    """

    intelligence_config: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    embedding_bands: int = 8
    anomaly_threshold: float = 2.0
    enable_topology: bool = True
    enable_lineage_tracking: bool = True
    memory_capacity: int = 500


class AISystemConnector:
    """Unified connector linking all MESIE libraries to AI internal systems.

    Provides a single interface for routing spectral data through
    embeddings, cognitive adapters, topology mapping, feature extraction,
    and intelligence protocols.

    Args:
        config: Connector configuration.
        node_graph: Optional topology adjacency mapping.
    """

    def __init__(
        self,
        config: Optional[ConnectorConfig] = None,
        node_graph: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.config = config or ConnectorConfig()

        # Core subsystems
        self._vectorizer = SpectralVectorizer(n_bands=self.config.embedding_bands)
        self._retriever = SpectralRetriever(vectorizer=self._vectorizer)
        self._electro = ElectroSpectralLayer()

        # Cognitive adapters
        self._memory_adapter = SpectralMemoryAdapter(vectorizer=self._vectorizer)
        self._attention_adapter = SpectralAttentionAdapter(vectorizer=self._vectorizer)
        self._state_adapter = AgentStateSpectralAdapter(vectorizer=self._vectorizer)
        self._anomaly_adapter = SpectralAnomalyAdapter(
            vectorizer=self._vectorizer,
            threshold=self.config.anomaly_threshold,
        )

        # Intelligence protocol
        self._protocol = IntelligenceProtocol(config=self.config.intelligence_config)

        # Topology
        self._topology_mapper = NodeTopologyMapper(node_graph) if self.config.enable_topology else None

        # Lineage
        self._lineage_tracker = RecordLineageTracker() if self.config.enable_lineage_tracking else None

        # Internal state
        self._processed_count = 0
        self._connected_systems: List[str] = []

    def connect(self, system_name: str) -> None:
        """Register a connected AI subsystem.

        Args:
            system_name: Name of the subsystem being connected.
        """
        if system_name not in self._connected_systems:
            self._connected_systems.append(system_name)

    @property
    def connected_systems(self) -> List[str]:
        """List of connected subsystem names."""
        return list(self._connected_systems)

    def ingest(self, record: RecordInput) -> Dict[str, Any]:
        """Ingest a spectral record through all connected systems.

        Routes the record through embedding, feature extraction,
        cognitive adaptation, topology mapping, and intelligence protocols.

        Args:
            record: Input spectral record.

        Returns:
            Comprehensive result dictionary from all subsystems.
        """
        rec = load_record(record)
        self._processed_count += 1

        result: Dict[str, Any] = {"record_id": rec.record_id}

        # Embedding
        embedding = self._vectorizer.transform(rec)
        result["embedding"] = embedding

        # Feature extraction
        signature = self._electro.compute_signature(rec)
        result["spectral_signature"] = {
            "centroid": signature.spectral_centroid,
            "spread": signature.spectral_spread,
            "resonance": signature.frequency_resonance,
            "coherence": signature.coherence_signature,
            "harmonic_alignment": signature.harmonic_alignment,
        }

        # Cognitive memory object
        result["memory_object"] = self._memory_adapter.to_memory_object(rec)

        # Agent state
        result["state_vector"] = self._state_adapter.to_state_vector(rec)

        # Anomaly scoring
        result["anomaly_score"] = self._anomaly_adapter.score_anomaly(rec)
        result["is_anomaly"] = self._anomaly_adapter.is_anomaly(rec)

        # Topology weighting
        if self._topology_mapper and rec.components:
            result["topology_weights"] = [
                self._topology_mapper.compute_weight(c) for c in rec.components
            ]

        # Lineage tracking
        if self._lineage_tracker:
            self._lineage_tracker.register(rec)
            self._lineage_tracker.add_operation(rec.record_id, "ai_connector_ingest")
            result["lineage"] = self._lineage_tracker.get_lineage(rec.record_id)

        # Intelligence protocol observation
        self._protocol.observe(embedding, context={"record_id": rec.record_id})

        return result

    def reason(self, record: RecordInput) -> ReasoningResult:
        """Perform AI reasoning about a spectral record.

        Args:
            record: Input spectral record.

        Returns:
            ReasoningResult from the intelligence protocol.
        """
        rec = load_record(record)
        embedding = self._vectorizer.transform(rec)
        return self._protocol.reason(embedding)

    def compute_attention(
        self,
        records: Sequence[RecordInput],
        query: Optional[RecordInput] = None,
    ) -> np.ndarray:
        """Compute attention weights over records.

        Args:
            records: Records to weight.
            query: Optional query record for similarity-based attention.

        Returns:
            Array of attention weights.
        """
        return self._attention_adapter.compute_attention_weights(records, query)

    def index_records(self, records: Sequence[RecordInput]) -> None:
        """Index records for retrieval.

        Args:
            records: Records to add to the retrieval index.
        """
        self._retriever.index(records)

    def retrieve_similar(self, record: RecordInput, top_k: int = 5) -> List[Any]:
        """Retrieve similar records from the index.

        Args:
            record: Query record.
            top_k: Number of results.

        Returns:
            List of (record_id, distance) tuples.
        """
        return self._retriever.query(record, top_k=top_k)

    def fit_anomaly_baseline(self, records: Sequence[RecordInput]) -> None:
        """Fit anomaly detector baseline.

        Args:
            records: Normal/baseline records.
        """
        self._anomaly_adapter.fit_baseline(records)

    @property
    def processed_count(self) -> int:
        """Number of records processed."""
        return self._processed_count

    @property
    def vectorizer(self) -> SpectralVectorizer:
        """Underlying vectorizer instance."""
        return self._vectorizer

    @property
    def protocol(self) -> IntelligenceProtocol:
        """Intelligence protocol instance."""
        return self._protocol
