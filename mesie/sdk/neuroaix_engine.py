"""NeuroAIX™ — Neural Architecture for Intelligent eXperience.

The NeuroAIX engine is the connectome intelligence binding layer that
unifies MESIE spectral processing with the 3D brain simulation.
It provides:

1. MAESIObservationEncoder — Converts raw physical/chemical/biological
   spectral data into structured observations for the connectome.

2. NeuroAIXEngine — The cognitive core that propagates spectral signals
   through 44 brain regions with biologically realistic conduction delays.

3. CognitiveIntegrationLoop — The full perception→integration→memory→action
   loop connecting MESIE sensory processing to agent behavior.

Architecture
------------
    MAESI Observation Layer
    ├── Physical Laws → spectral law embeddings
    ├── Chemical Elements → emission line embeddings
    ├── Biological Systems → process frequency embeddings
    └── Raw Sensor Data → MESIE spectral embeddings
            ↓
    NeuroAIX Connectome Engine
    ├── 44 brain regions (MNI coordinates)
    ├── 68+ white-matter connections
    ├── Signal propagation (~6 mm/ms conduction velocity)
    ├── Functional system integration
    └── Emergent coherence dynamics
            ↓
    Cognitive Integration Loop
    ├── Agent state vector (policy input)
    ├── Spectral memory store (temporal continuity)
    ├── Anomaly detection (deviation alerting)
    └── Lineage reconstruction (causal reasoning)

Naming Convention
-----------------
    MAESI = Multi-Agent Embodied Spectral Intelligence
    NeuroAIX = Neural Architecture for Intelligent eXperience
    MESIE = Multi-Element Spectral Intelligence Engine (foundation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesie.connectome.brain_regions import BrainRegion, BrainSystem, get_default_regions
from mesie.connectome.connectome_graph import ConnectomeGraph, build_default_connectome
from mesie.connectome.environment import ConnectomeEnvironment3D, ActivationState, SignalPacket
from mesie.sdk.constants import ALL_CONSTANTS, UniversalSpectralConstant
from mesie.sdk.physical_laws import PhysicalLaw, SpectralLawRegistry, get_fundamental_laws
from mesie.sdk.chemical_elements import SpectralElement, get_periodic_table
from mesie.sdk.biological_systems import BiologicalSystem, get_biological_systems


# ═══════════════════════════════════════════════════════════════════════════════
# MAESI OBSERVATION ENCODER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MAESIObservation:
    """Structured observation produced by the MAESI encoder.

    Represents the complete sensory input to the NeuroAIX connectome
    at a single timestep.
    """

    timestamp: float
    spectral_embedding: np.ndarray  # Primary MESIE embedding
    physical_context: np.ndarray  # Law constraint vector
    chemical_context: np.ndarray  # Element presence vector
    biological_context: np.ndarray  # Biological process vector
    raw_features: Dict[str, float] = field(default_factory=dict)
    source_modality: str = "multi_spectral"
    confidence: float = 1.0


class MAESIObservationEncoder:
    """Multi-Agent Embodied Spectral Intelligence observation encoder.

    Converts raw world data (spectra, signals, measurements) into
    structured observations that the NeuroAIX connectome can process.

    This is the sensory cortex of the system:
        Raw world → spectra → MESIE embedding zₜ → agent observation oₜ

    The encoder supports:
    - Multi-modal concatenation (spectral + physical + chemical + biological)
    - Lineage-conditioned encoding (past context informs current encoding)
    - Physics-constrained embeddings (laws impose structure on representations)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        physical_laws: Optional[List[PhysicalLaw]] = None,
        chemical_elements: Optional[List[SpectralElement]] = None,
        biological_systems: Optional[List[BiologicalSystem]] = None,
    ):
        self.embedding_dim = embedding_dim
        self.laws = physical_laws or get_fundamental_laws()
        self.elements = chemical_elements or get_periodic_table()
        self.bio_systems = biological_systems or get_biological_systems()

        # Pre-compute context matrices
        self._law_matrix = np.stack([l.to_embedding() for l in self.laws])
        self._element_matrix = np.stack([e.to_embedding() for e in self.elements])
        self._bio_matrix = np.stack([b.to_embedding() for b in self.bio_systems])

        # Lineage buffer for temporal conditioning
        self._lineage_buffer: List[np.ndarray] = []
        self._max_lineage = 32

    def encode(
        self,
        spectral_data: np.ndarray,
        timestamp: float = 0.0,
        modality: str = "multi_spectral",
    ) -> MAESIObservation:
        """Encode raw spectral data into a MAESI observation.

        Parameters
        ----------
        spectral_data : np.ndarray
            Raw spectral amplitudes or feature vector.
        timestamp : float
            Current time in seconds.
        modality : str
            Source modality identifier.

        Returns
        -------
        MAESIObservation
            Structured observation ready for connectome injection.
        """
        # Primary spectral embedding via projection
        if len(spectral_data) >= self.embedding_dim:
            embedding = spectral_data[:self.embedding_dim]
        else:
            embedding = np.zeros(self.embedding_dim)
            embedding[:len(spectral_data)] = spectral_data

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Physical context: project onto law space
        physical_context = self._compute_physical_context(embedding)

        # Chemical context: identify active elements
        chemical_context = self._compute_chemical_context(embedding)

        # Biological context: identify active biological processes
        biological_context = self._compute_biological_context(embedding)

        # Update lineage
        self._lineage_buffer.append(embedding.copy())
        if len(self._lineage_buffer) > self._max_lineage:
            self._lineage_buffer.pop(0)

        # Apply lineage conditioning
        if len(self._lineage_buffer) > 1:
            lineage_context = np.mean(self._lineage_buffer[:-1], axis=0)
            # Blend current with temporal context (90% current, 10% history)
            embedding = 0.9 * embedding + 0.1 * lineage_context

        return MAESIObservation(
            timestamp=timestamp,
            spectral_embedding=embedding,
            physical_context=physical_context,
            chemical_context=chemical_context,
            biological_context=biological_context,
            source_modality=modality,
            confidence=float(norm / (norm + 1.0)),
        )

    def _compute_physical_context(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding onto physical law space."""
        # Use first 64 dims of law matrix for dot product similarity
        proj_dim = min(self.embedding_dim, self._law_matrix.shape[1])
        similarities = self._law_matrix[:, :proj_dim] @ embedding[:proj_dim]
        # Softmax activation
        exp_sim = np.exp(similarities - similarities.max())
        return exp_sim / (exp_sim.sum() + 1e-8)

    def _compute_chemical_context(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding onto chemical element space."""
        proj_dim = min(self.embedding_dim, self._element_matrix.shape[1])
        similarities = self._element_matrix[:, :proj_dim] @ embedding[:proj_dim]
        exp_sim = np.exp(similarities - similarities.max())
        return exp_sim / (exp_sim.sum() + 1e-8)

    def _compute_biological_context(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding onto biological system space."""
        proj_dim = min(self.embedding_dim, self._bio_matrix.shape[1])
        similarities = self._bio_matrix[:, :proj_dim] @ embedding[:proj_dim]
        exp_sim = np.exp(similarities - similarities.max())
        return exp_sim / (exp_sim.sum() + 1e-8)

    def reset_lineage(self):
        """Clear the temporal lineage buffer."""
        self._lineage_buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# NEUROAIX ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class NeuroAIXEngine:
    """Neural Architecture for Intelligent eXperience — Core Engine.

    Wraps the ConnectomeEnvironment3D with MAESI-aware injection,
    physics-constrained propagation, and cognitive state extraction.

    This IS the brain. This IS the AI.
    """

    def __init__(
        self,
        connectome: Optional[ConnectomeGraph] = None,
        dt_ms: float = 1.0,
        decay_rate: float = 0.02,
        propagation_gain: float = 0.8,
    ):
        self.connectome = connectome or build_default_connectome()
        self.environment = ConnectomeEnvironment3D(
            connectome=self.connectome,
            dt_ms=dt_ms,
            decay_rate=decay_rate,
            propagation_gain=propagation_gain,
        )
        self.law_registry = SpectralLawRegistry()
        self._step_count = 0
        self._cognitive_history: List[Dict[str, Any]] = []

    def inject_observation(self, observation: MAESIObservation) -> None:
        """Inject a MAESI observation into the connectome.

        Routes different aspects of the observation to appropriate
        brain regions based on modality and content.
        """
        embedding = observation.spectral_embedding

        # Route to sensory regions (occipital for visual, temporal for auditory, etc.)
        sensory_targets = {
            "multi_spectral": ["V1", "V2V3", "STG_L"],
            "visual": ["V1", "V2V3", "FFA"],
            "auditory": ["STG_L", "STG_R", "WER"],
            "somatosensory": ["S1_L", "S1_R"],
            "chemical": ["INS_L", "OFC"],
        }
        targets = sensory_targets.get(observation.source_modality, ["V1", "STG_L"])

        for target in targets:
            amplitude = float(np.mean(np.abs(embedding[:8])))
            self.environment.inject_stimulus(
                region=target,
                amplitude=min(amplitude * 2.0, 1.0),
                frequency_hz=40.0,  # Gamma band for binding
            )

        # Inject physical context into prefrontal regions
        if observation.physical_context is not None:
            physics_activation = float(np.max(observation.physical_context))
            self.environment.inject_stimulus("DLPFC_L", physics_activation * 0.5, 12.0)  # Alpha

        # Inject biological context into insular cortex
        if observation.biological_context is not None:
            bio_activation = float(np.max(observation.biological_context))
            self.environment.inject_stimulus("INS_L", bio_activation * 0.5, 8.0)  # Theta

    def step(self, duration_ms: float = 10.0) -> ActivationState:
        """Advance the connectome simulation by duration_ms.

        Returns the resulting activation state across all 44 brain regions.
        """
        states = self.environment.run(duration_ms=duration_ms)
        self._step_count += 1
        state = states[-1] if states else self.environment.step(1)

        # Record cognitive history
        self._cognitive_history.append({
            "step": self._step_count,
            "coherence": state.global_coherence,
            "dominant_system": state.dominant_system,
            "timestamp": state.timestamp,
        })

        return state

    def get_cognitive_state(self) -> Dict[str, Any]:
        """Extract the current cognitive state for agent policy input."""
        state = self.environment.step(1)
        return {
            "activations": state.activations,
            "coherence": state.global_coherence,
            "dominant_system": state.dominant_system.value if state.dominant_system else None,
            "step": self._step_count,
            "timestamp": state.timestamp,
            "system_activations": {
                system.value: self.environment.get_system_activation(system)
                for system in BrainSystem
            },
        }

    def get_3d_visualization_state(self) -> Dict[str, Any]:
        """Get full 3D state for visualization."""
        return self.environment.get_3d_state()

    @property
    def total_steps(self) -> int:
        """Total simulation steps executed."""
        return self._step_count


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE INTEGRATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CognitiveOutput:
    """Output of a single cognitive integration cycle."""

    state_vector: np.ndarray
    coherence: float
    dominant_system: Optional[str]
    anomaly_score: float
    memory_context: np.ndarray
    timestamp: float


class CognitiveIntegrationLoop:
    """Full perception → integration → memory → action loop.

    Connects:
    1. MAESI encoder (sensory cortex)
    2. NeuroAIX engine (association cortex)
    3. Spectral memory (hippocampal system)
    4. Agent output (motor/prefrontal cortex)

    This is the complete cognitive architecture.
    """

    def __init__(
        self,
        encoder: Optional[MAESIObservationEncoder] = None,
        engine: Optional[NeuroAIXEngine] = None,
        memory_capacity: int = 1000,
        state_dim: int = 128,
    ):
        self.encoder = encoder or MAESIObservationEncoder(embedding_dim=state_dim)
        self.engine = engine or NeuroAIXEngine()
        self.state_dim = state_dim
        self.memory_capacity = memory_capacity

        # Memory store (simplified spectral memory)
        self._memory_embeddings: List[np.ndarray] = []
        self._memory_timestamps: List[float] = []
        self._anomaly_baseline: Optional[np.ndarray] = None
        self._baseline_std: float = 1.0

    def perceive(self, spectral_data: np.ndarray, timestamp: float = 0.0) -> MAESIObservation:
        """Step 1: Encode raw data into structured observation."""
        return self.encoder.encode(spectral_data, timestamp=timestamp)

    def integrate(self, observation: MAESIObservation, propagation_ms: float = 20.0) -> ActivationState:
        """Step 2: Inject observation and propagate through connectome."""
        self.engine.inject_observation(observation)
        return self.engine.step(duration_ms=propagation_ms)

    def remember(self, observation: MAESIObservation) -> np.ndarray:
        """Step 3: Store in memory and retrieve context."""
        # Store
        self._memory_embeddings.append(observation.spectral_embedding.copy())
        self._memory_timestamps.append(observation.timestamp)

        # Consolidate if over capacity
        if len(self._memory_embeddings) > self.memory_capacity:
            self._memory_embeddings.pop(0)
            self._memory_timestamps.pop(0)

        # Retrieve context (k-NN with k=5)
        if len(self._memory_embeddings) > 1:
            query = observation.spectral_embedding
            memories = np.stack(self._memory_embeddings[:-1])
            similarities = memories @ query
            top_k = min(5, len(memories))
            top_indices = np.argsort(similarities)[-top_k:]
            context = np.mean(memories[top_indices], axis=0)
        else:
            context = observation.spectral_embedding.copy()

        return context

    def detect_anomaly(self, observation: MAESIObservation) -> float:
        """Compute anomaly score relative to baseline."""
        if self._anomaly_baseline is None:
            if len(self._memory_embeddings) >= 10:
                self._anomaly_baseline = np.mean(self._memory_embeddings[:10], axis=0)
                self._baseline_std = float(np.std(self._memory_embeddings[:10])) + 1e-8
            return 0.0

        deviation = np.linalg.norm(observation.spectral_embedding - self._anomaly_baseline)
        return float(deviation / self._baseline_std)

    def cycle(self, spectral_data: np.ndarray, timestamp: float = 0.0) -> CognitiveOutput:
        """Execute one full cognitive cycle.

        Raw data → perceive → integrate → remember → output

        Parameters
        ----------
        spectral_data : np.ndarray
            Raw spectral input.
        timestamp : float
            Current time.

        Returns
        -------
        CognitiveOutput
            Complete cognitive state for agent decision-making.
        """
        # Perceive
        observation = self.perceive(spectral_data, timestamp)

        # Integrate
        activation_state = self.integrate(observation)

        # Remember
        memory_context = self.remember(observation)

        # Detect anomalies
        anomaly_score = self.detect_anomaly(observation)

        # Build state vector (concatenate key representations)
        cognitive_state = self.engine.get_cognitive_state()
        activations_vec = np.array(list(cognitive_state["activations"].values()))

        # Construct output state vector
        state_vector = np.zeros(self.state_dim)
        n_act = min(len(activations_vec), self.state_dim // 2)
        state_vector[:n_act] = activations_vec[:n_act]
        n_emb = min(len(observation.spectral_embedding), self.state_dim - n_act)
        state_vector[n_act:n_act + n_emb] = observation.spectral_embedding[:n_emb]

        return CognitiveOutput(
            state_vector=state_vector,
            coherence=activation_state.global_coherence,
            dominant_system=activation_state.dominant_system.value if activation_state.dominant_system else None,
            anomaly_score=anomaly_score,
            memory_context=memory_context,
            timestamp=timestamp,
        )

    def reset(self):
        """Reset the cognitive loop to initial state."""
        self.encoder.reset_lineage()
        self._memory_embeddings.clear()
        self._memory_timestamps.clear()
        self._anomaly_baseline = None
        self._baseline_std = 1.0
