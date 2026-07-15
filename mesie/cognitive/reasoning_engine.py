"""Spectral Reasoning Engine — Causal Inference and Deep Reasoning for Spectral Data.

Provides a multi-layer reasoning system that combines causal inference,
Bayesian updating, abductive reasoning, and counterfactual analysis
for deep spectral intelligence. Integrates with TAURUS for memory-augmented
reasoning and NeuroCores for attention-driven focus.

Architecture:
    ReasoningChain → CausalGraph → BayesianUpdater → AbductiveReasoner
         ↓              ↓               ↓                  ↓
    CounterfactualEngine → HypothesisGenerator → EvidenceAccumulator
         ↓
    ReasoningResult with full provenance
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Enumerations and Type Definitions
# =============================================================================


class ReasoningMode(Enum):
    """Modes of reasoning available in the engine."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"


class EvidenceType(Enum):
    """Types of evidence that can be accumulated."""

    SPECTRAL_PATTERN = "spectral_pattern"
    TEMPORAL_CORRELATION = "temporal_correlation"
    CROSS_BAND_RELATIONSHIP = "cross_band_relationship"
    HARMONIC_STRUCTURE = "harmonic_structure"
    ANOMALY_SIGNATURE = "anomaly_signature"
    MEMORY_RECALL = "memory_recall"
    STATISTICAL = "statistical"
    CAUSAL_LINK = "causal_link"


class ConfidenceLevel(Enum):
    """Discrete confidence levels for reasoning conclusions."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CERTAIN = "certain"

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert a numeric confidence score to a level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.HIGH
        elif score < 0.95:
            return cls.VERY_HIGH
        else:
            return cls.CERTAIN


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class Evidence:
    """A piece of evidence supporting or contradicting a hypothesis.

    Args:
        evidence_type: Category of evidence.
        description: Human-readable description.
        strength: How strongly this evidence supports (positive) or contradicts (negative).
        source: Where this evidence came from.
        timestamp: When this evidence was gathered.
        spectral_signature: Optional spectral data associated with evidence.
        metadata: Additional metadata.
    """

    evidence_type: EvidenceType
    description: str
    strength: float  # [-1, 1] where positive supports, negative contradicts
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    spectral_signature: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_supporting(self) -> bool:
        """Whether this evidence supports the hypothesis."""
        return self.strength > 0.0

    @property
    def absolute_strength(self) -> float:
        """Absolute strength regardless of direction."""
        return abs(self.strength)


@dataclass
class Hypothesis:
    """A hypothesis about spectral data behavior.

    Args:
        hypothesis_id: Unique identifier.
        description: Human-readable hypothesis statement.
        prior_probability: Initial probability before evidence.
        evidence: List of evidence for/against.
        alternatives: Alternative hypotheses considered.
        reasoning_mode: Mode used to generate this hypothesis.
        metadata: Additional metadata.
    """

    hypothesis_id: str
    description: str
    prior_probability: float = 0.5
    evidence: List[Evidence] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    reasoning_mode: ReasoningMode = ReasoningMode.ABDUCTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    _posterior: Optional[float] = field(default=None, repr=False)

    @property
    def posterior_probability(self) -> float:
        """Compute posterior probability given accumulated evidence."""
        if self._posterior is not None:
            return self._posterior

        if not self.evidence:
            return self.prior_probability

        # Log-odds Bayesian update
        log_odds = np.log(self.prior_probability / (1 - self.prior_probability + 1e-12) + 1e-12)

        for e in self.evidence:
            # Convert evidence strength to likelihood ratio
            if e.strength > 0:
                lr = 1.0 + e.strength * 3.0  # Supporting evidence
            else:
                lr = 1.0 / (1.0 + abs(e.strength) * 3.0)  # Contradicting evidence
            log_odds += np.log(lr + 1e-12)

        # Convert back to probability
        posterior = 1.0 / (1.0 + np.exp(-log_odds))
        return float(np.clip(posterior, 0.001, 0.999))

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Discrete confidence level for this hypothesis."""
        return ConfidenceLevel.from_score(self.posterior_probability)

    @property
    def evidence_weight(self) -> float:
        """Total weight of all evidence."""
        return sum(e.absolute_strength for e in self.evidence)


@dataclass
class CausalLink:
    """A causal relationship between spectral phenomena.

    Args:
        cause: Description of the cause.
        effect: Description of the effect.
        strength: Causal strength (0-1).
        mechanism: Description of the causal mechanism.
        delay: Time delay between cause and effect.
        confidence: Confidence in this causal link.
        evidence: Supporting evidence.
    """

    cause: str
    effect: str
    strength: float
    mechanism: str = ""
    delay: float = 0.0
    confidence: float = 0.5
    evidence: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_strong(self) -> bool:
        """Whether this is a strong causal link."""
        return self.strength > 0.7 and self.confidence > 0.6


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain.

    Args:
        step_id: Sequential step identifier.
        mode: Reasoning mode used in this step.
        input_state: Input to this reasoning step.
        output_state: Output from this reasoning step.
        rationale: Explanation of the reasoning.
        confidence: Confidence in this step.
        duration: Time taken for this step (seconds).
    """

    step_id: int
    mode: ReasoningMode
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    rationale: str
    confidence: float
    duration: float = 0.0


@dataclass
class ReasoningChainResult:
    """Complete result from a reasoning chain execution.

    Args:
        chain_id: Unique identifier for this chain.
        steps: All reasoning steps executed.
        final_conclusion: The ultimate conclusion.
        overall_confidence: Aggregate confidence.
        hypotheses: All hypotheses considered.
        causal_links: Discovered causal relationships.
        total_duration: Total reasoning time.
        metadata: Additional metadata.
    """

    chain_id: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_conclusion: str = ""
    overall_confidence: float = 0.0
    hypotheses: List[Hypothesis] = field(default_factory=list)
    causal_links: List[CausalLink] = field(default_factory=list)
    total_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of reasoning steps."""
        return len(self.steps)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Discrete confidence level."""
        return ConfidenceLevel.from_score(self.overall_confidence)


# =============================================================================
# Causal Graph
# =============================================================================


class CausalGraph:
    """Directed acyclic graph representing causal relationships in spectral data.

    Maintains a graph of causal relationships between spectral phenomena,
    supporting causal discovery, intervention analysis, and counterfactual
    reasoning.

    Args:
        max_nodes: Maximum number of nodes in the graph.
    """

    def __init__(self, max_nodes: int = 200) -> None:
        self.max_nodes = max_nodes
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[CausalLink] = []
        self._adjacency: Dict[str, List[str]] = {}

    def add_node(
        self,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a causal node to the graph.

        Args:
            node_id: Unique node identifier.
            properties: Node properties/metadata.
        """
        if len(self._nodes) >= self.max_nodes:
            self._evict_weakest_node()

        self._nodes[node_id] = properties or {}
        if node_id not in self._adjacency:
            self._adjacency[node_id] = []

    def add_edge(self, link: CausalLink) -> None:
        """Add a causal edge to the graph.

        Args:
            link: CausalLink representing the directed edge.
        """
        # Ensure nodes exist
        if link.cause not in self._nodes:
            self.add_node(link.cause)
        if link.effect not in self._nodes:
            self.add_node(link.effect)

        # Check for cycles
        if not self._would_create_cycle(link.cause, link.effect):
            self._edges.append(link)
            self._adjacency[link.cause].append(link.effect)

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding an edge would create a cycle."""
        if source == target:
            return True

        # BFS from target to see if we can reach source
        visited = set()
        queue = [target]
        while queue:
            current = queue.pop(0)
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._adjacency.get(current, []))

        return False

    def get_causes(self, node_id: str) -> List[CausalLink]:
        """Get all direct causes of a node.

        Args:
            node_id: Target node.

        Returns:
            List of causal links where this node is the effect.
        """
        return [e for e in self._edges if e.effect == node_id]

    def get_effects(self, node_id: str) -> List[CausalLink]:
        """Get all direct effects of a node.

        Args:
            node_id: Source node.

        Returns:
            List of causal links where this node is the cause.
        """
        return [e for e in self._edges if e.cause == node_id]

    def get_causal_chain(self, start: str, end: str) -> List[CausalLink]:
        """Find the causal chain (path) from start to end.

        Args:
            start: Starting node.
            end: Ending node.

        Returns:
            List of causal links forming the path, or empty if no path.
        """
        if start not in self._nodes or end not in self._nodes:
            return []

        # BFS for shortest causal path
        visited = set()
        queue: List[Tuple[str, List[CausalLink]]] = [(start, [])]

        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)

            for edge in self._edges:
                if edge.cause == current and edge.effect not in visited:
                    queue.append((edge.effect, path + [edge]))

        return []

    def compute_causal_strength(self, start: str, end: str) -> float:
        """Compute aggregate causal strength along the path.

        Args:
            start: Starting node.
            end: Ending node.

        Returns:
            Product of edge strengths along the path (0 if no path).
        """
        chain = self.get_causal_chain(start, end)
        if not chain:
            return 0.0
        return float(np.prod([link.strength for link in chain]))

    def get_root_causes(self) -> List[str]:
        """Find nodes with no incoming edges (root causes)."""
        effects = {e.effect for e in self._edges}
        return [n for n in self._nodes if n not in effects]

    def get_terminal_effects(self) -> List[str]:
        """Find nodes with no outgoing edges (terminal effects)."""
        causes = {e.cause for e in self._edges}
        return [n for n in self._nodes if n not in causes]

    def intervene(self, node_id: str, value: Any) -> Dict[str, float]:
        """Simulate an intervention (do-calculus) on a node.

        Cuts all incoming edges to the node and propagates effects downstream.

        Args:
            node_id: Node to intervene on.
            value: Intervention value.

        Returns:
            Dictionary of downstream effects with predicted magnitudes.
        """
        if node_id not in self._nodes:
            return {}

        effects: Dict[str, float] = {}
        visited = set()
        queue = [(node_id, 1.0)]

        while queue:
            current, accumulated_strength = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for edge in self._edges:
                if edge.cause == current:
                    effect_magnitude = accumulated_strength * edge.strength
                    effects[edge.effect] = effect_magnitude
                    queue.append((edge.effect, effect_magnitude))

        return effects

    def _evict_weakest_node(self) -> None:
        """Remove the node with weakest causal connections."""
        if not self._nodes:
            return

        # Score nodes by total edge strength
        scores: Dict[str, float] = {n: 0.0 for n in self._nodes}
        for edge in self._edges:
            scores[edge.cause] = scores.get(edge.cause, 0.0) + edge.strength
            scores[edge.effect] = scores.get(edge.effect, 0.0) + edge.strength

        # Remove weakest
        weakest = min(scores, key=lambda k: scores[k])
        self.remove_node(weakest)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        self._nodes.pop(node_id, None)
        self._edges = [e for e in self._edges if e.cause != node_id and e.effect != node_id]
        self._adjacency.pop(node_id, None)
        for adj_list in self._adjacency.values():
            while node_id in adj_list:
                adj_list.remove(node_id)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the causal graph to a dictionary."""
        return {
            "nodes": list(self._nodes.keys()),
            "edges": [
                {
                    "cause": e.cause,
                    "effect": e.effect,
                    "strength": e.strength,
                    "mechanism": e.mechanism,
                }
                for e in self._edges
            ],
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
        }


# =============================================================================
# Bayesian Updater
# =============================================================================


class BayesianUpdater:
    """Bayesian belief updating for spectral hypotheses.

    Maintains a belief distribution over hypotheses and updates
    it as new evidence arrives. Supports multiple prior types
    and likelihood models.

    Args:
        prior_type: Type of prior distribution ('uniform', 'informative', 'jeffreys').
        learning_rate: Rate of belief update (0-1).
        evidence_decay: Temporal decay rate for old evidence.
    """

    def __init__(
        self,
        prior_type: str = "uniform",
        learning_rate: float = 1.0,
        evidence_decay: float = 0.01,
    ) -> None:
        self.prior_type = prior_type
        self.learning_rate = learning_rate
        self.evidence_decay = evidence_decay
        self._beliefs: Dict[str, float] = {}
        self._evidence_history: List[Tuple[str, Evidence]] = []
        self._update_count: int = 0

    def initialize_beliefs(self, hypotheses: List[str]) -> None:
        """Initialize belief distribution over hypotheses.

        Args:
            hypotheses: List of hypothesis identifiers.
        """
        n = len(hypotheses)
        if n == 0:
            return

        if self.prior_type == "uniform":
            prior = 1.0 / n
            self._beliefs = {h: prior for h in hypotheses}
        elif self.prior_type == "jeffreys":
            # Jeffreys prior (improper, normalized)
            self._beliefs = {h: 1.0 / (n * np.log(n + 1)) for h in hypotheses}
            # Normalize
            total = sum(self._beliefs.values())
            self._beliefs = {h: v / total for h, v in self._beliefs.items()}
        else:
            self._beliefs = {h: 1.0 / n for h in hypotheses}

    def update(self, hypothesis_id: str, evidence: Evidence) -> float:
        """Update belief in a hypothesis given new evidence.

        Uses Bayesian updating with configurable learning rate.

        Args:
            hypothesis_id: Hypothesis to update.
            evidence: New evidence.

        Returns:
            Updated posterior probability.
        """
        if hypothesis_id not in self._beliefs:
            self._beliefs[hypothesis_id] = 0.5

        self._update_count += 1
        self._evidence_history.append((hypothesis_id, evidence))

        # Compute likelihood ratio from evidence strength
        if evidence.strength > 0:
            likelihood_ratio = 1.0 + evidence.strength * self.learning_rate * 5.0
        else:
            likelihood_ratio = 1.0 / (1.0 + abs(evidence.strength) * self.learning_rate * 5.0)

        # Bayesian update
        prior = self._beliefs[hypothesis_id]
        unnormalized_posterior = prior * likelihood_ratio

        # Update all beliefs to maintain normalization
        other_mass = sum(v for k, v in self._beliefs.items() if k != hypothesis_id)
        total = unnormalized_posterior + other_mass
        self._beliefs[hypothesis_id] = unnormalized_posterior / (total + 1e-12)

        # Normalize others proportionally
        if other_mass > 0:
            scale = (1.0 - self._beliefs[hypothesis_id]) / other_mass
            for k in self._beliefs:
                if k != hypothesis_id:
                    self._beliefs[k] *= scale

        return self._beliefs[hypothesis_id]

    def batch_update(
        self,
        evidence_pairs: List[Tuple[str, Evidence]],
    ) -> Dict[str, float]:
        """Update beliefs with multiple pieces of evidence.

        Args:
            evidence_pairs: List of (hypothesis_id, evidence) tuples.

        Returns:
            Updated belief distribution.
        """
        for hypothesis_id, evidence in evidence_pairs:
            self.update(hypothesis_id, evidence)
        return self._beliefs.copy()

    def decay_evidence(self, current_time: Optional[float] = None) -> None:
        """Apply temporal decay to past evidence influence.

        Args:
            current_time: Reference time for decay calculation.
        """
        now = current_time or time.time()

        # Reset beliefs to uniform and replay with decayed evidence
        hypotheses = list(self._beliefs.keys())
        self.initialize_beliefs(hypotheses)

        for hypothesis_id, evidence in self._evidence_history:
            age = max(0.0, now - evidence.timestamp)
            decay_factor = np.exp(-self.evidence_decay * age)
            decayed_evidence = Evidence(
                evidence_type=evidence.evidence_type,
                description=evidence.description,
                strength=evidence.strength * decay_factor,
                source=evidence.source,
                timestamp=evidence.timestamp,
            )
            self.update(hypothesis_id, decayed_evidence)

    def get_belief(self, hypothesis_id: str) -> float:
        """Get current belief in a hypothesis.

        Args:
            hypothesis_id: Hypothesis to query.

        Returns:
            Current posterior probability.
        """
        return self._beliefs.get(hypothesis_id, 0.0)

    def get_top_hypotheses(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get the most probable hypotheses.

        Args:
            top_k: Number of top hypotheses to return.

        Returns:
            List of (hypothesis_id, probability) sorted by probability.
        """
        sorted_beliefs = sorted(
            self._beliefs.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_beliefs[:top_k]

    def get_entropy(self) -> float:
        """Compute entropy of the belief distribution.

        Returns:
            Shannon entropy (higher = more uncertain).
        """
        if not self._beliefs:
            return 0.0
        probs = np.array(list(self._beliefs.values()))
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    @property
    def beliefs(self) -> Dict[str, float]:
        """Current belief distribution."""
        return self._beliefs.copy()

    @property
    def update_count(self) -> int:
        """Total number of updates performed."""
        return self._update_count


# =============================================================================
# Abductive Reasoner
# =============================================================================


class AbductiveReasoner:
    """Inference to the best explanation for spectral observations.

    Generates and evaluates candidate explanations for observed
    spectral patterns, ranking them by explanatory power and
    parsimony.

    Args:
        max_hypotheses: Maximum number of hypotheses to maintain.
        parsimony_weight: Weight given to simpler explanations.
        coverage_weight: Weight given to explanatory coverage.
    """

    def __init__(
        self,
        max_hypotheses: int = 20,
        parsimony_weight: float = 0.3,
        coverage_weight: float = 0.7,
    ) -> None:
        self.max_hypotheses = max_hypotheses
        self.parsimony_weight = parsimony_weight
        self.coverage_weight = coverage_weight
        self._hypotheses: List[Hypothesis] = []
        self._observations: List[Dict[str, Any]] = []
        self._explanation_patterns: Dict[str, List[str]] = {
            "peak_shift": [
                "frequency_drift",
                "temperature_change",
                "material_degradation",
                "loading_change",
            ],
            "amplitude_increase": [
                "resonance_approach",
                "excitation_increase",
                "damping_decrease",
                "structural_loosening",
            ],
            "amplitude_decrease": [
                "damping_increase",
                "excitation_decrease",
                "energy_dissipation",
                "structural_stiffening",
            ],
            "new_peak": [
                "new_excitation_source",
                "structural_crack",
                "harmonic_generation",
                "mode_coupling",
            ],
            "peak_disappearance": [
                "mode_suppression",
                "frequency_shift_out_of_band",
                "damping_increase",
                "structural_change",
            ],
            "broadband_increase": [
                "noise_increase",
                "turbulence",
                "distributed_damage",
                "environmental_change",
            ],
            "harmonic_series": [
                "nonlinear_response",
                "rotating_machinery",
                "periodic_impact",
                "resonance_cascade",
            ],
        }

    def observe(self, observation: Dict[str, Any]) -> None:
        """Record a spectral observation for explanation.

        Args:
            observation: Dictionary with observation data including
                         'pattern_type', 'magnitude', 'location', etc.
        """
        self._observations.append(observation)

    def generate_hypotheses(
        self,
        observation: Dict[str, Any],
    ) -> List[Hypothesis]:
        """Generate candidate hypotheses for an observation.

        Args:
            observation: The observation to explain.

        Returns:
            List of candidate hypotheses.
        """
        pattern_type = observation.get("pattern_type", "unknown")
        magnitude = observation.get("magnitude", 1.0)

        # Get relevant explanations
        explanations = self._explanation_patterns.get(
            pattern_type,
            ["unknown_phenomenon", "measurement_artifact", "environmental_noise"],
        )

        hypotheses = []
        for i, explanation in enumerate(explanations):
            # Prior based on common frequency of this explanation
            prior = max(0.1, 1.0 - 0.15 * i)  # Decreasing prior for less common

            hypothesis = Hypothesis(
                hypothesis_id=f"h_{pattern_type}_{i}",
                description=explanation,
                prior_probability=prior / len(explanations),
                reasoning_mode=ReasoningMode.ABDUCTIVE,
                metadata={
                    "pattern_type": pattern_type,
                    "magnitude": magnitude,
                    "generation_order": i,
                },
            )
            hypotheses.append(hypothesis)

        # Keep within limits
        self._hypotheses.extend(hypotheses)
        if len(self._hypotheses) > self.max_hypotheses:
            # Keep highest posterior hypotheses
            self._hypotheses.sort(
                key=lambda h: h.posterior_probability, reverse=True
            )
            self._hypotheses = self._hypotheses[: self.max_hypotheses]

        return hypotheses

    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        test_data: np.ndarray,
    ) -> float:
        """Evaluate how well a hypothesis explains the data.

        Args:
            hypothesis: Hypothesis to evaluate.
            test_data: Spectral data to explain.

        Returns:
            Explanatory score (0-1).
        """
        # Compute data characteristics
        data_energy = float(np.sum(test_data ** 2))
        data_peaks = int(np.sum(
            (test_data[1:-1] > test_data[:-2]) & (test_data[1:-1] > test_data[2:])
        )) if len(test_data) > 2 else 0
        data_std = float(np.std(test_data))
        data_mean = float(np.mean(test_data))

        # Score based on hypothesis type vs data characteristics
        description = hypothesis.description
        score = 0.5  # Base score

        if "resonance" in description and data_peaks > 0:
            score += 0.2 * min(1.0, data_peaks / 5.0)
        if "noise" in description and data_std > data_mean:
            score += 0.15
        if "harmonic" in description and data_peaks > 2:
            score += 0.15 * min(1.0, data_peaks / 10.0)
        if "damping" in description and data_energy < 1.0:
            score += 0.1
        if "excitation" in description and data_energy > 10.0:
            score += 0.1

        # Parsimony bonus for simpler explanations
        complexity = len(description.split("_"))
        parsimony_bonus = self.parsimony_weight * (1.0 / complexity)
        score += parsimony_bonus

        return float(np.clip(score, 0.0, 1.0))

    def rank_hypotheses(self, test_data: np.ndarray) -> List[Tuple[Hypothesis, float]]:
        """Rank all hypotheses by explanatory power.

        Args:
            test_data: Data to explain.

        Returns:
            List of (hypothesis, score) sorted by score descending.
        """
        scored = [
            (h, self.evaluate_hypothesis(h, test_data))
            for h in self._hypotheses
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def best_explanation(self, test_data: np.ndarray) -> Optional[Hypothesis]:
        """Get the best explanation for the data.

        Args:
            test_data: Data to explain.

        Returns:
            Best hypothesis or None if no hypotheses exist.
        """
        ranked = self.rank_hypotheses(test_data)
        return ranked[0][0] if ranked else None

    @property
    def n_hypotheses(self) -> int:
        """Number of active hypotheses."""
        return len(self._hypotheses)

    @property
    def n_observations(self) -> int:
        """Number of recorded observations."""
        return len(self._observations)


# =============================================================================
# Counterfactual Engine
# =============================================================================


class CounterfactualEngine:
    """Counterfactual reasoning for spectral scenarios.

    Enables "what-if" analysis by simulating alternative spectral
    outcomes under different conditions. Useful for causal reasoning
    and decision support.

    Args:
        simulation_resolution: Number of points in simulated spectra.
        noise_level: Background noise level for simulations.
    """

    def __init__(
        self,
        simulation_resolution: int = 256,
        noise_level: float = 0.01,
    ) -> None:
        self.simulation_resolution = simulation_resolution
        self.noise_level = noise_level
        self._scenarios: List[Dict[str, Any]] = []
        self._results: List[Dict[str, Any]] = []

    def create_counterfactual(
        self,
        observed_spectrum: np.ndarray,
        intervention: str,
        intervention_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Generate a counterfactual spectrum under an intervention.

        Args:
            observed_spectrum: The actual observed spectrum.
            intervention: Type of intervention to simulate.
            intervention_params: Parameters for the intervention.

        Returns:
            Simulated counterfactual spectrum.
        """
        params = intervention_params or {}
        spectrum = observed_spectrum.copy()

        if intervention == "remove_peak":
            # Simulate removal of a spectral peak
            center = int(params.get("center", len(spectrum) // 2))
            width = int(params.get("width", 10))
            start = max(0, center - width)
            end = min(len(spectrum), center + width)
            # Interpolate across the peak
            if start > 0 and end < len(spectrum):
                spectrum[start:end] = np.linspace(
                    spectrum[max(0, start - 1)],
                    spectrum[min(len(spectrum) - 1, end)],
                    end - start,
                )

        elif intervention == "add_damping":
            # Simulate increased damping
            damping_factor = params.get("factor", 0.5)
            spectrum *= damping_factor

        elif intervention == "shift_frequency":
            # Simulate frequency shift
            shift = int(params.get("shift", 5))
            if shift > 0:
                spectrum = np.concatenate([np.zeros(shift), spectrum[:-shift]])
            elif shift < 0:
                spectrum = np.concatenate([spectrum[-shift:], np.zeros(-shift)])

        elif intervention == "add_harmonic":
            # Simulate appearance of a new harmonic
            fundamental = params.get("fundamental", 10.0)
            amplitude = params.get("amplitude", 0.5)
            x = np.arange(len(spectrum))
            harmonic = amplitude * np.sin(2 * np.pi * fundamental * x / len(spectrum))
            spectrum += harmonic

        elif intervention == "increase_noise":
            # Simulate noise increase
            noise_factor = params.get("factor", 2.0)
            noise = np.random.randn(len(spectrum)) * self.noise_level * noise_factor
            spectrum += noise

        elif intervention == "temperature_change":
            # Simulate temperature-induced frequency shift
            delta_t = params.get("delta_t", 10.0)
            shift_rate = params.get("shift_rate", 0.001)
            shift = int(delta_t * shift_rate * len(spectrum))
            if shift > 0:
                spectrum = np.concatenate([np.zeros(shift), spectrum[:-shift]])

        elif intervention == "structural_damage":
            # Simulate structural damage (new peaks, reduced stiffness)
            damage_severity = params.get("severity", 0.3)
            # Reduce overall amplitude
            spectrum *= (1.0 - damage_severity * 0.5)
            # Add sub-harmonic peaks
            n_new_peaks = int(damage_severity * 5)
            for _ in range(n_new_peaks):
                pos = np.random.randint(0, len(spectrum))
                spectrum[pos] += damage_severity * np.max(spectrum)

        # Add simulation noise
        spectrum += np.random.randn(len(spectrum)) * self.noise_level

        # Record scenario
        self._scenarios.append({
            "intervention": intervention,
            "params": params,
            "original_energy": float(np.sum(observed_spectrum ** 2)),
            "counterfactual_energy": float(np.sum(spectrum ** 2)),
        })

        return spectrum

    def compare_outcomes(
        self,
        observed: np.ndarray,
        counterfactual: np.ndarray,
    ) -> Dict[str, float]:
        """Compare observed outcome with counterfactual.

        Args:
            observed: Actually observed spectrum.
            counterfactual: Simulated counterfactual spectrum.

        Returns:
            Dictionary of comparison metrics.
        """
        # Ensure same length
        min_len = min(len(observed), len(counterfactual))
        obs = observed[:min_len]
        cf = counterfactual[:min_len]

        # Metrics
        mse = float(np.mean((obs - cf) ** 2))
        correlation = float(np.corrcoef(obs, cf)[0, 1]) if min_len > 1 else 0.0
        energy_ratio = float(np.sum(cf ** 2) / (np.sum(obs ** 2) + 1e-12))
        max_deviation = float(np.max(np.abs(obs - cf)))

        # Spectral distance
        obs_norm = obs / (np.linalg.norm(obs) + 1e-12)
        cf_norm = cf / (np.linalg.norm(cf) + 1e-12)
        cosine_similarity = float(np.dot(obs_norm, cf_norm))

        return {
            "mse": mse,
            "correlation": correlation,
            "energy_ratio": energy_ratio,
            "max_deviation": max_deviation,
            "cosine_similarity": cosine_similarity,
            "causal_effect_size": 1.0 - cosine_similarity,
        }

    def assess_causal_effect(
        self,
        observed: np.ndarray,
        intervention: str,
        intervention_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Assess the causal effect of an intervention.

        Args:
            observed: Observed spectrum.
            intervention: Type of intervention.
            intervention_params: Intervention parameters.

        Returns:
            Assessment dictionary with effect size and interpretation.
        """
        counterfactual = self.create_counterfactual(
            observed, intervention, intervention_params
        )
        comparison = self.compare_outcomes(observed, counterfactual)

        effect_size = comparison["causal_effect_size"]
        if effect_size < 0.1:
            interpretation = "negligible_effect"
        elif effect_size < 0.3:
            interpretation = "small_effect"
        elif effect_size < 0.5:
            interpretation = "moderate_effect"
        elif effect_size < 0.7:
            interpretation = "large_effect"
        else:
            interpretation = "very_large_effect"

        result = {
            "intervention": intervention,
            "effect_size": effect_size,
            "interpretation": interpretation,
            "comparison_metrics": comparison,
            "counterfactual_spectrum": counterfactual,
        }
        self._results.append(result)
        return result

    @property
    def n_scenarios(self) -> int:
        """Number of counterfactual scenarios simulated."""
        return len(self._scenarios)


# =============================================================================
# Evidence Accumulator
# =============================================================================


class EvidenceAccumulator:
    """Drift-diffusion evidence accumulation for spectral decisions.

    Implements a sequential evidence accumulation model where evidence
    is collected over time until a decision threshold is reached.
    Based on drift-diffusion models from cognitive neuroscience.

    Args:
        threshold: Evidence threshold for decision (positive = accept, negative = reject).
        drift_rate: Base rate of evidence accumulation.
        noise_std: Standard deviation of accumulation noise.
        max_steps: Maximum steps before forced decision.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        drift_rate: float = 0.1,
        noise_std: float = 0.5,
        max_steps: int = 100,
    ) -> None:
        self.threshold = threshold
        self.drift_rate = drift_rate
        self.noise_std = noise_std
        self.max_steps = max_steps
        self._accumulated_evidence: float = 0.0
        self._history: List[float] = [0.0]
        self._step_count: int = 0
        self._decision_made: bool = False
        self._decision: Optional[str] = None

    def accumulate(self, evidence: Evidence) -> Optional[str]:
        """Accumulate a piece of evidence toward a decision.

        Args:
            evidence: New evidence to incorporate.

        Returns:
            Decision string if threshold reached, None otherwise.
        """
        if self._decision_made:
            return self._decision

        self._step_count += 1

        # Evidence contributes based on strength and type
        contribution = evidence.strength * self.drift_rate
        noise = np.random.randn() * self.noise_std * 0.1
        self._accumulated_evidence += contribution + noise
        self._history.append(self._accumulated_evidence)

        # Check thresholds
        if self._accumulated_evidence >= self.threshold:
            self._decision_made = True
            self._decision = "accept"
            return "accept"
        elif self._accumulated_evidence <= -self.threshold:
            self._decision_made = True
            self._decision = "reject"
            return "reject"
        elif self._step_count >= self.max_steps:
            self._decision_made = True
            self._decision = "accept" if self._accumulated_evidence > 0 else "reject"
            return self._decision

        return None

    def batch_accumulate(self, evidence_list: List[Evidence]) -> Optional[str]:
        """Accumulate multiple pieces of evidence.

        Args:
            evidence_list: List of evidence to process sequentially.

        Returns:
            Decision if reached, None otherwise.
        """
        for evidence in evidence_list:
            decision = self.accumulate(evidence)
            if decision is not None:
                return decision
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get current accumulator state.

        Returns:
            Dictionary with current state information.
        """
        return {
            "accumulated_evidence": self._accumulated_evidence,
            "step_count": self._step_count,
            "threshold": self.threshold,
            "decision_made": self._decision_made,
            "decision": self._decision,
            "progress_to_accept": self._accumulated_evidence / self.threshold,
            "progress_to_reject": -self._accumulated_evidence / self.threshold,
            "history": self._history[-20:],  # Last 20 steps
        }

    def reset(self) -> None:
        """Reset the accumulator for a new decision."""
        self._accumulated_evidence = 0.0
        self._history = [0.0]
        self._step_count = 0
        self._decision_made = False
        self._decision = None

    @property
    def is_decided(self) -> bool:
        """Whether a decision has been reached."""
        return self._decision_made

    @property
    def current_evidence(self) -> float:
        """Current accumulated evidence level."""
        return self._accumulated_evidence


# =============================================================================
# Spectral Pattern Recognizer
# =============================================================================


class SpectralPatternRecognizer:
    """Pattern recognition system for spectral data.

    Identifies common spectral patterns (peaks, harmonics, broadband,
    transients, etc.) and provides structured descriptions for the
    reasoning engine.

    Args:
        min_peak_prominence: Minimum prominence for peak detection.
        harmonic_tolerance: Frequency ratio tolerance for harmonic detection.
        n_frequency_bands: Number of frequency bands for analysis.
    """

    def __init__(
        self,
        min_peak_prominence: float = 1.0,
        harmonic_tolerance: float = 0.05,
        n_frequency_bands: int = 8,
    ) -> None:
        self.min_peak_prominence = min_peak_prominence
        self.harmonic_tolerance = harmonic_tolerance
        self.n_frequency_bands = n_frequency_bands

    def detect_peaks(self, spectrum: np.ndarray) -> List[Dict[str, float]]:
        """Detect spectral peaks with properties.

        Args:
            spectrum: Input spectrum.

        Returns:
            List of peak dictionaries with index, amplitude, prominence, width.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        peaks = []

        if len(spectrum) < 3:
            return peaks

        # Local maxima
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                # Compute prominence
                left_min = np.min(spectrum[max(0, i - 20):i])
                right_min = np.min(spectrum[i + 1:min(len(spectrum), i + 21)])
                prominence = spectrum[i] - max(left_min, right_min)

                if prominence >= self.min_peak_prominence:
                    # Estimate width (half-prominence width)
                    half_height = spectrum[i] - prominence / 2
                    width_left = 0
                    width_right = 0
                    for j in range(i - 1, max(0, i - 50), -1):
                        if spectrum[j] < half_height:
                            width_left = i - j
                            break
                    for j in range(i + 1, min(len(spectrum), i + 50)):
                        if spectrum[j] < half_height:
                            width_right = j - i
                            break

                    peaks.append({
                        "index": float(i),
                        "frequency_bin": float(i) / len(spectrum),
                        "amplitude": float(spectrum[i]),
                        "prominence": float(prominence),
                        "width": float(width_left + width_right),
                        "sharpness": float(prominence / (width_left + width_right + 1)),
                    })

        return peaks

    def detect_harmonics(
        self,
        spectrum: np.ndarray,
        peaks: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect harmonic series in the spectrum.

        Args:
            spectrum: Input spectrum.
            peaks: Pre-detected peaks (optional).

        Returns:
            List of harmonic series found.
        """
        if peaks is None:
            peaks = self.detect_peaks(spectrum)

        if not peaks:
            return []

        harmonics = []
        peak_indices = [p["index"] for p in peaks]

        for i, fundamental in enumerate(peak_indices):
            if fundamental < 1:
                continue

            series = [fundamental]
            for harmonic_n in range(2, 11):
                expected = fundamental * harmonic_n
                # Find closest peak
                for other_idx in peak_indices:
                    if abs(other_idx - expected) / expected < self.harmonic_tolerance:
                        series.append(other_idx)
                        break

            if len(series) >= 3:
                harmonics.append({
                    "fundamental_index": fundamental,
                    "fundamental_freq": fundamental / len(spectrum),
                    "n_harmonics": len(series),
                    "harmonic_indices": series,
                    "completeness": len(series) / 10.0,
                })

        return harmonics

    def analyze_bands(self, spectrum: np.ndarray) -> List[Dict[str, float]]:
        """Analyze spectral energy distribution across frequency bands.

        Args:
            spectrum: Input spectrum.

        Returns:
            List of band analysis dictionaries.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        band_size = len(spectrum) // self.n_frequency_bands
        bands = []

        for i in range(self.n_frequency_bands):
            start = i * band_size
            end = start + band_size if i < self.n_frequency_bands - 1 else len(spectrum)
            band_data = spectrum[start:end]

            bands.append({
                "band_index": float(i),
                "start_bin": float(start),
                "end_bin": float(end),
                "mean_amplitude": float(np.mean(band_data)),
                "max_amplitude": float(np.max(band_data)),
                "energy": float(np.sum(band_data ** 2)),
                "std": float(np.std(band_data)),
                "flatness": float(
                    np.exp(np.mean(np.log(np.abs(band_data) + 1e-12)))
                    / (np.mean(np.abs(band_data)) + 1e-12)
                ),
            })

        return bands

    def detect_transients(self, spectrum: np.ndarray) -> List[Dict[str, float]]:
        """Detect transient-like features in the spectrum.

        Args:
            spectrum: Input spectrum.

        Returns:
            List of detected transient features.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        transients = []

        if len(spectrum) < 5:
            return transients

        # Compute local variance
        window = 5
        local_var = np.array([
            np.var(spectrum[max(0, i - window):i + window + 1])
            for i in range(len(spectrum))
        ])

        # Detect sudden changes
        var_threshold = np.mean(local_var) + 2 * np.std(local_var)
        for i in range(1, len(local_var)):
            if local_var[i] > var_threshold and local_var[i - 1] < var_threshold:
                transients.append({
                    "index": float(i),
                    "magnitude": float(local_var[i]),
                    "rise_rate": float(local_var[i] - local_var[i - 1]),
                })

        return transients

    def full_analysis(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """Perform complete pattern analysis.

        Args:
            spectrum: Input spectrum.

        Returns:
            Complete analysis dictionary.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        peaks = self.detect_peaks(spectrum)
        harmonics = self.detect_harmonics(spectrum, peaks)
        bands = self.analyze_bands(spectrum)
        transients = self.detect_transients(spectrum)

        # Overall characterization
        total_energy = float(np.sum(spectrum ** 2))
        spectral_centroid = float(
            np.sum(np.arange(len(spectrum)) * np.abs(spectrum))
            / (np.sum(np.abs(spectrum)) + 1e-12)
        )
        spectral_flatness = float(
            np.exp(np.mean(np.log(np.abs(spectrum) + 1e-12)))
            / (np.mean(np.abs(spectrum)) + 1e-12)
        )

        return {
            "peaks": peaks,
            "harmonics": harmonics,
            "bands": bands,
            "transients": transients,
            "n_peaks": len(peaks),
            "n_harmonics": len(harmonics),
            "n_transients": len(transients),
            "total_energy": total_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_flatness": spectral_flatness,
            "spectrum_length": len(spectrum),
        }


# =============================================================================
# Reasoning Engine (Orchestrator)
# =============================================================================


class SpectralReasoningEngine:
    """Complete spectral reasoning engine orchestrating all components.

    Provides a unified interface for deep reasoning about spectral data
    by coordinating causal graphs, Bayesian updating, abductive reasoning,
    counterfactual analysis, and pattern recognition.

    Args:
        max_chain_depth: Maximum depth of reasoning chains.
        confidence_threshold: Minimum confidence for conclusions.
        enable_counterfactuals: Whether to include counterfactual analysis.
        enable_causal: Whether to build and use causal graphs.
    """

    def __init__(
        self,
        max_chain_depth: int = 10,
        confidence_threshold: float = 0.6,
        enable_counterfactuals: bool = True,
        enable_causal: bool = True,
    ) -> None:
        self.max_chain_depth = max_chain_depth
        self.confidence_threshold = confidence_threshold
        self.enable_counterfactuals = enable_counterfactuals
        self.enable_causal = enable_causal

        # Components
        self._causal_graph = CausalGraph()
        self._bayesian_updater = BayesianUpdater()
        self._abductive_reasoner = AbductiveReasoner()
        self._counterfactual_engine = CounterfactualEngine()
        self._pattern_recognizer = SpectralPatternRecognizer()
        self._evidence_accumulator = EvidenceAccumulator()

        # State
        self._reasoning_chains: List[ReasoningChainResult] = []
        self._chain_counter: int = 0

    def reason(
        self,
        spectrum: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        mode: ReasoningMode = ReasoningMode.ENSEMBLE,
    ) -> ReasoningChainResult:
        """Perform deep reasoning about a spectral observation.

        Executes a multi-step reasoning chain that:
        1. Recognizes patterns in the spectrum
        2. Generates hypotheses about causes
        3. Accumulates evidence from data
        4. Updates beliefs via Bayesian inference
        5. Optionally builds causal models
        6. Optionally generates counterfactuals

        Args:
            spectrum: Input spectral data.
            context: Optional contextual information.
            mode: Primary reasoning mode to employ.

        Returns:
            Complete ReasoningChainResult with all steps and conclusions.
        """
        start_time = time.time()
        self._chain_counter += 1
        chain_id = f"chain_{self._chain_counter}"
        steps: List[ReasoningStep] = []
        hypotheses: List[Hypothesis] = []
        causal_links: List[CausalLink] = []

        spectrum = np.atleast_1d(spectrum).flatten()
        ctx = context or {}

        # Step 1: Pattern Recognition
        step_start = time.time()
        analysis = self._pattern_recognizer.full_analysis(spectrum)
        steps.append(ReasoningStep(
            step_id=1,
            mode=ReasoningMode.DEDUCTIVE,
            input_state={"spectrum_length": len(spectrum)},
            output_state={"n_peaks": analysis["n_peaks"], "n_harmonics": analysis["n_harmonics"]},
            rationale="Pattern recognition: identified spectral features",
            confidence=0.9,
            duration=time.time() - step_start,
        ))

        # Step 2: Hypothesis Generation
        step_start = time.time()
        observations = self._characterize_observations(analysis)
        for obs in observations:
            self._abductive_reasoner.observe(obs)
            hyps = self._abductive_reasoner.generate_hypotheses(obs)
            hypotheses.extend(hyps)

        steps.append(ReasoningStep(
            step_id=2,
            mode=ReasoningMode.ABDUCTIVE,
            input_state={"n_observations": len(observations)},
            output_state={"n_hypotheses": len(hypotheses)},
            rationale="Abductive reasoning: generated candidate explanations",
            confidence=0.7,
            duration=time.time() - step_start,
        ))

        # Step 3: Evidence Accumulation
        step_start = time.time()
        evidence_list = self._gather_evidence(spectrum, analysis)
        self._evidence_accumulator.reset()
        decision = self._evidence_accumulator.batch_accumulate(evidence_list)

        steps.append(ReasoningStep(
            step_id=3,
            mode=ReasoningMode.INDUCTIVE,
            input_state={"n_evidence": len(evidence_list)},
            output_state={"decision": decision, "accumulated": self._evidence_accumulator.current_evidence},
            rationale="Evidence accumulation: collected and weighed spectral evidence",
            confidence=0.75,
            duration=time.time() - step_start,
        ))

        # Step 4: Bayesian Update
        step_start = time.time()
        if hypotheses:
            self._bayesian_updater.initialize_beliefs(
                [h.hypothesis_id for h in hypotheses]
            )
            for evidence in evidence_list:
                # Assign evidence to most relevant hypothesis
                for h in hypotheses:
                    if self._evidence_matches_hypothesis(evidence, h):
                        self._bayesian_updater.update(h.hypothesis_id, evidence)
                        h.evidence.append(evidence)

        top_hypotheses = self._bayesian_updater.get_top_hypotheses(5)
        steps.append(ReasoningStep(
            step_id=4,
            mode=ReasoningMode.BAYESIAN,
            input_state={"n_hypotheses": len(hypotheses)},
            output_state={
                "top_hypothesis": top_hypotheses[0] if top_hypotheses else None,
                "belief_entropy": self._bayesian_updater.get_entropy(),
            },
            rationale="Bayesian updating: computed posterior beliefs",
            confidence=0.8,
            duration=time.time() - step_start,
        ))

        # Step 5: Causal Modeling (optional)
        if self.enable_causal and analysis["n_peaks"] > 0:
            step_start = time.time()
            causal_links = self._build_causal_model(analysis, hypotheses)
            steps.append(ReasoningStep(
                step_id=5,
                mode=ReasoningMode.CAUSAL,
                input_state={"n_peaks": analysis["n_peaks"]},
                output_state={"n_causal_links": len(causal_links)},
                rationale="Causal modeling: identified causal relationships",
                confidence=0.65,
                duration=time.time() - step_start,
            ))

        # Step 6: Counterfactual Analysis (optional)
        if self.enable_counterfactuals and len(spectrum) > 10:
            step_start = time.time()
            cf_results = self._run_counterfactuals(spectrum, analysis)
            steps.append(ReasoningStep(
                step_id=6,
                mode=ReasoningMode.COUNTERFACTUAL,
                input_state={"n_interventions": len(cf_results)},
                output_state={"counterfactual_effects": [r["effect_size"] for r in cf_results]},
                rationale="Counterfactual analysis: simulated alternative scenarios",
                confidence=0.6,
                duration=time.time() - step_start,
            ))

        # Determine final conclusion
        final_conclusion, overall_confidence = self._synthesize_conclusion(
            steps, hypotheses, top_hypotheses
        )

        total_duration = time.time() - start_time

        result = ReasoningChainResult(
            chain_id=chain_id,
            steps=steps,
            final_conclusion=final_conclusion,
            overall_confidence=overall_confidence,
            hypotheses=hypotheses,
            causal_links=causal_links,
            total_duration=total_duration,
            metadata={
                "mode": mode.value,
                "spectrum_length": len(spectrum),
                "context": ctx,
                "belief_entropy": self._bayesian_updater.get_entropy(),
            },
        )

        self._reasoning_chains.append(result)
        return result

    def _characterize_observations(
        self, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert analysis results into structured observations."""
        observations = []

        if analysis["n_peaks"] > 5:
            observations.append({
                "pattern_type": "harmonic_series",
                "magnitude": analysis["n_peaks"],
            })
        elif analysis["n_peaks"] > 0:
            observations.append({
                "pattern_type": "peak_shift",
                "magnitude": analysis["peaks"][0]["amplitude"] if analysis["peaks"] else 0,
            })

        if analysis["spectral_flatness"] > 0.8:
            observations.append({
                "pattern_type": "broadband_increase",
                "magnitude": analysis["spectral_flatness"],
            })

        if analysis["n_transients"] > 0:
            observations.append({
                "pattern_type": "new_peak",
                "magnitude": analysis["n_transients"],
            })

        if not observations:
            observations.append({
                "pattern_type": "amplitude_increase",
                "magnitude": analysis["total_energy"],
            })

        return observations

    def _gather_evidence(
        self,
        spectrum: np.ndarray,
        analysis: Dict[str, Any],
    ) -> List[Evidence]:
        """Gather evidence from spectral analysis results."""
        evidence_list = []

        # Peak-based evidence
        for peak in analysis["peaks"][:5]:
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.SPECTRAL_PATTERN,
                description=f"Peak at bin {peak['index']:.0f} with amplitude {peak['amplitude']:.3f}",
                strength=min(1.0, peak["prominence"] / 5.0),
                source="pattern_recognizer",
            ))

        # Harmonic evidence
        for harmonic in analysis["harmonics"][:3]:
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.HARMONIC_STRUCTURE,
                description=f"Harmonic series with {harmonic['n_harmonics']} components",
                strength=min(1.0, harmonic["n_harmonics"] / 5.0),
                source="harmonic_detector",
            ))

        # Band energy evidence
        if analysis["bands"]:
            energies = [b["energy"] for b in analysis["bands"]]
            max_band = int(np.argmax(energies))
            evidence_list.append(Evidence(
                evidence_type=EvidenceType.CROSS_BAND_RELATIONSHIP,
                description=f"Dominant energy in band {max_band}",
                strength=0.5,
                source="band_analyzer",
            ))

        # Statistical evidence
        evidence_list.append(Evidence(
            evidence_type=EvidenceType.STATISTICAL,
            description=f"Total energy: {analysis['total_energy']:.3f}",
            strength=0.3,
            source="statistics",
        ))

        return evidence_list

    def _evidence_matches_hypothesis(
        self, evidence: Evidence, hypothesis: Hypothesis
    ) -> bool:
        """Check if evidence is relevant to a hypothesis."""
        desc = hypothesis.description.lower()
        e_desc = evidence.description.lower()

        if "harmonic" in desc and evidence.evidence_type == EvidenceType.HARMONIC_STRUCTURE:
            return True
        if "resonance" in desc and evidence.evidence_type == EvidenceType.SPECTRAL_PATTERN:
            return True
        if "noise" in desc and evidence.evidence_type == EvidenceType.STATISTICAL:
            return True
        if "peak" in e_desc and ("shift" in desc or "excitation" in desc):
            return True
        return False

    def _build_causal_model(
        self,
        analysis: Dict[str, Any],
        hypotheses: List[Hypothesis],
    ) -> List[CausalLink]:
        """Build a causal model from analysis and hypotheses."""
        links = []

        # Create causal links between peaks and harmonics
        if analysis["harmonics"]:
            for harmonic in analysis["harmonics"]:
                link = CausalLink(
                    cause=f"fundamental_{harmonic['fundamental_index']:.0f}",
                    effect=f"harmonic_series_{harmonic['n_harmonics']}",
                    strength=harmonic["completeness"],
                    mechanism="nonlinear_generation",
                    confidence=0.7,
                )
                links.append(link)
                self._causal_graph.add_edge(link)

        # Create causal links from hypotheses
        for h in hypotheses[:5]:
            if h.posterior_probability > 0.3:
                link = CausalLink(
                    cause=h.description,
                    effect="observed_spectrum",
                    strength=h.posterior_probability,
                    mechanism="hypothesized",
                    confidence=h.posterior_probability * 0.8,
                )
                links.append(link)
                self._causal_graph.add_edge(link)

        return links

    def _run_counterfactuals(
        self,
        spectrum: np.ndarray,
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run counterfactual simulations."""
        results = []

        # Test removing dominant peak
        if analysis["peaks"]:
            top_peak = analysis["peaks"][0]
            result = self._counterfactual_engine.assess_causal_effect(
                spectrum,
                "remove_peak",
                {"center": top_peak["index"], "width": max(5, top_peak["width"])},
            )
            # Remove the spectrum from result to save memory
            result.pop("counterfactual_spectrum", None)
            results.append(result)

        # Test adding damping
        result = self._counterfactual_engine.assess_causal_effect(
            spectrum, "add_damping", {"factor": 0.5}
        )
        result.pop("counterfactual_spectrum", None)
        results.append(result)

        return results

    def _synthesize_conclusion(
        self,
        steps: List[ReasoningStep],
        hypotheses: List[Hypothesis],
        top_hypotheses: List[Tuple[str, float]],
    ) -> Tuple[str, float]:
        """Synthesize final conclusion from all reasoning steps."""
        if not top_hypotheses:
            return "insufficient_data", 0.3

        best_id, best_prob = top_hypotheses[0]
        # Find the hypothesis object
        best_hyp = None
        for h in hypotheses:
            if h.hypothesis_id == best_id:
                best_hyp = h
                break

        conclusion = best_hyp.description if best_hyp else "unknown"

        # Overall confidence is average of step confidences weighted by best hypothesis
        step_confidences = [s.confidence for s in steps]
        avg_step_confidence = float(np.mean(step_confidences)) if step_confidences else 0.5
        overall_confidence = (avg_step_confidence + best_prob) / 2.0

        return conclusion, float(np.clip(overall_confidence, 0.0, 1.0))

    @property
    def causal_graph(self) -> CausalGraph:
        """Access the causal graph."""
        return self._causal_graph

    @property
    def n_reasoning_chains(self) -> int:
        """Number of reasoning chains executed."""
        return len(self._reasoning_chains)

    @property
    def pattern_recognizer(self) -> SpectralPatternRecognizer:
        """Access the pattern recognizer."""
        return self._pattern_recognizer
