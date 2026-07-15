"""Spectral Knowledge Graph and Ontology System.

Provides a rich knowledge graph for spectral intelligence that captures
relationships between spectral phenomena, materials, physical processes,
and measurement conditions. Enables semantic reasoning, ontological
classification, and knowledge-driven spectral analysis.

Key Components:
    - SpectralOntology: Hierarchical classification of spectral phenomena
    - KnowledgeNode: Individual knowledge entities
    - KnowledgeRelation: Typed relationships between entities
    - SpectralKnowledgeGraph: Full knowledge graph implementation
    - OntologicalReasoner: Reasoning over the knowledge graph
    - KnowledgeDrivenClassifier: Classification using graph knowledge
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class RelationType(Enum):
    """Types of relationships in the spectral knowledge graph."""

    IS_A = "is_a"                          # Taxonomic
    HAS_PROPERTY = "has_property"          # Attribute
    CAUSES = "causes"                      # Causal
    CORRELATES_WITH = "correlates_with"    # Statistical
    PART_OF = "part_of"                    # Compositional
    SIMILAR_TO = "similar_to"             # Similarity
    OPPOSITE_OF = "opposite_of"           # Contrast
    PRECEDES = "precedes"                  # Temporal
    MEASURED_BY = "measured_by"            # Instrument
    OCCURS_IN = "occurs_in"               # Context
    PRODUCES = "produces"                  # Generative
    INHIBITS = "inhibits"                  # Suppressive
    MODULATES = "modulates"               # Modulation
    RESONATES_WITH = "resonates_with"     # Resonance
    HARMONICALLY_RELATED = "harmonically_related"  # Harmonic


class NodeType(Enum):
    """Types of nodes in the spectral knowledge graph."""

    PHENOMENON = "phenomenon"
    MATERIAL = "material"
    PROCESS = "process"
    MEASUREMENT = "measurement"
    PROPERTY = "property"
    CONDITION = "condition"
    INSTRUMENT = "instrument"
    DOMAIN = "domain"
    PATTERN = "pattern"
    FREQUENCY_BAND = "frequency_band"
    SPECTRAL_FEATURE = "spectral_feature"


class InferenceRule(Enum):
    """Rules for ontological inference."""

    TRANSITIVITY = "transitivity"     # If A->B and B->C then A->C
    SYMMETRY = "symmetry"            # If A->B then B->A
    INHERITANCE = "inheritance"       # Subclass inherits properties
    EXCLUSION = "exclusion"          # Contradicting properties exclude
    COMPOSITION = "composition"       # Parts compose wholes
    CAUSATION = "causation"          # Causes propagate


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class KnowledgeNode:
    """A node in the spectral knowledge graph.

    Args:
        node_id: Unique identifier.
        name: Human-readable name.
        node_type: Ontological type.
        properties: Node properties/attributes.
        embedding: Optional vector representation.
        confidence: Confidence in this knowledge (0-1).
        source: Source of this knowledge.
        created_at: Creation timestamp.
    """

    node_id: str
    name: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    source: str = "system"
    created_at: float = field(default_factory=time.time)

    @property
    def has_embedding(self) -> bool:
        """Whether this node has a vector representation."""
        return self.embedding is not None


@dataclass
class KnowledgeRelation:
    """A typed relation between two knowledge nodes.

    Args:
        source_id: Source node identifier.
        target_id: Target node identifier.
        relation_type: Type of relationship.
        strength: Relationship strength (0-1).
        confidence: Confidence in this relation.
        evidence: Supporting evidence for this relation.
        metadata: Additional relation metadata.
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_strong(self) -> bool:
        """Whether this is a strong, confident relation."""
        return self.strength > 0.7 and self.confidence > 0.7


@dataclass
class InferenceResult:
    """Result from ontological inference.

    Args:
        conclusion: Inferred knowledge.
        rule_used: Inference rule applied.
        premises: Source relations/nodes used.
        confidence: Confidence in the inference.
        path: Reasoning path.
    """

    conclusion: str
    rule_used: InferenceRule
    premises: List[str] = field(default_factory=list)
    confidence: float = 0.0
    path: List[str] = field(default_factory=list)


# =============================================================================
# Spectral Ontology
# =============================================================================


class SpectralOntology:
    """Hierarchical ontology for spectral phenomena.

    Provides a structured classification system for all spectral
    concepts, enabling type checking, inheritance, and semantic
    reasoning about spectral data.
    """

    def __init__(self) -> None:
        self._taxonomy: Dict[str, List[str]] = {}  # parent -> children
        self._parents: Dict[str, str] = {}  # child -> parent
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._initialize_base_ontology()

    def _initialize_base_ontology(self) -> None:
        """Initialize the base spectral ontology."""
        # Root categories
        self._add_category("spectral_entity", None)
        
        # First-level categories
        self._add_category("spectral_phenomenon", "spectral_entity")
        self._add_category("spectral_property", "spectral_entity")
        self._add_category("spectral_process", "spectral_entity")
        self._add_category("frequency_range", "spectral_entity")
        self._add_category("signal_type", "spectral_entity")

        # Phenomena
        self._add_category("resonance", "spectral_phenomenon")
        self._add_category("harmonic", "spectral_phenomenon")
        self._add_category("anti_resonance", "spectral_phenomenon")
        self._add_category("mode_shape", "spectral_phenomenon")
        self._add_category("spectral_peak", "spectral_phenomenon")
        self._add_category("spectral_valley", "spectral_phenomenon")
        self._add_category("broadband_noise", "spectral_phenomenon")
        self._add_category("narrowband_signal", "spectral_phenomenon")
        self._add_category("transient", "spectral_phenomenon")
        self._add_category("steady_state", "spectral_phenomenon")

        # Resonance subtypes
        self._add_category("natural_frequency", "resonance")
        self._add_category("forced_resonance", "resonance")
        self._add_category("parametric_resonance", "resonance")
        self._add_category("stochastic_resonance", "resonance")
        self._add_category("acoustic_resonance", "resonance")
        self._add_category("structural_resonance", "resonance")
        self._add_category("electromagnetic_resonance", "resonance")

        # Harmonic subtypes
        self._add_category("fundamental_frequency", "harmonic")
        self._add_category("overtone", "harmonic")
        self._add_category("subharmonic", "harmonic")
        self._add_category("intermodulation", "harmonic")
        self._add_category("combination_tone", "harmonic")

        # Properties
        self._add_category("amplitude", "spectral_property")
        self._add_category("frequency", "spectral_property")
        self._add_category("phase", "spectral_property")
        self._add_category("bandwidth", "spectral_property")
        self._add_category("damping", "spectral_property")
        self._add_category("coherence", "spectral_property")
        self._add_category("power_density", "spectral_property")
        self._add_category("spectral_slope", "spectral_property")
        self._add_category("spectral_centroid", "spectral_property")
        self._add_category("spectral_flatness", "spectral_property")
        self._add_category("spectral_rolloff", "spectral_property")
        self._add_category("crest_factor", "spectral_property")

        # Processes
        self._add_category("frequency_shift", "spectral_process")
        self._add_category("amplitude_modulation", "spectral_process")
        self._add_category("frequency_modulation", "spectral_process")
        self._add_category("damping_change", "spectral_process")
        self._add_category("mode_coupling", "spectral_process")
        self._add_category("energy_transfer", "spectral_process")
        self._add_category("nonlinear_distortion", "spectral_process")
        self._add_category("spectral_leakage", "spectral_process")
        self._add_category("aliasing", "spectral_process")

        # Frequency ranges
        self._add_category("infrasonic", "frequency_range", {"min_hz": 0, "max_hz": 20})
        self._add_category("low_frequency", "frequency_range", {"min_hz": 20, "max_hz": 200})
        self._add_category("mid_frequency", "frequency_range", {"min_hz": 200, "max_hz": 2000})
        self._add_category("high_frequency", "frequency_range", {"min_hz": 2000, "max_hz": 20000})
        self._add_category("ultrasonic", "frequency_range", {"min_hz": 20000, "max_hz": 1000000})

        # Signal types
        self._add_category("deterministic", "signal_type")
        self._add_category("stochastic", "signal_type")
        self._add_category("periodic", "deterministic")
        self._add_category("quasi_periodic", "deterministic")
        self._add_category("aperiodic", "deterministic")
        self._add_category("white_noise", "stochastic")
        self._add_category("pink_noise", "stochastic")
        self._add_category("brown_noise", "stochastic")

    def _add_category(
        self, name: str, parent: Optional[str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a category to the ontology."""
        if name not in self._taxonomy:
            self._taxonomy[name] = []
        if parent is not None:
            self._parents[name] = parent
            if parent not in self._taxonomy:
                self._taxonomy[parent] = []
            self._taxonomy[parent].append(name)
        if properties:
            self._properties[name] = properties

    def is_a(self, child: str, ancestor: str) -> bool:
        """Check if child is a subtype of ancestor (transitive).

        Args:
            child: Potential subtype.
            ancestor: Potential supertype.

        Returns:
            True if child IS-A ancestor.
        """
        if child == ancestor:
            return True
        current = child
        visited = set()
        while current in self._parents and current not in visited:
            visited.add(current)
            current = self._parents[current]
            if current == ancestor:
                return True
        return False

    def get_ancestors(self, category: str) -> List[str]:
        """Get all ancestors of a category.

        Args:
            category: Category to query.

        Returns:
            List of ancestors from immediate parent to root.
        """
        ancestors = []
        current = category
        visited = set()
        while current in self._parents and current not in visited:
            visited.add(current)
            parent = self._parents[current]
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_descendants(self, category: str) -> List[str]:
        """Get all descendants of a category (recursive).

        Args:
            category: Category to query.

        Returns:
            List of all descendant categories.
        """
        descendants = []
        queue = list(self._taxonomy.get(category, []))
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)
            queue.extend(self._taxonomy.get(current, []))
        return descendants

    def get_siblings(self, category: str) -> List[str]:
        """Get sibling categories (same parent).

        Args:
            category: Category to query.

        Returns:
            List of sibling categories.
        """
        parent = self._parents.get(category)
        if parent is None:
            return []
        return [c for c in self._taxonomy.get(parent, []) if c != category]

    def get_properties(self, category: str) -> Dict[str, Any]:
        """Get inherited properties for a category.

        Args:
            category: Category to query.

        Returns:
            Merged properties from category and all ancestors.
        """
        props = {}
        # Collect from ancestors first (inheritance)
        ancestors = self.get_ancestors(category)
        for anc in reversed(ancestors):
            props.update(self._properties.get(anc, {}))
        # Override with own properties
        props.update(self._properties.get(category, {}))
        return props

    def classify(self, features: Dict[str, float]) -> List[Tuple[str, float]]:
        """Classify spectral features into ontological categories.

        Args:
            features: Dictionary of measured features.

        Returns:
            List of (category, confidence) sorted by confidence.
        """
        classifications = []

        # Rule-based classification
        if "n_peaks" in features:
            if features["n_peaks"] == 0:
                classifications.append(("broadband_noise", 0.7))
            elif features["n_peaks"] == 1:
                classifications.append(("spectral_peak", 0.8))
            elif features["n_peaks"] > 5:
                classifications.append(("harmonic", 0.6))

        if "spectral_flatness" in features:
            if features["spectral_flatness"] > 0.8:
                classifications.append(("white_noise", 0.7))
            elif features["spectral_flatness"] > 0.5:
                classifications.append(("stochastic", 0.6))
            else:
                classifications.append(("deterministic", 0.6))

        if "damping" in features:
            if features["damping"] < 0.01:
                classifications.append(("resonance", 0.8))
            elif features["damping"] > 0.1:
                classifications.append(("steady_state", 0.6))

        if "bandwidth" in features:
            if features["bandwidth"] < 0.05:
                classifications.append(("narrowband_signal", 0.8))
            else:
                classifications.append(("broadband_noise", 0.6))

        classifications.sort(key=lambda x: x[1], reverse=True)
        return classifications

    @property
    def n_categories(self) -> int:
        """Total number of categories in the ontology."""
        return len(self._taxonomy)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ontology to dictionary."""
        return {
            "taxonomy": {k: v for k, v in self._taxonomy.items() if v},
            "n_categories": self.n_categories,
            "root_categories": [k for k in self._taxonomy if k not in self._parents],
        }


# =============================================================================
# Spectral Knowledge Graph
# =============================================================================


class SpectralKnowledgeGraph:
    """Full knowledge graph for spectral intelligence.

    Stores nodes (entities) and relations (edges) with support
    for querying, traversal, inference, and embedding-based search.

    Args:
        max_nodes: Maximum number of nodes.
        max_relations: Maximum number of relations.
    """

    def __init__(self, max_nodes: int = 5000, max_relations: int = 20000) -> None:
        self.max_nodes = max_nodes
        self.max_relations = max_relations
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._relations: List[KnowledgeRelation] = []
        self._outgoing: Dict[str, List[int]] = {}  # node_id -> relation indices
        self._incoming: Dict[str, List[int]] = {}  # node_id -> relation indices
        self._ontology = SpectralOntology()

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the knowledge graph.

        Args:
            node: KnowledgeNode to add.
        """
        if len(self._nodes) >= self.max_nodes:
            self._evict_weakest_node()

        self._nodes[node.node_id] = node
        if node.node_id not in self._outgoing:
            self._outgoing[node.node_id] = []
        if node.node_id not in self._incoming:
            self._incoming[node.node_id] = []

    def add_relation(self, relation: KnowledgeRelation) -> None:
        """Add a relation to the knowledge graph.

        Args:
            relation: KnowledgeRelation to add.
        """
        if len(self._relations) >= self.max_relations:
            self._evict_weakest_relation()

        # Ensure nodes exist
        if relation.source_id not in self._nodes:
            self.add_node(KnowledgeNode(
                node_id=relation.source_id,
                name=relation.source_id,
                node_type=NodeType.PHENOMENON,
            ))
        if relation.target_id not in self._nodes:
            self.add_node(KnowledgeNode(
                node_id=relation.target_id,
                name=relation.target_id,
                node_type=NodeType.PHENOMENON,
            ))

        idx = len(self._relations)
        self._relations.append(relation)
        self._outgoing[relation.source_id].append(idx)
        self._incoming[relation.target_id].append(idx)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing",
    ) -> List[Tuple[KnowledgeNode, KnowledgeRelation]]:
        """Get neighboring nodes with their relations.

        Args:
            node_id: Source node.
            relation_type: Optional filter by relation type.
            direction: 'outgoing', 'incoming', or 'both'.

        Returns:
            List of (node, relation) tuples.
        """
        neighbors = []

        if direction in ("outgoing", "both"):
            for idx in self._outgoing.get(node_id, []):
                rel = self._relations[idx]
                if relation_type is None or rel.relation_type == relation_type:
                    target = self._nodes.get(rel.target_id)
                    if target:
                        neighbors.append((target, rel))

        if direction in ("incoming", "both"):
            for idx in self._incoming.get(node_id, []):
                rel = self._relations[idx]
                if relation_type is None or rel.relation_type == relation_type:
                    source = self._nodes.get(rel.source_id)
                    if source:
                        neighbors.append((source, rel))

        return neighbors

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
    ) -> Optional[List[Tuple[str, KnowledgeRelation]]]:
        """Find shortest path between two nodes.

        Args:
            start_id: Starting node.
            end_id: Target node.
            max_depth: Maximum path length.

        Returns:
            List of (node_id, relation) pairs forming the path, or None.
        """
        if start_id not in self._nodes or end_id not in self._nodes:
            return None

        # BFS
        visited: Set[str] = set()
        queue: List[Tuple[str, List[Tuple[str, KnowledgeRelation]]]] = [
            (start_id, [])
        ]

        while queue:
            current, path = queue.pop(0)
            if current == end_id:
                return path
            if current in visited or len(path) >= max_depth:
                continue
            visited.add(current)

            for idx in self._outgoing.get(current, []):
                rel = self._relations[idx]
                if rel.target_id not in visited:
                    queue.append((rel.target_id, path + [(rel.target_id, rel)]))

        return None

    def query_by_type(self, node_type: NodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: Type to filter by.

        Returns:
            List of matching nodes.
        """
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def query_by_property(self, property_name: str, value: Any) -> List[KnowledgeNode]:
        """Get all nodes with a specific property value.

        Args:
            property_name: Property key.
            value: Expected value.

        Returns:
            List of matching nodes.
        """
        return [
            n for n in self._nodes.values()
            if n.properties.get(property_name) == value
        ]

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_type: Optional[NodeType] = None,
    ) -> List[Tuple[KnowledgeNode, float]]:
        """Search the knowledge graph by embedding similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            node_type: Optional type filter.

        Returns:
            List of (node, similarity) sorted by similarity.
        """
        query = np.atleast_1d(query_embedding).flatten()
        query_norm = np.linalg.norm(query) + 1e-12

        results = []
        for node in self._nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if node.embedding is None:
                continue

            emb = node.embedding.flatten()
            min_len = min(len(query), len(emb))
            sim = float(
                np.dot(query[:min_len], emb[:min_len])
                / (query_norm * (np.linalg.norm(emb[:min_len]) + 1e-12))
            )
            results.append((node, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compute_centrality(self) -> Dict[str, float]:
        """Compute degree centrality for all nodes.

        Returns:
            Dictionary of node_id -> centrality score.
        """
        centrality = {}
        n = max(1, len(self._nodes) - 1)

        for node_id in self._nodes:
            degree = len(self._outgoing.get(node_id, [])) + len(self._incoming.get(node_id, []))
            centrality[node_id] = degree / n

        return centrality

    def get_subgraph(
        self,
        center_id: str,
        radius: int = 2,
    ) -> Tuple[List[KnowledgeNode], List[KnowledgeRelation]]:
        """Extract a subgraph around a center node.

        Args:
            center_id: Center node.
            radius: Maximum distance from center.

        Returns:
            Tuple of (nodes, relations) in the subgraph.
        """
        visited_nodes: Set[str] = set()
        queue = [(center_id, 0)]
        subgraph_relations = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited_nodes or depth > radius:
                continue
            visited_nodes.add(current)

            for idx in self._outgoing.get(current, []):
                rel = self._relations[idx]
                subgraph_relations.append(rel)
                if rel.target_id not in visited_nodes:
                    queue.append((rel.target_id, depth + 1))

            for idx in self._incoming.get(current, []):
                rel = self._relations[idx]
                subgraph_relations.append(rel)
                if rel.source_id not in visited_nodes:
                    queue.append((rel.source_id, depth + 1))

        nodes = [self._nodes[nid] for nid in visited_nodes if nid in self._nodes]
        return nodes, subgraph_relations

    def _evict_weakest_node(self) -> None:
        """Remove the least connected node."""
        if not self._nodes:
            return
        min_degree = float("inf")
        weakest = None
        for node_id in self._nodes:
            degree = len(self._outgoing.get(node_id, [])) + len(self._incoming.get(node_id, []))
            if degree < min_degree:
                min_degree = degree
                weakest = node_id
        if weakest:
            self.remove_node(weakest)

    def _evict_weakest_relation(self) -> None:
        """Remove the weakest relation."""
        if not self._relations:
            return
        weakest_idx = int(np.argmin([r.strength * r.confidence for r in self._relations]))
        self._relations.pop(weakest_idx)
        # Rebuild indices
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild outgoing/incoming indices."""
        self._outgoing = {nid: [] for nid in self._nodes}
        self._incoming = {nid: [] for nid in self._nodes}
        for idx, rel in enumerate(self._relations):
            if rel.source_id in self._outgoing:
                self._outgoing[rel.source_id].append(idx)
            if rel.target_id in self._incoming:
                self._incoming[rel.target_id].append(idx)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its relations."""
        self._nodes.pop(node_id, None)
        self._relations = [
            r for r in self._relations
            if r.source_id != node_id and r.target_id != node_id
        ]
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        self._rebuild_indices()

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def n_relations(self) -> int:
        """Number of relations."""
        return len(self._relations)

    @property
    def ontology(self) -> SpectralOntology:
        """Access the spectral ontology."""
        return self._ontology

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self._nodes:
            return {"n_nodes": 0, "n_relations": 0}

        degrees = [
            len(self._outgoing.get(nid, [])) + len(self._incoming.get(nid, []))
            for nid in self._nodes
        ]
        type_counts = {}
        for n in self._nodes.values():
            type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1

        return {
            "n_nodes": self.n_nodes,
            "n_relations": self.n_relations,
            "mean_degree": float(np.mean(degrees)),
            "max_degree": int(np.max(degrees)),
            "type_distribution": type_counts,
            "density": self.n_relations / max(1, self.n_nodes * (self.n_nodes - 1)),
        }


# =============================================================================
# Ontological Reasoner
# =============================================================================


class OntologicalReasoner:
    """Reasoning engine for the spectral knowledge graph.

    Performs inference over the knowledge graph using ontological
    rules, transitive closure, inheritance, and analogy.

    Args:
        knowledge_graph: The knowledge graph to reason over.
        max_inference_depth: Maximum depth for transitive inference.
    """

    def __init__(
        self,
        knowledge_graph: SpectralKnowledgeGraph,
        max_inference_depth: int = 5,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.max_inference_depth = max_inference_depth
        self._inferences: List[InferenceResult] = []

    def infer_transitive(
        self,
        start_id: str,
        relation_type: RelationType,
    ) -> List[InferenceResult]:
        """Perform transitive inference along a relation type.

        If A -[rel]-> B and B -[rel]-> C, infer A -[rel]-> C.

        Args:
            start_id: Starting node.
            relation_type: Relation to follow transitively.

        Returns:
            List of inferred results.
        """
        results = []
        visited = set()
        queue = [(start_id, [start_id], 1.0)]

        while queue:
            current, path, accumulated_confidence = queue.pop(0)
            if current in visited or len(path) > self.max_inference_depth:
                continue
            visited.add(current)

            neighbors = self.knowledge_graph.get_neighbors(
                current, relation_type=relation_type, direction="outgoing"
            )
            for node, rel in neighbors:
                if node.node_id not in visited:
                    new_confidence = accumulated_confidence * rel.confidence
                    new_path = path + [node.node_id]

                    if len(new_path) > 2:  # Only inferred (not direct)
                        results.append(InferenceResult(
                            conclusion=f"{start_id} -[{relation_type.value}]-> {node.node_id}",
                            rule_used=InferenceRule.TRANSITIVITY,
                            premises=[f"{path[-1]} -> {node.node_id}"],
                            confidence=new_confidence,
                            path=new_path,
                        ))

                    queue.append((node.node_id, new_path, new_confidence))

        self._inferences.extend(results)
        return results

    def infer_inheritance(self, node_id: str) -> List[InferenceResult]:
        """Infer inherited properties from parent types.

        Args:
            node_id: Node to compute inheritance for.

        Returns:
            List of inherited property inferences.
        """
        results = []
        node = self.knowledge_graph.get_node(node_id)
        if not node:
            return results

        # Find IS-A parents
        parents = self.knowledge_graph.get_neighbors(
            node_id, relation_type=RelationType.IS_A, direction="outgoing"
        )

        for parent_node, rel in parents:
            # Get parent's properties
            parent_relations = self.knowledge_graph.get_neighbors(
                parent_node.node_id, relation_type=RelationType.HAS_PROPERTY, direction="outgoing"
            )
            for prop_node, prop_rel in parent_relations:
                results.append(InferenceResult(
                    conclusion=f"{node_id} has_property {prop_node.node_id}",
                    rule_used=InferenceRule.INHERITANCE,
                    premises=[
                        f"{node_id} is_a {parent_node.node_id}",
                        f"{parent_node.node_id} has_property {prop_node.node_id}",
                    ],
                    confidence=rel.confidence * prop_rel.confidence * 0.9,
                    path=[node_id, parent_node.node_id, prop_node.node_id],
                ))

        self._inferences.extend(results)
        return results

    def find_analogies(
        self,
        source_id: str,
        target_type: Optional[NodeType] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find analogous entities based on structural similarity.

        Entities are analogous if they have similar relation patterns
        (same types of connections to similar types of nodes).

        Args:
            source_id: Source entity for analogy.
            target_type: Optional type constraint.
            top_k: Number of analogies to return.

        Returns:
            List of (node_id, analogy_score) pairs.
        """
        source_node = self.knowledge_graph.get_node(source_id)
        if not source_node:
            return []

        # Get source's relation pattern
        source_pattern = self._get_relation_pattern(source_id)
        if not source_pattern:
            return []

        # Compare with all other nodes
        scores = []
        for node_id, node in self.knowledge_graph._nodes.items():
            if node_id == source_id:
                continue
            if target_type and node.node_type != target_type:
                continue

            pattern = self._get_relation_pattern(node_id)
            similarity = self._pattern_similarity(source_pattern, pattern)
            if similarity > 0:
                scores.append((node_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _get_relation_pattern(self, node_id: str) -> Dict[str, int]:
        """Get the relation type pattern for a node."""
        pattern: Dict[str, int] = {}
        for idx in self.knowledge_graph._outgoing.get(node_id, []):
            rel = self.knowledge_graph._relations[idx]
            key = f"out_{rel.relation_type.value}"
            pattern[key] = pattern.get(key, 0) + 1
        for idx in self.knowledge_graph._incoming.get(node_id, []):
            rel = self.knowledge_graph._relations[idx]
            key = f"in_{rel.relation_type.value}"
            pattern[key] = pattern.get(key, 0) + 1
        return pattern

    def _pattern_similarity(self, p1: Dict[str, int], p2: Dict[str, int]) -> float:
        """Compute Jaccard-like similarity between relation patterns."""
        all_keys = set(p1.keys()) | set(p2.keys())
        if not all_keys:
            return 0.0

        intersection = sum(min(p1.get(k, 0), p2.get(k, 0)) for k in all_keys)
        union = sum(max(p1.get(k, 0), p2.get(k, 0)) for k in all_keys)
        return intersection / (union + 1e-12)

    @property
    def n_inferences(self) -> int:
        """Total inferences made."""
        return len(self._inferences)


# =============================================================================
# Knowledge-Driven Classifier
# =============================================================================


class KnowledgeDrivenClassifier:
    """Classify spectral data using knowledge graph reasoning.

    Combines pattern recognition with ontological knowledge to
    provide interpretable classifications with explanations.

    Args:
        knowledge_graph: Spectral knowledge graph.
        confidence_threshold: Minimum confidence for classification.
    """

    def __init__(
        self,
        knowledge_graph: SpectralKnowledgeGraph,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.confidence_threshold = confidence_threshold
        self._classification_count: int = 0

    def classify(
        self,
        spectrum: np.ndarray,
        features: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a spectrum using knowledge graph reasoning.

        Args:
            spectrum: Input spectral data.
            features: Optional pre-computed features.

        Returns:
            List of classification results with explanations.
        """
        self._classification_count += 1
        spectrum = np.atleast_1d(spectrum).flatten()

        # Compute features if not provided
        if features is None:
            features = self._extract_features(spectrum)

        # Ontological classification
        onto_classes = self.knowledge_graph.ontology.classify(features)

        # Embedding-based search
        embedding_matches = self.knowledge_graph.semantic_search(
            spectrum[:min(128, len(spectrum))],
            top_k=5,
        )

        # Combine results
        classifications = []
        for category, conf in onto_classes:
            if conf >= self.confidence_threshold:
                # Get ancestors for explanation
                ancestors = self.knowledge_graph.ontology.get_ancestors(category)
                classifications.append({
                    "category": category,
                    "confidence": conf,
                    "method": "ontological",
                    "explanation": f"{category} (is-a: {' > '.join(ancestors[:3])})",
                    "features_used": list(features.keys()),
                })

        for node, sim in embedding_matches:
            if sim >= self.confidence_threshold:
                classifications.append({
                    "category": node.name,
                    "confidence": sim,
                    "method": "embedding_similarity",
                    "explanation": f"Similar to known {node.node_type.value}: {node.name}",
                    "node_id": node.node_id,
                })

        classifications.sort(key=lambda x: x["confidence"], reverse=True)
        return classifications

    def _extract_features(self, spectrum: np.ndarray) -> Dict[str, float]:
        """Extract classification-relevant features."""
        features = {
            "mean_amplitude": float(np.mean(spectrum)),
            "std_amplitude": float(np.std(spectrum)),
            "max_amplitude": float(np.max(spectrum)),
            "energy": float(np.sum(spectrum ** 2)),
        }

        # Spectral flatness
        abs_spec = np.abs(spectrum) + 1e-12
        geo_mean = np.exp(np.mean(np.log(abs_spec)))
        arith_mean = np.mean(abs_spec)
        features["spectral_flatness"] = float(geo_mean / (arith_mean + 1e-12))

        # Count peaks
        n_peaks = 0
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                if spectrum[i] > np.mean(spectrum) + np.std(spectrum):
                    n_peaks += 1
        features["n_peaks"] = float(n_peaks)

        # Bandwidth estimate
        total = np.sum(abs_spec)
        cumulative = np.cumsum(abs_spec) / (total + 1e-12)
        low_idx = np.searchsorted(cumulative, 0.25)
        high_idx = np.searchsorted(cumulative, 0.75)
        features["bandwidth"] = float(high_idx - low_idx) / len(spectrum)

        return features

    @property
    def classification_count(self) -> int:
        """Number of classifications performed."""
        return self._classification_count
