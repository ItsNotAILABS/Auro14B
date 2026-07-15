"""Node topology mapping for graph-based spectral weighting."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from mesie.core.records import MultiElementRecord, SpectralComponent

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False


class NodeTopologyMapper:
    """Node topology mapper for graph-based component weighting.

    Maps spectral components to nodes in a topology graph,
    enabling graph-based weighting and alignment scoring.

    Args:
        node_graph: Adjacency mapping {node_id: [neighbor_ids]}.
    """

    def __init__(self, node_graph: Optional[Mapping[str, Sequence[str]]] = None) -> None:
        self.node_graph: Dict[str, list] = {k: list(v) for k, v in (node_graph or {}).items()}
        self._graph = None
        if HAS_NETWORKX and self.node_graph:
            g = nx.Graph()
            for node, neighbors in self.node_graph.items():
                g.add_node(node)
                for n in neighbors:
                    g.add_edge(node, n)
            self._graph = g

    def compute_weight(
        self,
        component: SpectralComponent,
        node_weights: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Compute topology-weighted importance for a component.

        Args:
            component: Input spectral component.
            node_weights: Optional node weight overrides.

        Returns:
            Computed weight for the component.
        """
        base = max(component.element_weight, 0.0)
        if not component.node_id:
            return base

        override = float((node_weights or {}).get(component.node_id, 1.0))
        centrality = 1.0
        if self._graph is not None and component.node_id in self._graph:
            centrality += float(nx.degree_centrality(self._graph).get(component.node_id, 0.0))
        return base * override * centrality

    def alignment_score(self, reference: MultiElementRecord, candidate: MultiElementRecord) -> float:
        """Compute node topology alignment between two records.

        Uses Jaccard similarity over node IDs.

        Args:
            reference: Reference record.
            candidate: Candidate record.

        Returns:
            Alignment score in [0, 1].
        """
        ref_nodes = {c.node_id for c in reference.components if c.node_id}
        cand_nodes = {c.node_id for c in candidate.components if c.node_id}
        if not ref_nodes and not cand_nodes:
            return 1.0
        if not ref_nodes or not cand_nodes:
            return 0.0
        inter = len(ref_nodes & cand_nodes)
        union = len(ref_nodes | cand_nodes)
        if union == 0:
            return 0.0
        return inter / union
