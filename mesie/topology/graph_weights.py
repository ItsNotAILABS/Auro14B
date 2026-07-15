"""Graph-based weight computation for node topologies."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False


def compute_centrality_weights(
    adjacency: Dict[str, Sequence[str]],
    method: str = "degree",
) -> Dict[str, float]:
    """Compute node centrality weights from an adjacency graph.

    Args:
        adjacency: Node adjacency mapping.
        method: Centrality method ('degree', 'betweenness', 'closeness').

    Returns:
        Dictionary mapping node IDs to centrality weights.

    Raises:
        ImportError: If networkx is not available.
        ValueError: If method is not supported.
    """
    if not HAS_NETWORKX:
        # Fallback: uniform weights
        return {k: 1.0 for k in adjacency}

    g = nx.Graph()
    for node, neighbors in adjacency.items():
        g.add_node(node)
        for n in neighbors:
            g.add_edge(node, n)

    if method == "degree":
        centrality = nx.degree_centrality(g)
    elif method == "betweenness":
        centrality = nx.betweenness_centrality(g)
    elif method == "closeness":
        centrality = nx.closeness_centrality(g)
    else:
        raise ValueError(f"Unsupported centrality method: '{method}'")

    return {str(k): float(v) for k, v in centrality.items()}
