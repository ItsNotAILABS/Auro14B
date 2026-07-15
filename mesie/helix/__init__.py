"""Vector Helix — helical vector space for spectral intelligence.

The Vector Helix is a novel representation that arranges spectral embeddings
along a helical manifold, combining linear progression (temporal/sequential)
with rotational encoding (spectral phase). This enables:
- Phase-aware nearest-neighbor retrieval
- Helical traversal for evolutionary spectral analysis
- Rotational coherence scoring across embedded vectors
- Multi-resolution unwinding for hierarchical analysis
"""

from mesie.helix.vector_helix import (
    VectorHelix,
    HelixConfig,
    HelixNode,
    HelixTraversalResult,
)
from mesie.helix.helix_encoder import HelixEncoder, HelixProjection
from mesie.helix.helix_retrieval import HelixRetriever, HelixSearchResult

__all__ = [
    "HelixConfig",
    "HelixEncoder",
    "HelixNode",
    "HelixProjection",
    "HelixRetriever",
    "HelixSearchResult",
    "HelixTraversalResult",
    "VectorHelix",
]
