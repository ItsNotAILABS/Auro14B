"""AI/Robotics substrate modules for MESIE.

Provides multi-modal signal fusion, recursive spectral memory,
and neuromorphic runtime optimization for embodied AI systems.
"""

from mesie.robotics.multimodal_fusion import (
    MultiModalFusion,
    FusionConfig,
    ModalityStream,
    FusedRepresentation,
)
from mesie.robotics.spectral_memory import (
    SpectralMemory,
    MemoryConfig,
    MemoryEntry,
    QueryResult,
)
from mesie.robotics.neuromorphic_runtime import (
    NeuromorphicRuntime,
    RuntimeConfig,
    SpikeEvent,
    RuntimeMetrics,
)

__all__ = [
    "MultiModalFusion",
    "FusionConfig",
    "ModalityStream",
    "FusedRepresentation",
    "SpectralMemory",
    "MemoryConfig",
    "MemoryEntry",
    "QueryResult",
    "NeuromorphicRuntime",
    "RuntimeConfig",
    "SpikeEvent",
    "RuntimeMetrics",
]
