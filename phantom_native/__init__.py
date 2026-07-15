"""Phantom Native Stack — Sovereign MESIE Integration.

Zero-dependency native driver layer built on MESIE spectral primitives
(spectral objects, helix encoding, resonance, NeuroCores, TAURUS).
Provides SIMD-vectorized tensor ops, resonance attention kernels,
TAURUS memory management, and swarm runtime orchestration.
"""

from phantom_native.sovereign_tensor import SovereignTensor
from phantom_native.taurus import TaurusMemory
from phantom_native.neurocore import SovereignNeuroCore
from phantom_native.swarm_runtime import SovereignSwarmRuntime

__all__ = [
    "SovereignTensor",
    "TaurusMemory",
    "SovereignNeuroCore",
    "SovereignSwarmRuntime",
]
