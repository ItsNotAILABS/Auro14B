"""NeuroEmergence and neuromorphic residual integration for AURO LLMs."""

from auro_native_llm.neuro.emergence import NeuroEmergenceCore, NeuroBridge
from auro_native_llm.neuro.spiking_gate import (
    SpikingGateConfig,
    SpikingGateReceipt,
    SpikingResidualGate,
)

__all__ = [
    "NeuroBridge",
    "NeuroEmergenceCore",
    "SpikingGateConfig",
    "SpikingGateReceipt",
    "SpikingResidualGate",
]
