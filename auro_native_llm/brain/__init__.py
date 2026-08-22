"""Permanent HIM brain APIs plus BRAIN-AI/SOLUS compatibility exports."""

from auro_native_llm.brain.organs import (
    MiniBrainCluster,
    DomainBrain,
    OrganismHeart,
    CurriculumTeacher,
    build_brain_cluster,
)
from auro_native_llm.brain.fused import BrainRegion
from auro_native_llm.brain.neuromorphic_brain import HIMBrain, BrainCycle
from auro_native_llm.brain.feline_neuromorphic import (
    FelineNeuromorphicEngine,
    NeuromorphicConfig,
    NeuromorphicCycle,
    SpikeRegionState,
    Synapse,
)
from auro_native_llm.brain.timing_plasticity import (
    TimingPlasticityConfig,
    TimingPlasticityController,
    TimingPlasticityReceipt,
)

__all__ = [
    "BrainCycle",
    "BrainRegion",
    "CurriculumTeacher",
    "DomainBrain",
    "FelineNeuromorphicEngine",
    "HIMBrain",
    "MiniBrainCluster",
    "NeuromorphicConfig",
    "NeuromorphicCycle",
    "OrganismHeart",
    "SpikeRegionState",
    "Synapse",
    "TimingPlasticityConfig",
    "TimingPlasticityController",
    "TimingPlasticityReceipt",
    "build_brain_cluster",
]
