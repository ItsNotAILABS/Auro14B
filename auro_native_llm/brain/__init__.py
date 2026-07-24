"""Permanent HIM brain APIs plus BRAIN-AI/SOLUS compatibility exports."""

from auro_native_llm.brain.organs import (
    MiniBrainCluster,
    DomainBrain,
    OrganismHeart,
    CurriculumTeacher,
    build_brain_cluster,
)
from auro_native_llm.brain.fused import HIMBrain, BrainCycle, BrainRegion

__all__ = [
    "BrainCycle",
    "BrainRegion",
    "CurriculumTeacher",
    "DomainBrain",
    "HIMBrain",
    "MiniBrainCluster",
    "OrganismHeart",
    "build_brain_cluster",
]
