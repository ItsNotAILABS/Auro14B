"""MESIE processing engines — pluggable units on the internal API bus."""

from mesie.engines.base import Engine, EngineRegistry
from mesie.engines.control_engine import ControlEngine
from mesie.engines.embedding_engine import EmbeddingEngine
from mesie.engines.generation_engine import GenerationEngine
from mesie.engines.intelligence_engine import IntelligenceEngine
from mesie.engines.logic_engine import LogicEngine
from mesie.engines.matching_engine import MatchingEngine
from mesie.engines.movement_engine import MovementEngine
from mesie.engines.hardware_abstraction_engine import HardwareAbstractionEngine
from mesie.engines.scalability_engine import ScalabilityEngine
from mesie.engines.attestation_engine import AttestationEngine
from mesie.engines.reproducibility_engine import ReproducibilityEngine
from mesie.engines.auto_validation_agent import AutoValidationAgent, AutoValidationEngine, run_auto_validation
from mesie.engines.multimodel_julia_engine import (
    MultiModelEmbeddingEngine,
    MultiModelFingerprintEngine,
    MultiModelMatchingEngine,
    MultiModelValidationEngine,
)
from mesie.engines.registry import build_default_registry
from mesie.engines.validation_engine import ValidationEngine
from mesie.engines.workflow_engine import WorkflowEngine

__all__ = [
    "AttestationEngine",
    "AutoValidationAgent",
    "AutoValidationEngine",
    "ControlEngine",
    "EmbeddingEngine",
    "Engine",
    "EngineRegistry",
    "GenerationEngine",
    "HardwareAbstractionEngine",
    "IntelligenceEngine",
    "LogicEngine",
    "MatchingEngine",
    "MovementEngine",
    "MultiModelEmbeddingEngine",
    "MultiModelFingerprintEngine",
    "MultiModelMatchingEngine",
    "MultiModelValidationEngine",
    "ReproducibilityEngine",
    "ScalabilityEngine",
    "ValidationEngine",
    "WorkflowEngine",
    "build_default_registry",
    "run_auto_validation",
]