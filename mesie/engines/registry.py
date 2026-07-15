"""Default engine registry for internal API."""

from __future__ import annotations

from mesie.engines.base import EngineRegistry
from mesie.engines.control_engine import ControlEngine
from mesie.engines.embedding_engine import EmbeddingEngine
from mesie.engines.generation_engine import GenerationEngine
from mesie.engines.intelligence_engine import IntelligenceEngine
from mesie.engines.logic_engine import LogicEngine
from mesie.engines.matching_engine import MatchingEngine
from mesie.engines.movement_engine import MovementEngine
from mesie.engines.validation_engine import ValidationEngine
from mesie.engines.fingerprint_engine import FingerprintEngine
from mesie.engines.hardware_abstraction_engine import HardwareAbstractionEngine
from mesie.engines.scalability_engine import ScalabilityEngine
from mesie.engines.attestation_engine import AttestationEngine
from mesie.engines.reproducibility_engine import ReproducibilityEngine
from mesie.engines.auto_validation_agent import AutoValidationEngine
from mesie.engines.multimodel_julia_engine import (
    MultiModelEmbeddingEngine,
    MultiModelFingerprintEngine,
    MultiModelMatchingEngine,
    MultiModelValidationEngine,
)
from mesie.engines.polyglot_engine import PolyglotEngine
from mesie.engines.research_engine import ResearchEngine
from mesie.engines.workflow_engine import WorkflowEngine
from mesie.internal_api.bus import InternalBus

try:
    from mesie.polyglot.suite import AISVectorPolyglotSuite
except ImportError:
    AISVectorPolyglotSuite = None  # type: ignore


def build_default_registry(
    bus: InternalBus | None = None,
    polyglot_suite: "AISVectorPolyglotSuite | None" = None,
) -> EngineRegistry:
    """Register all built-in engines; attach workflow engine to bus."""
    registry = EngineRegistry()
    bus = bus or InternalBus()

    engines = [
        EmbeddingEngine(),
        MatchingEngine(),
        GenerationEngine(),
        ValidationEngine(),
        IntelligenceEngine(),
        ControlEngine(),
        MovementEngine(),
        LogicEngine(),
        FingerprintEngine(),
        PolyglotEngine(suite=polyglot_suite),
        # Verification and trust engines
        HardwareAbstractionEngine(),
        ScalabilityEngine(),
        AttestationEngine(),
        ReproducibilityEngine(),
        AutoValidationEngine(),
        # Research Agent engine
        ResearchEngine(),
        # Multi-model Python↔Julia engines
        MultiModelValidationEngine(),
        MultiModelMatchingEngine(),
        MultiModelEmbeddingEngine(),
        MultiModelFingerprintEngine(),
    ]
    workflow = WorkflowEngine(bus=bus)
    engines.append(workflow)

    for eng in engines:
        registry.register(eng)
        bus.register_engine(eng.name, eng.handle)

    return registry