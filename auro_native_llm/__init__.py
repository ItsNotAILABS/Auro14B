"""Auro — first-class text LLM family on the MESIE compute engine.

Family: Auro-2B · Auro-4B · Auro-8B · Auro-14B · Auro-100B

Core product:
  ``AuroLanguageModel`` — causal MoE transformer (MESIE SpectralGPT) with
  meaning engines (Latin / Sanskrit / Nahuatl), spectral vector fusion,
  φ-mathematics, train + generate + checkpoints.

Not a wrapper of OpenAI/Anthropic/Ollama. Compute plane = MESIE.
"""

from auro_native_llm.family import (
    get_lane,
    list_model_ids,
    load_family,
    validate_family,
)
from auro_native_llm.mesie_compute import (
    MESIEComputePlane,
    MesieComputeProfile,
    get_compute_plane,
)
from auro_native_llm.model import (
    AuroGenerateResult,
    AuroLanguageModel,
    AuroLMConfig,
    AuroTokenizer,
    TrainConfig,
    family_config,
    load_checkpoint,
    save_checkpoint,
    train_language_model,
)
from auro_native_llm.native_model import AuroNativeFamily, AuroNativeModel, NativeGeneration
from auro_native_llm.native_runtime import AuroNativeRuntime, bootstrap_runtime
from auro_native_llm.scripture import (
    Canon,
    CognitiveLoopResult,
    InnerGovernance,
    ProcessModel,
    RulesEngine,
    ScripturalExecutor,
    ScripturalMemory,
    ScripturalSubstrate,
    StructuredCognitiveLoop,
    load_canon,
)
from auro_native_llm.work import WorkAgent, WorkResult
from auro_native_llm.organism import (
    AuroMind,
    FAMILY_IDS,
    build_family,
    build_mind,
    load_mind,
    save_mind,
)
from auro_native_llm.organism.value_train import ValueTrainConfig, run_value_training
from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter, route_role
from auro_native_llm.types import (
    FAMILY_CONTRACT_VERSION,
    FAMILY_ID,
    FAMILY_PARAMETER_TARGETS,
    ArchitectureSpec,
    FamilyManifest,
    ModelLane,
    ModelTier,
    SubAgentDispatch,
    SubAgentRole,
    SubAgentSpec,
)

__version__ = "0.4.0-alpha"

__all__ = [
    "FAMILY_CONTRACT_VERSION",
    "FAMILY_ID",
    "FAMILY_PARAMETER_TARGETS",
    "ArchitectureSpec",
    "AuroGenerateResult",
    "AuroLMConfig",
    "AuroLanguageModel",
    "AuroNativeFamily",
    "AuroNativeModel",
    "AuroNativeRuntime",
    "AuroTokenizer",
    "FamilyManifest",
    "MESIEComputePlane",
    "MesieComputeProfile",
    "ModelLane",
    "ModelTier",
    "MultiEmbeddedSubAgentRouter",
    "NativeGeneration",
    "SubAgentDispatch",
    "SubAgentRole",
    "SubAgentSpec",
    "Canon",
    "CognitiveLoopResult",
    "InnerGovernance",
    "ProcessModel",
    "RulesEngine",
    "ScripturalExecutor",
    "ScripturalMemory",
    "ScripturalSubstrate",
    "StructuredCognitiveLoop",
    "TrainConfig",
    "AuroMind",
    "FAMILY_IDS",
    "ValueTrainConfig",
    "WorkAgent",
    "WorkResult",
    "bootstrap_runtime",
    "build_family",
    "build_mind",
    "load_mind",
    "run_value_training",
    "save_mind",
    "family_config",
    "get_compute_plane",
    "get_lane",
    "list_model_ids",
    "load_canon",
    "load_checkpoint",
    "load_family",
    "route_role",
    "save_checkpoint",
    "train_language_model",
    "validate_family",
    "__version__",
]
