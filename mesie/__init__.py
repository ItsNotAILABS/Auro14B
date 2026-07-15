"""MESIE — Multi-Element Spectral Intelligence Engine.

A modular Python framework for spectral matching, signal generation,
resonance-aware embeddings, and AI-native spectral representation.
"""

__version__ = "0.4.0"

from mesie.core.records import MultiElementRecord, SpectralComponent, SpectralMetadata
from mesie.core.config import GenerationConfig
from mesie.validation.validators import validate_record, ValidationReport
from mesie.io.loaders import load_record
from mesie.processing.normalize import normalize_record
from mesie.matching.matcher import match_records, SpectralMatcher, MatchResult
from mesie.generation.psd import generate_psd
from mesie.generation.fas import generate_fas
from mesie.generation.rotdnn import generate_rotdnn
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.ai.models import SpectralAutoencoder, SpectralClassifier, SpectralTransformer
from mesie.ai.training import TrainingPipeline, TrainingConfig
from mesie.ai.inference import InferenceEngine, PredictionResult
from mesie.ai.transfer import TransferAdapter, DomainAdaptation
from mesie.ai.intelligence_protocols import (
    IntelligenceProtocol,
    IntelligenceConfig,
    IntelligenceLevel,
    ReasoningResult,
    ReasoningStrategy,
    SpectralMemoryBuffer,
    AttentionFocusModule,
)
from mesie.ai.transformer_pipeline import (
    SpectralTransformerPipeline,
    TransformerConfig,
    TransformerOutput,
    SpectralTokenizer,
)
from mesie.protocols.spectral_protocol import SpectralDataProtocol, ProtocolMessage
from mesie.protocols.streaming import StreamingProtocol, StreamBuffer
from mesie.protocols.serialization import SpectralSerializer, SerializationFormat
from mesie.integration.ai_connector import AISystemConnector, ConnectorConfig
from mesie.polyglot import AISVectorPolyglotSuite, SUITE_NAME as AIS_POLYGLOT_SUITE_NAME
from mesie.integration.library_bridge import LibraryBridge, BridgeState
from mesie.integration.pipeline_orchestrator import PipelineOrchestrator, OrchestratorConfig
from mesie.helix.vector_helix import VectorHelix, HelixConfig, HelixNode, HelixTraversalResult
from mesie.helix.helix_encoder import HelixEncoder, HelixProjection
from mesie.helix.helix_retrieval import HelixRetriever, HelixSearchResult
from mesie.cognitive.taurus_memory import (
    TaurusMemoryStore,
    TaurusWorkingMemory,
    MemoryTrace,
)
from mesie.cognitive.neurocores import (
    SpectralNeuroCore,
    NeuroCoreCluster,
    NeuroCoreConfig,
    CoreProcessingResult,
)
from mesie.edge.hz_ladder import HzLadder, FrequencyTier, LadderLink
from mesie.edge.satellite_nodes import (
    SatelliteEdgeNode,
    OrbitalTier,
    EcoHzReference,
    VirtualNodeNetwork,
)
from mesie.edge.edge_protocol import EdgeSpectralProtocol, EdgeMessage, EdgeRoute
from mesie.io.corpus import SpectralCorpus
from mesie.sdk.intelligence_sdk import SpectralIntelligenceSDK
from mesie.sdk.universal_lab_sdk import UniversalLabSDK
from mesie.internal_api import InternalBus, InternalRouter, MessageEnvelope, MessageTopic, EngineResponse
from mesie.engines import Engine, EngineRegistry, build_default_registry
from mesie.octopus import ArmId, OctopusArm, OctopusController, OctopusConfig, OctopusRunReport
from mesie.agentic import (
    AgentNetwork,
    AgentSpawner,
    AgentState,
    GhostAgent,
    GhostConfig,
    GhostResult,
    NetworkTopology,
    SpawnerConfig,
    TaskSpec,
)

__all__ = [
    "__version__",
    "AIS_POLYGLOT_SUITE_NAME",
    "AISVectorPolyglotSuite",
    "AISystemConnector",
    "AgentNetwork",
    "AgentSpawner",
    "AgentState",
    "ArmId",
    "AttentionFocusModule",
    "BridgeState",
    "ConnectorConfig",
    "CoreProcessingResult",
    "DomainAdaptation",
    "EcoHzReference",
    "EdgeMessage",
    "EdgeRoute",
    "EdgeSpectralProtocol",
    "Engine",
    "EngineRegistry",
    "EngineResponse",
    "FrequencyTier",
    "GenerationConfig",
    "GhostAgent",
    "GhostConfig",
    "GhostResult",
    "HelixConfig",
    "HelixEncoder",
    "HelixNode",
    "HelixProjection",
    "HelixRetriever",
    "HelixSearchResult",
    "HelixTraversalResult",
    "HzLadder",
    "InferenceEngine",
    "InternalBus",
    "InternalRouter",
    "IntelligenceConfig",
    "IntelligenceLevel",
    "IntelligenceProtocol",
    "LadderLink",
    "LibraryBridge",
    "MatchResult",
    "MessageEnvelope",
    "MessageTopic",
    "MemoryTrace",
    "MultiElementRecord",
    "NetworkTopology",
    "NeuroCoreCluster",
    "NeuroCoreConfig",
    "OctopusArm",
    "OctopusConfig",
    "OctopusController",
    "OctopusRunReport",
    "OrbitalTier",
    "OrchestratorConfig",
    "PipelineOrchestrator",
    "PredictionResult",
    "ProtocolMessage",
    "ReasoningResult",
    "ReasoningStrategy",
    "SatelliteEdgeNode",
    "SerializationFormat",
    "SpawnerConfig",
    "SpectralAutoencoder",
    "SpectralClassifier",
    "SpectralComponent",
    "SpectralCorpus",
    "SpectralDataProtocol",
    "SpectralIntelligenceSDK",
    "UniversalLabSDK",
    "SpectralMatcher",
    "SpectralMemoryBuffer",
    "SpectralMetadata",
    "SpectralNeuroCore",
    "SpectralSerializer",
    "SpectralTokenizer",
    "SpectralTransformer",
    "SpectralTransformerPipeline",
    "SpectralVectorizer",
    "StreamBuffer",
    "StreamingProtocol",
    "TaskSpec",
    "TaurusMemoryStore",
    "TaurusWorkingMemory",
    "TrainingConfig",
    "TrainingPipeline",
    "TransferAdapter",
    "TransformerConfig",
    "TransformerOutput",
    "ValidationReport",
    "VectorHelix",
    "VirtualNodeNetwork",
    "build_default_registry",
    "generate_fas",
    "generate_psd",
    "generate_rotdnn",
    "load_record",
    "match_records",
    "normalize_record",
    "validate_record",
]