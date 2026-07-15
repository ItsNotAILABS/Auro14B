"""MESIE AI — Neural spectral models, training, inference, transfer learning, and advanced AI."""

from mesie.ai.models import (
    SpectralAutoencoder,
    SpectralClassifier,
    SpectralTransformer,
)
from mesie.ai.training import TrainingPipeline, TrainingConfig
from mesie.ai.inference import InferenceEngine, PredictionResult
from mesie.ai.transfer import TransferAdapter, DomainAdaptation
from mesie.ai.foundation_model import (
    SpectralFoundationModel,
    SpectralPatchEmbedding,
    RotaryPositionalEncoding,
    GatedMultiHeadAttention,
    MixtureOfExperts,
    MaskedSpectralModeling,
    ContrastiveSpectralLearning,
    SpectralTransferHead,
)
from mesie.ai.meta_learning import (
    PrototypicalNetwork,
    MAMLAdapter,
    TaskDistribution,
    MetaLearningConfig,
    MetaTask,
    MetaResult,
    MetaStrategy,
)
from mesie.ai.bayesian import (
    BayesianSpectralNetwork,
    CalibrationModule,
    EnsemblePredictor,
    BayesianConfig,
    UncertaintyEstimate,
    UncertaintyType,
)
from mesie.ai.generative import (
    SpectralVAE,
    SpectralDiffusion,
    SpectralGAN,
    VAEConfig,
    DiffusionConfig,
    GenerationResult,
    GenerativeModelType,
)
from mesie.ai.explainability import (
    PerturbationExplainer,
    GradientExplainer,
    CounterfactualExplainer,
    SpectralAttentionVisualizer,
    Explanation,
    ExplanationType,
    SpectralFeatureImportance,
)
from mesie.ai.time_series import (
    AutoregressiveForecaster,
    SpectralDecompositionForecaster,
    NeuralForecaster,
    EnsembleForecaster,
    ForecastConfig,
    ForecastResult,
    ForecastMethod,
)
from mesie.ai.reinforcement import (
    QLearningAgent,
    PolicyGradientAgent,
    MultiArmedBandit,
    SpectralEnvironment,
    RLConfig,
    Experience,
    AgentMetrics,
    AgentType,
)
from mesie.ai.local_models import (
    LocalModelRegistry,
    LocalModelBackend,
    BackendConfig,
    BackendKind,
    InferenceMode,
    InferenceResult,
    EmbeddingResult,
    StreamChunk,
    StopCondition,
    SpectralInferenceContext,
    SovereignBackend,
    OllamaBackend,
    LlamaCppBackend,
    register_backend,
)

__all__ = [
    # Core models
    "ContrastiveSpectralLearning",
    "DomainAdaptation",
    "GatedMultiHeadAttention",
    "InferenceEngine",
    "MaskedSpectralModeling",
    "MixtureOfExperts",
    "PredictionResult",
    "RotaryPositionalEncoding",
    "SpectralAutoencoder",
    "SpectralClassifier",
    "SpectralFoundationModel",
    "SpectralPatchEmbedding",
    "SpectralTransferHead",
    "SpectralTransformer",
    "TrainingConfig",
    "TrainingPipeline",
    "TransferAdapter",
    # Meta-learning
    "PrototypicalNetwork",
    "MAMLAdapter",
    "TaskDistribution",
    "MetaLearningConfig",
    "MetaTask",
    "MetaResult",
    "MetaStrategy",
    # Bayesian
    "BayesianSpectralNetwork",
    "CalibrationModule",
    "EnsemblePredictor",
    "BayesianConfig",
    "UncertaintyEstimate",
    "UncertaintyType",
    # Generative
    "SpectralVAE",
    "SpectralDiffusion",
    "SpectralGAN",
    "VAEConfig",
    "DiffusionConfig",
    "GenerationResult",
    "GenerativeModelType",
    # Explainability
    "PerturbationExplainer",
    "GradientExplainer",
    "CounterfactualExplainer",
    "SpectralAttentionVisualizer",
    "Explanation",
    "ExplanationType",
    "SpectralFeatureImportance",
    # Time series
    "AutoregressiveForecaster",
    "SpectralDecompositionForecaster",
    "NeuralForecaster",
    "EnsembleForecaster",
    "ForecastConfig",
    "ForecastResult",
    "ForecastMethod",
    # Reinforcement learning
    "QLearningAgent",
    "PolicyGradientAgent",
    "MultiArmedBandit",
    "SpectralEnvironment",
    "RLConfig",
    "Experience",
    "AgentMetrics",
    "AgentType",
    # Local models
    "LocalModelRegistry",
    "LocalModelBackend",
    "BackendConfig",
    "BackendKind",
    "InferenceMode",
    "InferenceResult",
    "EmbeddingResult",
    "StreamChunk",
    "StopCondition",
    "SpectralInferenceContext",
    "SovereignBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "register_backend",
]
