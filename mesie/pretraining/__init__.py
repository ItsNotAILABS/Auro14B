"""Spectral pretraining via self-supervised world tasks.

This module provides auxiliary pretraining heads and training infrastructure
that force the model to learn spectral reasoning skills:

- Resonance detection and prediction
- Coherence estimation between channels/components
- Harmonic structure classification and reconstruction
- Spectral drift detection and quantification
- Temporal lineage inference from current spectra

Foundation pretraining objectives:
- Masked Spectral Modeling: reconstruct masked frequency bands
- InfoNCE Contrastive Learning: learn invariant spectral representations
- Temporal Prediction: predict future spectral embeddings from past

Agent-level integration:
- Observation encoder with multi-modal concatenation
- Lineage-conditioned encoding for temporal reasoning
- Digital twin simulation environments for agent pretraining
- Spectral memory store with lineage-aware retrieval
- Multi-stage training recipe orchestration
"""

from mesie.pretraining.world_tasks import (
    ResonanceHead,
    CoherenceHead,
    HarmonicStructureHead,
    SpectralDriftHead,
    TemporalLineageHead,
    WorldTaskSuite,
)
from mesie.pretraining.digital_twin import (
    DigitalTwinEnvironment,
    SpectralEntity,
    SpectralStream,
)
from mesie.pretraining.spectral_memory import (
    SpectralMemoryStore,
    MemoryEntry,
    LineageQuery,
)
from mesie.pretraining.training_recipe import (
    TrainingRecipe,
    PretrainingStage,
    EnvironmentStage,
    FineTuningStage,
)
from mesie.pretraining.foundation_objectives import (
    MaskedSpectralModeling,
    MaskConfig,
    InfoNCEContrastiveLoss,
    AugmentationConfig,
    TemporalPrediction,
    TemporalPredictionConfig,
    FoundationObjectiveSuite,
)
from mesie.pretraining.observation_encoder import (
    ObservationEncoder,
    LineageConditionedEncoder,
    SpectralTransform,
    ModalityConfig,
)

__all__ = [
    "AugmentationConfig",
    "CoherenceHead",
    "DigitalTwinEnvironment",
    "EnvironmentStage",
    "FineTuningStage",
    "FoundationObjectiveSuite",
    "HarmonicStructureHead",
    "InfoNCEContrastiveLoss",
    "LineageConditionedEncoder",
    "LineageQuery",
    "MaskConfig",
    "MaskedSpectralModeling",
    "MemoryEntry",
    "ModalityConfig",
    "ObservationEncoder",
    "PretrainingStage",
    "ResonanceHead",
    "SpectralDriftHead",
    "SpectralEntity",
    "SpectralMemoryStore",
    "SpectralStream",
    "SpectralTransform",
    "TemporalLineageHead",
    "TemporalPrediction",
    "TemporalPredictionConfig",
    "TrainingRecipe",
    "WorldTaskSuite",
]
