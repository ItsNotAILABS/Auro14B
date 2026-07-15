"""SpectralGPT model architectures for foundation-model pretraining."""

from mesie.foundation.models.spectral_gpt import SpectralGPT
from mesie.foundation.models.transformer_blocks import (
    SpectralTransformerBlock,
    SpectralMultiHeadAttention,
    SpectralFeedForward,
    FrequencyBandAttention,
    MultiScaleSpectralAttention,
)
from mesie.foundation.models.positional_encoding import (
    SpectralPositionalEncoding,
    RotaryEmbedding,
    FrequencyAwarePositionalEncoding,
    SpectralHarmonicEncoding,
    ALiBiEncoding,
)
from mesie.foundation.models.spectral_encoder import (
    SpectralInputEncoder,
    LearnableDFTLayer,
    WaveletDecompositionLayer,
    HarmonicAttentionLayer,
    OctaveBandPooling,
)
from mesie.foundation.models.mixture_of_experts import (
    MixtureOfExperts,
    ExpertLayer,
    TopKRouter,
    ModalityAwareRouter,
)
from mesie.foundation.models.output_heads import (
    SpectralReconstructionHead,
    NextWindowPredictionHead,
    ContrastiveProjectionHead,
    ClassificationHead,
    MultiTaskHead,
)

__all__ = [
    "SpectralGPT",
    "SpectralTransformerBlock",
    "SpectralMultiHeadAttention",
    "SpectralFeedForward",
    "FrequencyBandAttention",
    "MultiScaleSpectralAttention",
    "SpectralPositionalEncoding",
    "RotaryEmbedding",
    "FrequencyAwarePositionalEncoding",
    "SpectralHarmonicEncoding",
    "ALiBiEncoding",
    "SpectralInputEncoder",
    "LearnableDFTLayer",
    "WaveletDecompositionLayer",
    "HarmonicAttentionLayer",
    "OctaveBandPooling",
    "MixtureOfExperts",
    "ExpertLayer",
    "TopKRouter",
    "ModalityAwareRouter",
    "SpectralReconstructionHead",
    "NextWindowPredictionHead",
    "ContrastiveProjectionHead",
    "ClassificationHead",
    "MultiTaskHead",
]
