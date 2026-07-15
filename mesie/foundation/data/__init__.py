"""Data pipeline for multi-modal spectral pretraining."""

from mesie.foundation.data.datasets import (
    SpectralDataset,
    SeismicDataset,
    VibrationDataset,
    EEGDataset,
    ECGDataset,
    AudioSpectrogramDataset,
    RFSweepDataset,
    SyntheticPhysicsDataset,
    MultiModalSpectralDataset,
)
from mesie.foundation.data.preprocessing import (
    SpectralPreprocessor,
    WindowExtractor,
    FrequencyTransform,
    NormalizationPipeline,
    ArtifactRemoval,
)
from mesie.foundation.data.samplers import (
    ModalityBalancedSampler,
    CurriculumSampler,
    DifficultyAwareSampler,
)

__all__ = [
    "SpectralDataset",
    "SeismicDataset",
    "VibrationDataset",
    "EEGDataset",
    "ECGDataset",
    "AudioSpectrogramDataset",
    "RFSweepDataset",
    "SyntheticPhysicsDataset",
    "MultiModalSpectralDataset",
    "SpectralPreprocessor",
    "WindowExtractor",
    "FrequencyTransform",
    "NormalizationPipeline",
    "ArtifactRemoval",
    "ModalityBalancedSampler",
    "CurriculumSampler",
    "DifficultyAwareSampler",
]
