"""Tests for the configuration system."""

import numpy as np
import pytest

from mesie.foundation.config.pretraining_config import (
    PretrainingConfig,
    ModelConfig,
    TokenizerConfig,
    DataConfig,
    TrainingConfig,
    ObjectiveConfig,
    EvaluationConfig,
    ModalityType,
    TokenizerType,
    AttentionType,
    ObjectiveType,
    SchedulerType,
    OptimizerType,
    spectral_gpt_tiny,
    spectral_gpt_small,
    spectral_gpt_base,
    spectral_gpt_large,
    spectral_gpt_xl,
)


class TestModalityType:
    """Tests for ModalityType enum."""

    def test_all_modalities_defined(self):
        """All expected modalities should be defined."""
        expected = ["SEISMIC", "VIBRATION", "EEG", "ECG", "AUDIO", "RF", "SYNTHETIC"]
        for name in expected:
            assert hasattr(ModalityType, name)

    def test_modality_values(self):
        """Modality values should be lowercase strings."""
        assert ModalityType.SEISMIC.value == "seismic"
        assert ModalityType.AUDIO.value == "audio"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_creation(self):
        """Should create with sensible defaults."""
        config = ModelConfig()
        assert config.hidden_dim > 0
        assert config.num_layers > 0
        assert config.num_heads > 0
        assert config.max_seq_len > 0

    def test_custom_creation(self):
        """Should accept custom parameters."""
        config = ModelConfig(
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            max_seq_len=2048,
        )
        assert config.hidden_dim == 512
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.max_seq_len == 2048


class TestTokenizerConfig:
    """Tests for TokenizerConfig."""

    def test_default_creation(self):
        config = TokenizerConfig()
        assert config.vqvae.codebook_size > 0
        assert config.vqvae.codebook_dim > 0


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_creation(self):
        config = DataConfig()
        assert config.batch_size > 0
        assert len(config.modalities) > 0


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_creation(self):
        config = TrainingConfig()
        assert config.max_epochs > 0
        assert config.learning_rate > 0
        assert config.warmup_steps >= 0


class TestPretrainingConfig:
    """Tests for full PretrainingConfig."""

    def test_default_creation(self):
        config = PretrainingConfig()
        assert config.model is not None
        assert config.tokenizer is not None
        assert config.data is not None
        assert config.training is not None

    def test_preset_tiny(self):
        config = spectral_gpt_tiny()
        assert config.model.hidden_dim <= 256
        assert config.model.num_layers <= 6

    def test_preset_small(self):
        config = spectral_gpt_small()
        assert config.model.hidden_dim <= 512

    def test_preset_base(self):
        config = spectral_gpt_base()
        assert config.model.hidden_dim >= 512

    def test_preset_large(self):
        config = spectral_gpt_large()
        assert config.model.hidden_dim >= 768

    def test_preset_xl(self):
        config = spectral_gpt_xl()
        assert config.model.hidden_dim >= 1024
