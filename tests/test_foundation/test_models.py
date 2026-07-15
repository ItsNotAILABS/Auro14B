"""Tests for model architectures."""

import numpy as np
import pytest

from mesie.foundation.models.positional_encoding import (
    RotaryEmbedding,
    FrequencyAwarePositionalEncoding,
    SpectralHarmonicEncoding,
    ALiBiEncoding,
    SpectralPositionalEncoding,
)
from mesie.foundation.models.transformer_blocks import (
    RMSNorm,
    SpectralMultiHeadAttention,
    SpectralFeedForward,
    SpectralTransformerBlock,
)
from mesie.foundation.models.spectral_encoder import (
    LearnableDFTLayer,
    WaveletDecompositionLayer,
    HarmonicAttentionLayer,
    OctaveBandPooling,
    SpectralInputEncoder,
)
from mesie.foundation.models.mixture_of_experts import (
    ExpertLayer,
    ModalityAwareRouter,
    MixtureOfExperts,
)
from mesie.foundation.models.output_heads import (
    SpectralReconstructionHead,
    NextWindowPredictionHead,
    ContrastiveProjectionHead,
    ClassificationHead,
)
from mesie.foundation.models.spectral_gpt import SpectralGPT


class TestPositionalEncoding:
    """Tests for positional encoding variants."""

    def test_rotary_shape(self):
        """Rotary encoding should preserve shape."""
        pe = RotaryEmbedding(dim=64, max_seq_len=512)
        x = np.random.randn(2, 128, 64)
        cos, sin = pe.get_cos_sin(128)
        out = pe.apply_rotary(x, cos, sin)
        assert out.shape == x.shape

    def test_frequency_aware_shape(self):
        """Frequency-aware encoding should produce correct shape."""
        pe = FrequencyAwarePositionalEncoding(dim=64, max_seq_len=512)
        out = pe.encode(seq_len=128)
        assert out.shape == (128, 64)

    def test_harmonic_shape(self):
        """Harmonic encoding should produce correct shape."""
        pe = SpectralHarmonicEncoding(dim=64, max_seq_len=512)
        out = pe.encode(seq_len=128)
        assert out.shape == (128, 64)

    def test_alibi_shape(self):
        """ALiBi should produce correct bias shape."""
        pe = ALiBiEncoding(num_heads=8, max_seq_len=512)
        bias = pe.get_bias(128)
        assert bias.shape == (8, 128, 128)


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = np.random.randn(2, 32, 64)
        out = norm.forward(x)
        assert out.shape == x.shape

    def test_normalization(self):
        """RMS of output should be approximately 1."""
        norm = RMSNorm(dim=64)
        x = np.random.randn(2, 32, 64) * 5
        out = norm.forward(x)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.2)


class TestMultiHeadAttention:
    """Tests for spectral multi-head attention."""

    def test_output_shape(self):
        attn = SpectralMultiHeadAttention(hidden_dim=64, num_heads=8)
        x = np.random.randn(2, 32, 64)
        out, _ = attn.forward(x)
        assert out.shape == x.shape

    def test_different_seq_lengths(self):
        attn = SpectralMultiHeadAttention(hidden_dim=64, num_heads=8)
        for seq_len in [16, 32, 64, 128]:
            x = np.random.randn(1, seq_len, 64)
            out, _ = attn.forward(x)
            assert out.shape == (1, seq_len, 64)


class TestSpectralFeedForward:
    """Tests for spectral feed-forward network."""

    def test_output_shape(self):
        ffn = SpectralFeedForward(hidden_dim=64, ffn_dim=256)
        x = np.random.randn(2, 32, 64)
        out = ffn.forward(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    """Tests for full transformer block."""

    def test_output_shape(self):
        block = SpectralTransformerBlock(hidden_dim=64, num_heads=8, ffn_dim=256)
        x = np.random.randn(2, 32, 64)
        out, _ = block.forward(x)
        assert out.shape == x.shape


class TestSpectralEncoder:
    """Tests for spectral input encoder."""

    def test_learnable_dft(self):
        dft = LearnableDFTLayer(input_dim=128, dft_dim=64)
        x = np.random.randn(2, 128)
        out, _ = dft.forward(x)
        assert out.shape == (2, 64)

    def test_wavelet_decomposition(self):
        wavelet = WaveletDecompositionLayer(input_dim=128, num_levels=4)
        x = np.random.randn(2, 128)
        out, _ = wavelet.forward(x)
        assert out.shape[0] == 2

    def test_harmonic_attention(self):
        harm = HarmonicAttentionLayer(input_dim=64, max_harmonics=8)
        x = np.random.randn(2, 32, 64)
        out, _ = harm.forward(x)
        assert out.shape == x.shape

    def test_octave_pooling(self):
        pool = OctaveBandPooling(input_dim=128, num_octaves=8)
        x = np.random.randn(2, 128)
        result = pool.forward(x)
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        assert out.shape[0] == 2

    def test_full_encoder(self):
        encoder = SpectralInputEncoder(input_dim=256, output_dim=128)
        x = np.random.randn(2, 256)
        result = encoder.forward(x)
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        assert out.shape[0] == 2


class TestMixtureOfExperts:
    """Tests for MoE module."""

    def test_expert_output_shape(self):
        expert = ExpertLayer(input_dim=64, hidden_dim=256, output_dim=64)
        x = np.random.randn(2, 32, 64)
        out = expert.forward(x)
        assert out.shape == x.shape

    def test_router_output(self):
        router = ModalityAwareRouter(
            input_dim=64, num_experts=4, num_modalities=7
        )
        x = np.random.randn(2, 32, 64)
        result = router.route(x)
        assert len(result) == 3  # weights, indices, load_balance

    def test_moe_output_shape(self):
        moe = MixtureOfExperts(hidden_dim=64, num_experts=4)
        x = np.random.randn(2, 32, 64)
        out, _ = moe.forward(x)
        assert out.shape == x.shape


class TestOutputHeads:
    """Tests for output heads."""

    def test_reconstruction_head(self):
        head = SpectralReconstructionHead(input_dim=64, output_dim=128)
        x = np.random.randn(2, 32, 64)
        out = head.forward(x)
        assert "magnitude" in out
        assert out["magnitude"].shape == (2, 32, 128)

    def test_next_window_head(self):
        head = NextWindowPredictionHead(input_dim=64, output_dim=128)
        x = np.random.randn(2, 32, 64)
        out = head.forward(x, context=None)
        assert "steps" in out
        assert len(out["steps"]) > 0

    def test_contrastive_head(self):
        head = ContrastiveProjectionHead(input_dim=64, projection_dim=32)
        x = np.random.randn(2, 32, 64)
        out = head.forward(x)
        assert out.shape == (2, 32, 32)

    def test_classification_head(self):
        head = ClassificationHead(input_dim=64, num_classes=10)
        x = np.random.randn(2, 32, 64)
        out = head.forward(x)
        assert out.shape == (2, 10)


class TestSpectralGPT:
    """Tests for the main SpectralGPT model."""

    def test_creation(self):
        model = SpectralGPT(
            vocab_size=1024,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=256,
        )
        assert model is not None

    def test_forward_discrete(self):
        model = SpectralGPT(
            vocab_size=1024,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=256,
        )
        tokens = np.random.randint(0, 1024, (2, 32))
        output = model.forward(token_ids=tokens)
        assert "last_hidden_state" in output
        assert output["last_hidden_state"].shape == (2, 32, 64)

    def test_forward_continuous(self):
        model = SpectralGPT(
            vocab_size=1024,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=256,
            continuous_dim=128,
        )
        x = np.random.randn(2, 32, 128)
        output = model.forward(continuous_input=x)
        assert "last_hidden_state" in output

    def test_get_embeddings(self):
        model = SpectralGPT(
            vocab_size=1024,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=256,
            continuous_dim=128,
        )
        x = np.random.randn(2, 32, 128)
        emb = model.get_embeddings(continuous_input=x)
        assert emb.shape[0] == 2
        assert emb.shape[-1] == 64
