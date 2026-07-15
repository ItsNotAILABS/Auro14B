"""Tests for the spectral transformer pipeline."""

import numpy as np
import pytest

from mesie.ai.transformer_pipeline import (
    MultiHeadSpectralAttention,
    PositionalEncoder,
    SpectralTokenizer,
    SpectralTransformerPipeline,
    TransformerConfig,
    TransformerEncoderLayer,
    TransformerOutput,
)


class TestSpectralTokenizer:
    """Tests for SpectralTokenizer."""

    def test_bin_tokenize(self):
        tokenizer = SpectralTokenizer(strategy="frequency_bins", n_tokens=32, d_model=64)
        spectrum = np.random.randn(256)
        tokens = tokenizer.tokenize(spectrum)
        assert tokens.shape == (32, 64)

    def test_wavelet_tokenize(self):
        tokenizer = SpectralTokenizer(strategy="wavelets", n_tokens=32, d_model=64)
        spectrum = np.random.randn(256)
        tokens = tokenizer.tokenize(spectrum)
        assert tokens.shape == (32, 64)

    def test_patch_tokenize(self):
        tokenizer = SpectralTokenizer(strategy="patches", n_tokens=16, d_model=32)
        spectrum = np.random.randn(128)
        tokens = tokenizer.tokenize(spectrum)
        assert tokens.shape == (16, 32)

    def test_prepend_cls(self):
        tokenizer = SpectralTokenizer(n_tokens=8, d_model=16)
        tokens = np.random.randn(8, 16)
        with_cls = tokenizer.prepend_cls(tokens)
        assert with_cls.shape == (9, 16)

    def test_short_spectrum(self):
        tokenizer = SpectralTokenizer(n_tokens=64, d_model=32)
        spectrum = np.random.randn(4)
        tokens = tokenizer.tokenize(spectrum)
        assert tokens.shape == (64, 32)


class TestPositionalEncoder:
    """Tests for PositionalEncoder."""

    def test_sinusoidal(self):
        encoder = PositionalEncoder(d_model=64, max_len=128, encoding_type="sinusoidal")
        tokens = np.random.randn(32, 64)
        encoded = encoder.encode(tokens)
        assert encoded.shape == (32, 64)
        # Should differ from input
        assert not np.allclose(encoded, tokens)

    def test_learnable(self):
        encoder = PositionalEncoder(d_model=32, max_len=64, encoding_type="learnable")
        tokens = np.random.randn(16, 32)
        encoded = encoder.encode(tokens)
        assert encoded.shape == (16, 32)


class TestMultiHeadSpectralAttention:
    """Tests for MultiHeadSpectralAttention."""

    def test_forward_shape(self):
        attn = MultiHeadSpectralAttention(d_model=64, n_heads=4)
        x = np.random.randn(16, 64)
        output, weights = attn.forward(x)
        assert output.shape == (16, 64)
        assert weights.shape == (16, 16)

    def test_attention_weights_sum_to_one(self):
        attn = MultiHeadSpectralAttention(d_model=32, n_heads=2)
        x = np.random.randn(8, 32)
        _, weights = attn.forward(x)
        # Each row should sum to ~1 (it's an average of softmax outputs)
        row_sums = np.sum(weights, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)


class TestTransformerEncoderLayer:
    """Tests for TransformerEncoderLayer."""

    def test_forward(self):
        layer = TransformerEncoderLayer(d_model=64, n_heads=4, d_feedforward=256)
        x = np.random.randn(16, 64)
        output, attn_weights = layer.forward(x)
        assert output.shape == (16, 64)
        assert attn_weights.shape == (16, 16)


class TestSpectralTransformerPipeline:
    """Tests for SpectralTransformerPipeline."""

    def test_process_default_config(self):
        config = TransformerConfig(d_model=32, n_heads=4, n_encoder_layers=2, max_seq_len=16)
        pipeline = SpectralTransformerPipeline(config=config)
        spectrum = np.random.randn(64)
        output = pipeline.process(spectrum)
        assert isinstance(output, TransformerOutput)
        assert output.pooled_output.shape == (32,)
        assert len(output.attention_maps) == 2

    def test_cls_pooling(self):
        config = TransformerConfig(d_model=32, n_heads=4, n_encoder_layers=1, max_seq_len=8, pooling="cls")
        pipeline = SpectralTransformerPipeline(config=config)
        spectrum = np.random.randn(32)
        output = pipeline.process(spectrum)
        assert output.pooled_output.shape == (32,)

    def test_max_pooling(self):
        config = TransformerConfig(d_model=16, n_heads=2, n_encoder_layers=1, max_seq_len=8, pooling="max")
        pipeline = SpectralTransformerPipeline(config=config)
        spectrum = np.random.randn(32)
        output = pipeline.process(spectrum)
        assert output.pooled_output.shape == (16,)

    def test_extract_embeddings(self):
        config = TransformerConfig(d_model=64, n_heads=4, n_encoder_layers=2, max_seq_len=16)
        pipeline = SpectralTransformerPipeline(config=config)
        embedding = pipeline.extract_embeddings(np.random.randn(128))
        assert embedding.shape == (64,)

    def test_batch_process(self):
        config = TransformerConfig(d_model=32, n_heads=2, n_encoder_layers=1, max_seq_len=8)
        pipeline = SpectralTransformerPipeline(config=config)
        spectra = [np.random.randn(64) for _ in range(3)]
        outputs = pipeline.batch_process(spectra)
        assert len(outputs) == 3

    def test_attention_analysis(self):
        config = TransformerConfig(d_model=32, n_heads=4, n_encoder_layers=2, max_seq_len=8)
        pipeline = SpectralTransformerPipeline(config=config)
        analysis = pipeline.get_attention_analysis(np.random.randn(32))
        assert analysis["n_layers"] == 2
        assert len(analysis["layer_analyses"]) == 2

    def test_processing_count(self):
        config = TransformerConfig(d_model=16, n_heads=2, n_encoder_layers=1, max_seq_len=4)
        pipeline = SpectralTransformerPipeline(config=config)
        assert pipeline.processing_count == 0
        pipeline.process(np.random.randn(16))
        pipeline.process(np.random.randn(16))
        assert pipeline.processing_count == 2
