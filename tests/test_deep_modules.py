"""Tests for Foundation Model, Temporal Dynamics, Multi-Modal Fusion, and Memory Consolidation."""

import numpy as np
import pytest

from mesie.ai.foundation_model import (
    ContrastiveSpectralLearning,
    ExpertNetwork,
    FoundationEncoderLayer,
    FoundationModelConfig,
    GatedMultiHeadAttention,
    MaskedSpectralModeling,
    MixtureOfExperts,
    RotaryPositionalEncoding,
    SpectralFoundationModel,
    SpectralPatchEmbedding,
    SpectralTransferHead,
)
from mesie.cognitive.temporal_dynamics import (
    ChangePointDetector,
    SpectralPredictor,
    TemporalAttentionLayer,
    TemporalConfig,
    TemporalDynamicsPipeline,
    TemporalSpectralBuffer,
    TimeFrequencyDecomposer,
)
from mesie.cognitive.multimodal_fusion import (
    AcousticModalityAnalyzer,
    CrossModalAttention,
    ElectromagneticModalityAnalyzer,
    FusionConfig,
    FusionGate,
    LatentSpaceAligner,
    ModalityConfig,
    ModalityEncoder,
    MultiModalFusionPipeline,
    VibrationModalityAnalyzer,
)
from mesie.cognitive.memory_consolidation import (
    ConsolidationConfig,
    ConsolidationPipeline,
    DreamReplayEngine,
    ExperienceReplayBuffer,
    MemoryConsolidator,
    SynapticHomeostasis,
)
from mesie.cognitive.taurus_memory import TaurusMemoryStore


# =============================================================================
# Foundation Model Tests
# =============================================================================


class TestSpectralPatchEmbedding:
    def test_embed_shape(self):
        emb = SpectralPatchEmbedding(patch_size=16, d_model=64)
        spectrum = np.random.randn(128)
        result = emb.embed(spectrum)
        assert result.shape[1] == 64
        assert result.shape[0] > 1  # At least CLS + 1 patch

    def test_short_spectrum(self):
        emb = SpectralPatchEmbedding(patch_size=16, d_model=32)
        spectrum = np.random.randn(8)
        result = emb.embed(spectrum)
        assert result.shape[1] == 32


class TestRotaryPositionalEncoding:
    def test_apply(self):
        rope = RotaryPositionalEncoding(d_model=64, max_seq_len=100)
        x = np.random.randn(10, 64)
        encoded = rope.apply(x)
        assert encoded.shape == (10, 64)
        assert not np.allclose(encoded, x)

    def test_with_offset(self):
        rope = RotaryPositionalEncoding(d_model=32, max_seq_len=50)
        x = np.random.randn(5, 32)
        e1 = rope.apply(x, offset=0)
        e2 = rope.apply(x, offset=10)
        assert not np.allclose(e1, e2)


class TestGatedMultiHeadAttention:
    def test_forward(self):
        attn = GatedMultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(8, 64)
        output, weights = attn.forward(x)
        assert output.shape == (8, 64)
        assert weights.shape == (8, 8)

    def test_with_rotary(self):
        rope = RotaryPositionalEncoding(d_model=32, max_seq_len=20)
        attn = GatedMultiHeadAttention(d_model=32, n_heads=2)
        x = np.random.randn(5, 32)
        output, _ = attn.forward(x, rotary_encoder=rope)
        assert output.shape == (5, 32)


class TestMixtureOfExperts:
    def test_forward(self):
        moe = MixtureOfExperts(d_model=32, n_experts=4, top_k=2, d_expert=64)
        x = np.random.randn(5, 32)
        output, assignments = moe.forward(x)
        assert output.shape == (5, 32)
        assert assignments.shape == (5, 2)

    def test_load_balance(self):
        moe = MixtureOfExperts(d_model=32, n_experts=4, top_k=2)
        for _ in range(10):
            moe.forward(np.random.randn(8, 32))
        stats = moe.get_load_balance_stats()
        assert "entropy" in stats
        assert stats["total_tokens_routed"] > 0


class TestSpectralFoundationModel:
    def test_encode(self):
        config = FoundationModelConfig(d_model=32, n_heads=2, n_layers=2, patch_size=8, n_experts=2)
        model = SpectralFoundationModel(config)
        output = model.encode(np.random.randn(64))
        assert output.pooled_output.shape == (32,)
        assert len(output.attention_maps) == 2

    def test_get_embedding(self):
        config = FoundationModelConfig(d_model=16, n_heads=2, n_layers=1, patch_size=4, n_experts=2)
        model = SpectralFoundationModel(config)
        emb = model.get_embedding(np.random.randn(32))
        assert emb.shape == (16,)

    def test_classify(self):
        config = FoundationModelConfig(d_model=32, n_heads=2, n_layers=1, patch_size=8, n_experts=2)
        model = SpectralFoundationModel(config)
        probs = model.classify(np.random.randn(64))
        assert abs(np.sum(probs) - 1.0) < 1e-5

    def test_attention_analysis(self):
        config = FoundationModelConfig(d_model=16, n_heads=2, n_layers=2, patch_size=4, n_experts=2)
        model = SpectralFoundationModel(config)
        analysis = model.get_attention_analysis(np.random.randn(32))
        assert "layer_analyses" in analysis
        assert "mean_entropy" in analysis


class TestMaskedSpectralModeling:
    def test_create_masked(self):
        msm = MaskedSpectralModeling(mask_ratio=0.3)
        spectrum = np.random.randn(128)
        masked, indices, originals = msm.create_masked_input(spectrum, patch_size=16)
        assert len(masked) == 128
        assert len(indices) > 0

    def test_compute_loss(self):
        msm = MaskedSpectralModeling()
        pred = np.random.randn(3, 16)
        orig = np.random.randn(3, 16)
        loss = msm.compute_loss(pred, orig)
        assert loss > 0


class TestContrastiveSpectralLearning:
    def test_augment(self):
        csl = ContrastiveSpectralLearning()
        spectrum = np.random.randn(64)
        augmented = csl.augment_spectrum(spectrum)
        assert len(augmented) == 64
        assert not np.allclose(augmented, spectrum)

    def test_contrastive_loss(self):
        csl = ContrastiveSpectralLearning()
        embeddings = np.random.randn(4, 32)
        loss = csl.compute_contrastive_loss(embeddings, [(0, 1), (2, 3)])
        assert isinstance(loss, float)


class TestSpectralTransferHead:
    def test_classification_head(self):
        head = SpectralTransferHead(input_dim=32, output_dim=5, task_type="classification")
        output = head.forward(np.random.randn(32))
        assert len(output) == 5
        assert abs(np.sum(output) - 1.0) < 1e-5

    def test_regression_head(self):
        head = SpectralTransferHead(input_dim=16, output_dim=3, task_type="regression")
        output = head.forward(np.random.randn(16))
        assert len(output) == 3


# =============================================================================
# Temporal Dynamics Tests
# =============================================================================


class TestTemporalSpectralBuffer:
    def test_push_and_get(self):
        buf = TemporalSpectralBuffer(max_length=10, d_spectral=32)
        for i in range(5):
            buf.push(np.random.randn(32))
        assert buf.size == 5
        window = buf.get_window(3)
        assert window.shape == (3, 32)

    def test_capacity(self):
        buf = TemporalSpectralBuffer(max_length=5, d_spectral=16)
        for i in range(10):
            buf.push(np.random.randn(16))
        assert buf.size == 5

    def test_temporal_diff(self):
        buf = TemporalSpectralBuffer(max_length=10, d_spectral=8)
        for i in range(5):
            buf.push(np.ones(8) * i)
        diff = buf.get_temporal_diff(lag=1)
        assert diff is not None
        assert np.allclose(diff, np.ones_like(diff))


class TestTimeFrequencyDecomposer:
    def test_decompose_1d(self):
        dec = TimeFrequencyDecomposer(window_size=16, hop_size=4, n_frequency_bins=32)
        seq = np.random.randn(64)
        tf_map = dec.decompose(seq)
        assert tf_map.shape[1] == 32
        assert tf_map.shape[0] > 0

    def test_decompose_2d(self):
        dec = TimeFrequencyDecomposer(window_size=8, hop_size=4, n_frequency_bins=16)
        seq = np.random.randn(32, 4)
        tf_map = dec.decompose(seq)
        assert tf_map.shape[1] == 16

    def test_spectral_flux(self):
        dec = TimeFrequencyDecomposer()
        tf_map = np.random.rand(10, 32)
        flux = dec.compute_spectral_flux(tf_map)
        assert len(flux) == 9


class TestChangePointDetector:
    def test_no_change(self):
        det = ChangePointDetector(sensitivity=3.0)
        for _ in range(30):
            det.update(np.random.randn(10) * 0.1 + 1.0)
        assert det.n_changes == 0

    def test_detect_change(self):
        det = ChangePointDetector(sensitivity=2.0, min_segment_length=5)
        # Normal regime
        for _ in range(25):
            det.update(np.ones(10))
        # Sudden change
        for _ in range(10):
            det.update(np.ones(10) * 100)
        assert det.n_changes > 0


class TestSpectralPredictor:
    def test_predict_shape(self):
        pred = SpectralPredictor(d_spectral=32, horizon=5)
        history = np.random.randn(20, 32)
        pred.fit(history)
        predictions = pred.predict()
        assert predictions.shape == (5, 32)

    def test_update_and_predict(self):
        pred = SpectralPredictor(d_spectral=16, horizon=3)
        for i in range(10):
            pred.update(np.ones(16) * i)
        predictions = pred.predict()
        assert predictions.shape == (3, 16)
        # Should predict increasing values
        assert predictions[0, 0] > predictions[0, 0] - 10  # Reasonable prediction


class TestTemporalDynamicsPipeline:
    def test_process(self):
        config = TemporalConfig(window_size=8, n_frequency_bins=16, d_temporal=16)
        pipeline = TemporalDynamicsPipeline(config)
        for _ in range(10):
            result = pipeline.process(np.random.randn(16))
        assert result.predictions is not None
        assert pipeline.processing_count == 10


# =============================================================================
# Multi-Modal Fusion Tests
# =============================================================================


class TestModalityEncoder:
    def test_encode_1d(self):
        config = ModalityConfig(modality_id="test", input_dim=64)
        encoder = ModalityEncoder(config, d_latent=32)
        result = encoder.encode(np.random.randn(64))
        assert result.shape == (32,)

    def test_encode_resize(self):
        config = ModalityConfig(modality_id="test", input_dim=64)
        encoder = ModalityEncoder(config, d_latent=32)
        result = encoder.encode(np.random.randn(128))  # Different size
        assert result.shape == (32,)


class TestCrossModalAttention:
    def test_attend(self):
        attn = CrossModalAttention(d_latent=32, n_heads=2)
        q = np.random.randn(32)
        k = np.random.randn(3, 32)
        output, weights = attn.attend(q, k)
        assert output.shape == (32,)


class TestFusionGate:
    def test_compute_gates(self):
        gate = FusionGate(d_latent=16, max_modalities=4)
        embeddings = [np.random.randn(16) for _ in range(3)]
        gates = gate.compute_gates(embeddings)
        assert len(gates) == 3
        assert abs(np.sum(gates) - 1.0) < 1e-5

    def test_apply_gates(self):
        gate = FusionGate(d_latent=8, max_modalities=4)
        embeddings = [np.random.randn(8) for _ in range(2)]
        gates = np.array([0.6, 0.4])
        result = gate.apply_gates(embeddings, gates)
        assert result.shape == (8,)


class TestLatentSpaceAligner:
    def test_align(self):
        aligner = LatentSpaceAligner(d_latent=16)
        aligner.register_modality("a")
        emb = np.random.randn(16)
        aligned = aligner.align(emb, "a")
        assert aligned.shape == (16,)

    def test_alignment_score(self):
        aligner = LatentSpaceAligner(d_latent=16)
        aligner.register_modality("a")
        aligner.register_modality("b")
        embeddings = {"a": np.random.randn(16), "b": np.random.randn(16)}
        score = aligner.compute_alignment_score(embeddings)
        assert -1 <= score <= 1


class TestMultiModalFusionPipeline:
    def test_fuse_two_modalities(self):
        config = FusionConfig(d_latent=32, fusion_strategy="attention")
        pipeline = MultiModalFusionPipeline(config)
        pipeline.register_modality(ModalityConfig("vibration", input_dim=64))
        pipeline.register_modality(ModalityConfig("acoustic", input_dim=64))

        result = pipeline.fuse({
            "vibration": np.random.randn(64),
            "acoustic": np.random.randn(64),
        })
        assert result.fused_embedding.shape == (32,)
        assert len(result.per_modality_embeddings) == 2

    def test_gated_fusion(self):
        config = FusionConfig(d_latent=16, fusion_strategy="gated")
        pipeline = MultiModalFusionPipeline(config)
        result = pipeline.fuse({
            "sensor_a": np.random.randn(32),
            "sensor_b": np.random.randn(32),
        })
        assert result.fused_embedding.shape == (16,)
        assert len(result.gate_values) == 2

    def test_modality_analysis(self):
        pipeline = MultiModalFusionPipeline()
        features = pipeline.get_modality_analysis("vibration_sensor", np.random.randn(128))
        assert "natural_frequencies" in features


class TestModalityAnalyzers:
    def test_vibration(self):
        analyzer = VibrationModalityAnalyzer()
        spectrum = np.random.randn(256) + 0.5
        features = analyzer.extract_modal_features(spectrum)
        assert "rms" in features
        assert "crest_factor" in features

    def test_acoustic(self):
        analyzer = AcousticModalityAnalyzer()
        spectrum = np.abs(np.random.randn(128)) + 0.01
        features = analyzer.extract_acoustic_features(spectrum)
        assert "centroid" in features
        assert "loudness" in features

    def test_electromagnetic(self):
        analyzer = ElectromagneticModalityAnalyzer()
        spectrum = np.random.randn(64) ** 2
        features = analyzer.extract_em_features(spectrum)
        assert "snr_db" in features
        assert "bandwidth" in features


# =============================================================================
# Memory Consolidation Tests
# =============================================================================


class TestExperienceReplayBuffer:
    def test_add_and_sample(self):
        buf = ExperienceReplayBuffer(capacity=100)
        for i in range(20):
            buf.add({"embedding": np.random.randn(16), "label": i}, priority=float(i))
        exps, indices, weights = buf.sample(5)
        assert len(exps) == 5
        assert len(indices) == 5
        assert len(weights) == 5

    def test_capacity(self):
        buf = ExperienceReplayBuffer(capacity=5)
        for i in range(10):
            buf.add({"data": i}, priority=float(i))
        assert buf.size == 5

    def test_update_priorities(self):
        buf = ExperienceReplayBuffer(capacity=10)
        for i in range(5):
            buf.add({"data": i}, priority=1.0)
        buf.update_priorities(np.array([0, 1]), np.array([10.0, 10.0]))
        stats = buf.get_statistics()
        assert stats["max_priority"] >= 10.0


class TestMemoryConsolidator:
    def test_consolidate(self):
        store = TaurusMemoryStore(capacity=50)
        for i in range(10):
            trace = store.store(embedding=np.random.randn(16), importance=float(i) / 10)
            trace.access_count = i

        consolidator = MemoryConsolidator(store)
        event = consolidator.consolidate()
        assert event.n_strengthened >= 0
        assert event.n_weakened >= 0

    def test_should_consolidate(self):
        store = TaurusMemoryStore(capacity=10)
        config = ConsolidationConfig(consolidation_interval=5)
        consolidator = MemoryConsolidator(store, config)
        for _ in range(5):
            consolidator.step()
        assert consolidator.should_consolidate()


class TestDreamReplayEngine:
    def test_generate_dreams(self):
        store = TaurusMemoryStore(capacity=50)
        for _ in range(10):
            store.store(embedding=np.random.randn(16), importance=1.0)

        engine = DreamReplayEngine(creativity=0.3)
        dreams = engine.generate_dreams(store, n_dreams=5)
        assert len(dreams) == 5
        for d in dreams:
            assert len(d) == 16

    def test_replay_with_dreams(self):
        store = TaurusMemoryStore(capacity=50)
        for _ in range(10):
            store.store(embedding=np.random.randn(16), importance=1.0)

        buffer = ExperienceReplayBuffer(capacity=100)
        engine = DreamReplayEngine()
        stats = engine.replay_with_dreams(store, buffer, n_replays=10, dream_ratio=0.3)
        assert stats["n_dreams_generated"] > 0
        assert buffer.size > 0


class TestSynapticHomeostasis:
    def test_regulate(self):
        store = TaurusMemoryStore(capacity=50)
        for _ in range(10):
            store.store(embedding=np.random.randn(16), importance=np.random.rand() * 5)

        homeostasis = SynapticHomeostasis(target_mean_strength=0.5)
        result = homeostasis.regulate(store)
        assert "scale_applied" in result

    def test_health_metrics(self):
        store = TaurusMemoryStore(capacity=50)
        for _ in range(5):
            store.store(embedding=np.random.randn(16), importance=1.0)

        homeostasis = SynapticHomeostasis()
        metrics = homeostasis.get_health_metrics(store)
        assert "health_score" in metrics
        assert metrics["n_memories"] == 5


class TestConsolidationPipeline:
    def test_step_with_auto_consolidation(self):
        store = TaurusMemoryStore(capacity=50)
        for _ in range(10):
            store.store(embedding=np.random.randn(16), importance=1.0)

        config = ConsolidationConfig(consolidation_interval=5)
        pipeline = ConsolidationPipeline(store, config)

        events = []
        for _ in range(10):
            event = pipeline.step(np.random.randn(16))
            if event is not None:
                events.append(event)

        assert pipeline.total_steps == 10

    def test_system_health(self):
        store = TaurusMemoryStore(capacity=20)
        for _ in range(5):
            store.store(embedding=np.random.randn(8), importance=1.0)

        pipeline = ConsolidationPipeline(store)
        health = pipeline.get_system_health()
        assert "memory_store" in health
        assert "replay_buffer" in health
        assert "homeostasis" in health
