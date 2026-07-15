"""Tests for mesie.ai.local_models — on-device inference backends."""

import numpy as np
import pytest

from mesie.ai.local_models import (
    BackendConfig,
    BackendKind,
    EmbeddingResult,
    InferenceMode,
    InferenceResult,
    LlamaCppBackend,
    LocalModelBackend,
    LocalModelRegistry,
    OllamaBackend,
    SovereignBackend,
    SpectralInferenceContext,
    StopCondition,
    StreamChunk,
    register_backend,
)


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_defaults(self):
        config = BackendConfig()
        assert config.kind == BackendKind.SOVEREIGN
        assert config.model_name == "mesie-neurocore-v1"
        assert config.host == "127.0.0.1"
        assert config.port == 11434
        assert config.d_model == 128
        assert config.n_heads == 8
        assert config.temperature == 0.7
        assert config.seed is None

    def test_custom_config(self):
        config = BackendConfig(
            kind=BackendKind.OLLAMA,
            model_name="llama3",
            port=11435,
            temperature=0.0,
            seed=42,
        )
        assert config.kind == BackendKind.OLLAMA
        assert config.model_name == "llama3"
        assert config.port == 11435
        assert config.seed == 42


class TestSpectralInferenceContext:
    """Tests for SpectralInferenceContext — MESIE-native context."""

    def test_empty_context(self):
        ctx = SpectralInferenceContext()
        assert ctx.record_id == ""
        assert ctx.frequency_range == (0.0, 0.0)
        assert ctx.peak_frequencies == []

    def test_context_block_format(self):
        ctx = SpectralInferenceContext(
            record_id="REC-001",
            frequency_range=(0.5, 50.0),
            peak_frequencies=[5.0, 12.5, 25.0],
            component_names=["north", "east", "vertical"],
            domain="frequency",
        )
        block = ctx.to_context_block()
        assert "[SPECTRAL CONTEXT]" in block
        assert "[/SPECTRAL CONTEXT]" in block
        assert "REC-001" in block
        assert "5.00" in block
        assert "north" in block

    def test_context_with_embedding(self):
        embedding = np.random.randn(64)
        ctx = SpectralInferenceContext(
            record_id="REC-002",
            embedding=embedding,
        )
        block = ctx.to_context_block()
        assert "embedding_dim: 64" in block
        assert "embedding_norm:" in block

    def test_from_record(self):
        """Test building context from a mock MultiElementRecord."""

        class MockComponent:
            name = "comp_x"
            frequency = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
            amplitude = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
            domain = "frequency"

        class MockRecord:
            record_id = "MOCK-001"
            components = [MockComponent()]
            metadata = {"source": "test"}

        ctx = SpectralInferenceContext.from_record(MockRecord())
        assert ctx.record_id == "MOCK-001"
        assert ctx.frequency_range == (1.0, 50.0)
        assert len(ctx.peak_frequencies) > 0
        assert "comp_x" in ctx.component_names
        assert ctx.domain == "frequency"


class TestSovereignBackend:
    """Tests for the MESIE-native SovereignBackend."""

    def test_kind(self):
        backend = SovereignBackend()
        assert backend.kind == BackendKind.SOVEREIGN

    def test_health_check_always_true(self):
        """Sovereign backend has zero deps — always healthy."""
        backend = SovereignBackend()
        config = BackendConfig()
        assert backend.health_check(config) is True

    def test_infer_basic(self):
        backend = SovereignBackend()
        config = BackendConfig(d_model=64, n_heads=4)
        result = backend.infer("Classify this spectral pattern", config)
        assert isinstance(result, InferenceResult)
        assert result.text != ""
        assert result.confidence > 0.0
        assert result.latency_ms > 0.0
        assert result.model_name == "mesie-neurocore-v1"
        assert result.stop_condition == StopCondition.END_TOKEN

    def test_infer_with_spectral_context(self):
        backend = SovereignBackend()
        config = BackendConfig(d_model=64)
        ctx = SpectralInferenceContext(
            record_id="TEST-001",
            frequency_range=(0.1, 100.0),
            peak_frequencies=[5.0, 15.0, 45.0],
            domain="frequency",
        )
        result = backend.infer("Analyze anomaly", config, context=ctx)
        assert isinstance(result, InferenceResult)
        assert result.spectral_context is ctx
        assert result.text != ""

    def test_embed_deterministic(self):
        """Sovereign embedding must be deterministic (MESIE reproducibility)."""
        backend = SovereignBackend()
        config = BackendConfig(d_model=64)
        r1 = backend.embed("test spectral signal", config)
        r2 = backend.embed("test spectral signal", config)
        assert isinstance(r1, EmbeddingResult)
        assert r1.dimensions == 64
        assert len(r1.vector) == 64
        # Must be deterministic
        np.testing.assert_array_equal(r1.vector, r2.vector)

    def test_embed_normalized(self):
        """Embedding vectors should be L2-normalized."""
        backend = SovereignBackend()
        config = BackendConfig(d_model=32)
        result = backend.embed("normalize this", config)
        norm = np.linalg.norm(result.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_embed_different_inputs_differ(self):
        backend = SovereignBackend()
        config = BackendConfig(d_model=64)
        r1 = backend.embed("signal alpha", config)
        r2 = backend.embed("signal beta", config)
        assert not np.array_equal(r1.vector, r2.vector)

    def test_stream_fallback(self):
        """Stream defaults to single-shot infer."""
        backend = SovereignBackend()
        config = BackendConfig(d_model=32)
        chunks = list(backend.stream("test prompt", config))
        assert len(chunks) == 1
        assert chunks[0].done is True
        assert chunks[0].text != ""


class TestOllamaBackend:
    """Tests for OllamaBackend (network-dependent tests are skipped)."""

    def test_kind(self):
        backend = OllamaBackend()
        assert backend.kind == BackendKind.OLLAMA

    def test_health_check_offline(self):
        """When Ollama isn't running, health check returns False."""
        backend = OllamaBackend()
        config = BackendConfig(host="127.0.0.1", port=19999, timeout_s=1.0)
        assert backend.health_check(config) is False

    def test_infer_offline_returns_error(self):
        """Graceful failure when Ollama is unavailable."""
        backend = OllamaBackend()
        config = BackendConfig(
            kind=BackendKind.OLLAMA,
            model_name="llama3",
            port=19999,
            timeout_s=1.0,
        )
        result = backend.infer("test", config)
        assert result.stop_condition == StopCondition.ERROR
        assert result.confidence == 0.0
        assert "error" in result.metadata


class TestLlamaCppBackend:
    """Tests for LlamaCppBackend."""

    def test_kind(self):
        backend = LlamaCppBackend()
        assert backend.kind == BackendKind.LLAMA_CPP

    def test_health_check_no_model(self):
        """Without llama-cpp-python installed, returns False."""
        backend = LlamaCppBackend()
        config = BackendConfig(model_name="/nonexistent/model.gguf")
        # Will be False because either llama_cpp not installed or model doesn't exist
        assert backend.health_check(config) is False


class TestLocalModelRegistry:
    """Tests for LocalModelRegistry — mirrors EngineRegistry pattern."""

    def test_default_registry(self):
        registry = LocalModelRegistry()
        backends = registry.available_backends()
        assert BackendKind.SOVEREIGN in backends
        assert BackendKind.OLLAMA in backends
        assert BackendKind.LLAMA_CPP in backends

    def test_infer_sovereign(self):
        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.SOVEREIGN, d_model=32))
        result = registry.infer("Test inference")
        assert isinstance(result, InferenceResult)
        assert result.text != ""
        assert result.latency_ms > 0.0

    def test_infer_with_overrides(self):
        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.SOVEREIGN, d_model=32))
        result = registry.infer("Test", temperature=0.0, max_tokens=64)
        assert isinstance(result, InferenceResult)

    def test_embed_sovereign(self):
        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.SOVEREIGN, d_model=64))
        result = registry.embed("spectral embedding test")
        assert isinstance(result, EmbeddingResult)
        assert result.dimensions == 64
        assert len(result.vector) == 64

    def test_reason_with_record(self):
        """Test structured reasoning with a mock spectral record."""

        class MockComponent:
            name = "vertical"
            frequency = np.array([0.5, 1.0, 5.0, 10.0, 25.0])
            amplitude = np.array([0.2, 0.4, 0.8, 0.6, 0.3])
            domain = "frequency"

        class MockRecord:
            record_id = "REASON-001"
            components = [MockComponent()]
            metadata = {}

        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.SOVEREIGN, d_model=32))
        result = registry.reason("Is this anomalous?", record=MockRecord())
        assert isinstance(result, InferenceResult)
        assert result.spectral_context is not None
        assert result.spectral_context.record_id == "REASON-001"

    def test_health_check_sovereign(self):
        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.SOVEREIGN))
        assert registry.health_check() is True

    def test_stream_sovereign(self):
        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.SOVEREIGN, d_model=32))
        chunks = list(registry.stream("Stream test"))
        assert len(chunks) >= 1
        assert chunks[-1].done is True

    def test_register_custom_backend(self):
        """Test custom backend registration — extensibility."""

        class CustomBackend(LocalModelBackend):
            kind = BackendKind.VLLM_LOCAL

            def infer(self, prompt, config, context=None):
                return InferenceResult(text="custom", confidence=1.0)

            def embed(self, text, config):
                return EmbeddingResult(vector=np.zeros(8), dimensions=8)

            def health_check(self, config):
                return True

        registry = LocalModelRegistry()
        registry.register(BackendKind.VLLM_LOCAL, CustomBackend)
        result = registry.infer("test", backend=BackendKind.VLLM_LOCAL)
        assert result.text == "custom"
        assert result.confidence == 1.0


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_high_confidence_threshold(self):
        result = InferenceResult(confidence=0.8)
        assert result.is_high_confidence is True

    def test_low_confidence_threshold(self):
        result = InferenceResult(confidence=0.5)
        assert result.is_high_confidence is False

    def test_boundary_confidence(self):
        """0.7 is the MESIE standard threshold."""
        result = InferenceResult(confidence=0.7)
        assert result.is_high_confidence is True


class TestBackendKindEnum:
    """Tests for BackendKind enumeration."""

    def test_all_kinds(self):
        assert BackendKind.SOVEREIGN.value == "sovereign"
        assert BackendKind.OLLAMA.value == "ollama"
        assert BackendKind.LLAMA_CPP.value == "llama_cpp"
        assert BackendKind.HUGGINGFACE_LOCAL.value == "hf_local"
        assert BackendKind.VLLM_LOCAL.value == "vllm_local"


class TestRegisterBackend:
    """Tests for the global register_backend function."""

    def test_register_and_use(self):
        class TestBackend(LocalModelBackend):
            kind = BackendKind.HUGGINGFACE_LOCAL

            def infer(self, prompt, config, context=None):
                return InferenceResult(text="hf_test", confidence=0.9)

            def embed(self, text, config):
                return EmbeddingResult(vector=np.ones(4), dimensions=4)

            def health_check(self, config):
                return True

        register_backend(BackendKind.HUGGINGFACE_LOCAL, TestBackend)
        registry = LocalModelRegistry(BackendConfig(kind=BackendKind.HUGGINGFACE_LOCAL))
        result = registry.infer("test")
        assert result.text == "hf_test"
