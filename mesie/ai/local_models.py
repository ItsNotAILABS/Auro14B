"""Local Models — On-device inference backends for MESIE spectral intelligence.

Implements the MESIE engine pattern for local AI inference, following the same
ABC + registry architecture as ``mesie.engines``. All backends run entirely
on-device with zero cloud dependency, consistent with the laptop virtual chip
philosophy.

Integration points:
    - ``mesie.core.records.MultiElementRecord`` for spectral I/O
    - ``mesie.embeddings.vectorizers.SpectralVectorizer`` for embedding
    - ``mesie.ai.intelligence_protocols.IntelligenceProtocol`` for reasoning
    - ``mesie.cognitive.memory_adapter.SpectralMemoryAdapter`` for memory
    - ``phantom_native.neurocore.SovereignNeuroCore`` for native inference

Architecture:
    LocalModelBackend (ABC)
        ├── OllamaBackend        — REST-based local LLM (localhost:11434)
        ├── LlamaCppBackend      — Direct llama.cpp binding
        ├── SovereignBackend     — MESIE-native NeuroCore (zero deps)
        └── (custom via register_backend)

    LocalModelRegistry
        └── dispatch by BackendKind enum → backend instance

    SpectralInferenceContext
        └── Wraps MultiElementRecord + vectorization for LLM injection

Example:
    >>> from mesie.ai.local_models import LocalModelRegistry, BackendKind
    >>> registry = LocalModelRegistry()
    >>> registry.register(BackendKind.SOVEREIGN)
    >>> result = registry.infer("Classify this spectral anomaly", backend=BackendKind.SOVEREIGN)
    >>> print(result.text, result.confidence)
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np


# =============================================================================
# Enumerations — aligned with IntelligenceLevel / ReasoningStrategy pattern
# =============================================================================


class BackendKind(Enum):
    """Available local inference backends.

    Each backend runs fully on-device. No cloud keys required.
    """

    SOVEREIGN = "sovereign"          # MESIE-native NeuroCore
    OLLAMA = "ollama"                # Ollama REST API (localhost)
    LLAMA_CPP = "llama_cpp"          # llama.cpp Python bindings
    HUGGINGFACE_LOCAL = "hf_local"   # HuggingFace local pipeline
    VLLM_LOCAL = "vllm_local"        # vLLM local server


class InferenceMode(Enum):
    """Mode of local inference — mirrors IntelligenceLevel engagement."""

    GENERATE = "generate"        # Free-form text generation
    REASON = "reason"            # Structured spectral reasoning
    EMBED = "embed"              # Vector embedding only
    CLASSIFY = "classify"        # Spectral classification
    STREAM = "stream"            # Streaming token generation


class StopCondition(Enum):
    """Why generation terminated."""

    END_TOKEN = "end_token"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    ERROR = "error"


# =============================================================================
# Configuration — dataclass pattern matching ModelConfig, IntelligenceConfig
# =============================================================================


@dataclass
class BackendConfig:
    """Configuration for a local inference backend.

    Follows the same dataclass + defaults pattern as
    ``mesie.ai.models.ModelConfig`` and ``IntelligenceConfig``.

    Args:
        kind: Which backend to use.
        model_name: Model identifier (path or name depending on backend).
        host: Host for server-based backends.
        port: Port for server-based backends.
        context_length: Maximum context window (tokens).
        d_model: Internal dimension (sovereign backend).
        n_heads: Attention heads (sovereign backend).
        temperature: Sampling temperature (0 = deterministic).
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum generation length.
        seed: Random seed for reproducibility (critical for MESIE determinism).
        timeout_s: Request timeout in seconds.
    """

    kind: BackendKind = BackendKind.SOVEREIGN
    model_name: str = "mesie-neurocore-v1"
    host: str = "127.0.0.1"
    port: int = 11434
    context_length: int = 4096
    d_model: int = 128
    n_heads: int = 8
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    max_tokens: int = 512
    seed: Optional[int] = None
    timeout_s: float = 30.0


# =============================================================================
# Spectral Context — native to MESIE, not a bolt-on
# =============================================================================


@dataclass
class SpectralInferenceContext:
    """Spectral context for local model inference.

    Wraps a MultiElementRecord's key properties into a format
    the local model can consume. Mirrors how IntelligenceEngine
    uses SpectralVectorizer + load_record to prepare reasoning input.

    Args:
        record_id: Source record identifier.
        frequency_range: Tuple of (min_hz, max_hz).
        peak_frequencies: Detected spectral peaks (Hz).
        embedding: Pre-computed spectral embedding vector.
        component_names: Names of spectral components in the record.
        domain: Signal domain ('frequency' or 'time').
        metadata: Additional record metadata.
    """

    record_id: str = ""
    frequency_range: tuple[float, float] = (0.0, 0.0)
    peak_frequencies: List[float] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    component_names: List[str] = field(default_factory=list)
    domain: str = "frequency"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_block(self) -> str:
        """Format as a context block for LLM prompt injection.

        Returns a structured text block that can be prepended to any
        prompt, giving the model spectral awareness.
        """
        lines = [
            "[SPECTRAL CONTEXT]",
            f"  record_id: {self.record_id}",
            f"  domain: {self.domain}",
            f"  frequency_range: {self.frequency_range[0]:.2f} – {self.frequency_range[1]:.2f} Hz",
            f"  peaks: {', '.join(f'{p:.2f}' for p in self.peak_frequencies[:10])}",
            f"  components: {', '.join(self.component_names[:8])}",
        ]
        if self.embedding is not None:
            lines.append(f"  embedding_dim: {len(self.embedding)}")
            lines.append(f"  embedding_norm: {float(np.linalg.norm(self.embedding)):.4f}")
        lines.append("[/SPECTRAL CONTEXT]")
        return "\n".join(lines)

    @classmethod
    def from_record(cls, record: Any) -> "SpectralInferenceContext":
        """Build context from a MultiElementRecord.

        Args:
            record: A ``mesie.core.records.MultiElementRecord`` instance.

        Returns:
            SpectralInferenceContext populated from the record.
        """
        components = getattr(record, "components", [])
        freqs: List[float] = []
        names: List[str] = []
        peaks: List[float] = []

        for comp in components:
            names.append(getattr(comp, "name", ""))
            freq = getattr(comp, "frequency", None)
            if freq is not None and hasattr(freq, "__len__") and len(freq) > 0:
                freqs.extend([float(freq[0]), float(freq[-1])])
                # Simple peak detection: top 3 amplitude indices
                amp = getattr(comp, "amplitude", None)
                if amp is not None and hasattr(amp, "__len__") and len(amp) > 0:
                    top_idx = np.argsort(amp)[-3:]
                    peaks.extend(float(freq[i]) for i in top_idx if i < len(freq))

        freq_range = (min(freqs) if freqs else 0.0, max(freqs) if freqs else 0.0)
        domain = getattr(components[0], "domain", "frequency") if components else "frequency"

        return cls(
            record_id=getattr(record, "record_id", ""),
            frequency_range=freq_range,
            peak_frequencies=sorted(set(peaks)),
            component_names=names,
            domain=domain,
            metadata=getattr(record, "metadata", {}),
        )


# =============================================================================
# Response types — mirrors PredictionResult / ReasoningResult patterns
# =============================================================================


@dataclass
class InferenceResult:
    """Result from a local model inference call.

    Follows the same pattern as ``PredictionResult`` and ``ReasoningResult``
    with structured outputs and confidence scoring.

    Args:
        text: Generated text output.
        confidence: Model confidence (0–1), derived from logprobs or heuristic.
        stop_condition: Why generation stopped.
        tokens_used: Number of tokens consumed.
        latency_ms: Wall-clock inference time in milliseconds.
        model_name: Which model produced this result.
        spectral_context: Optional spectral context that was used.
        metadata: Additional backend-specific metadata.
    """

    text: str = ""
    confidence: float = 0.0
    stop_condition: StopCondition = StopCondition.END_TOKEN
    tokens_used: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    spectral_context: Optional[SpectralInferenceContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Check if result exceeds typical MESIE confidence threshold (0.7)."""
        return self.confidence >= 0.7


@dataclass
class EmbeddingResult:
    """Result from a local embedding call.

    Args:
        vector: The embedding vector.
        dimensions: Vector dimensionality.
        model_name: Which model produced this embedding.
        latency_ms: Wall-clock time in milliseconds.
    """

    vector: np.ndarray = field(default_factory=lambda: np.array([]))
    dimensions: int = 0
    model_name: str = ""
    latency_ms: float = 0.0


@dataclass
class StreamChunk:
    """A single chunk from streaming inference.

    Args:
        text: Token(s) in this chunk.
        done: Whether this is the final chunk.
        tokens_so_far: Running token count.
    """

    text: str = ""
    done: bool = False
    tokens_so_far: int = 0


# =============================================================================
# Backend ABC — mirrors Engine ABC from mesie.engines.base
# =============================================================================


class LocalModelBackend(ABC):
    """Abstract base class for local inference backends.

    Follows the same ABC pattern as ``mesie.engines.base.Engine``:
    each backend declares a ``kind`` and implements the inference contract.

    Subclasses must implement:
        - ``infer``: Text generation / reasoning
        - ``embed``: Vector embedding
        - ``health_check``: Verify backend availability
    """

    kind: BackendKind

    @abstractmethod
    def infer(
        self,
        prompt: str,
        config: BackendConfig,
        context: Optional[SpectralInferenceContext] = None,
    ) -> InferenceResult:
        """Run inference on the given prompt.

        Args:
            prompt: Input text prompt.
            config: Backend configuration.
            context: Optional spectral context to inject.

        Returns:
            InferenceResult with generated text and metadata.
        """

    @abstractmethod
    def embed(self, text: str, config: BackendConfig) -> EmbeddingResult:
        """Compute embedding vector for input text.

        Args:
            text: Input text to embed.
            config: Backend configuration.

        Returns:
            EmbeddingResult with the embedding vector.
        """

    @abstractmethod
    def health_check(self, config: BackendConfig) -> bool:
        """Check if the backend is available and ready.

        Args:
            config: Backend configuration.

        Returns:
            True if backend is operational.
        """

    def stream(
        self,
        prompt: str,
        config: BackendConfig,
        context: Optional[SpectralInferenceContext] = None,
    ) -> Iterator[StreamChunk]:
        """Stream inference token by token (optional override).

        Default implementation falls back to single-shot infer.

        Args:
            prompt: Input text prompt.
            config: Backend configuration.
            context: Optional spectral context.

        Yields:
            StreamChunk instances.
        """
        result = self.infer(prompt, config, context)
        yield StreamChunk(text=result.text, done=True, tokens_so_far=result.tokens_used)


# =============================================================================
# Sovereign Backend — MESIE-native (zero external deps)
# =============================================================================


class SovereignBackend(LocalModelBackend):
    """MESIE-native inference using SovereignNeuroCore.

    This backend uses phantom_native's resonance attention and helix-encoded
    weights for on-device spectral reasoning. Zero ML library dependencies.
    Follows the same philosophy as SOLUS math caretakers: fully local,
    deterministic, reproducible.

    The sovereign backend excels at:
        - Spectral pattern classification
        - Anomaly reasoning with spectral context
        - Deterministic embeddings for fingerprint/ANN pipelines
    """

    kind = BackendKind.SOVEREIGN

    def __init__(self) -> None:
        self._core: Optional[Any] = None

    def _ensure_core(self, config: BackendConfig) -> Any:
        """Lazy-init the NeuroCore with config parameters."""
        if self._core is None:
            try:
                from phantom_native.neurocore import SovereignNeuroCore

                self._core = SovereignNeuroCore({
                    "d_model": config.d_model,
                    "n_heads": config.n_heads,
                    "memory_cap": 32,
                })
            except ImportError:
                # Graceful degradation — core not available
                self._core = None
        return self._core

    def infer(
        self,
        prompt: str,
        config: BackendConfig,
        context: Optional[SpectralInferenceContext] = None,
    ) -> InferenceResult:
        """Run sovereign inference using resonance attention.

        Converts the prompt + spectral context into a tensor representation,
        processes through NeuroCore, and produces structured reasoning output.
        """
        start = time.perf_counter()
        core = self._ensure_core(config)

        # Build input representation
        full_prompt = prompt
        if context is not None:
            full_prompt = context.to_context_block() + "\n\n" + prompt

        # Tokenize to numeric signal (character-level for sovereign)
        signal = [ord(c) / 128.0 for c in full_prompt[:config.context_length]]

        # Pad or truncate to d_model
        if len(signal) < config.d_model:
            signal.extend([0.0] * (config.d_model - len(signal)))
        else:
            signal = signal[: config.d_model]

        # Process through NeuroCore if available
        if core is not None:
            try:
                from phantom_native.sovereign_tensor import SovereignTensor

                tensor = SovereignTensor(data=signal, shape=(config.d_model,))
                output = core.forward(tensor)
                output_data = output.data if hasattr(output, "data") else signal

                # Derive confidence from output activation strength
                activation_strength = sum(abs(v) for v in output_data) / len(output_data)
                confidence = min(1.0, activation_strength / 2.0)

                # Generate structured response from sovereign reasoning
                text = self._decode_sovereign_output(output_data, prompt)
            except Exception:
                # Fallback: deterministic heuristic response
                text, confidence = self._heuristic_response(prompt, context)
        else:
            text, confidence = self._heuristic_response(prompt, context)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return InferenceResult(
            text=text,
            confidence=confidence,
            stop_condition=StopCondition.END_TOKEN,
            tokens_used=len(signal),
            latency_ms=elapsed_ms,
            model_name=config.model_name,
            spectral_context=context,
        )

    def embed(self, text: str, config: BackendConfig) -> EmbeddingResult:
        """Produce a deterministic embedding via sovereign helix encoding.

        Uses the same helix initialization as NeuroCore weights to project
        text into a fixed-size spectral-compatible vector.
        """
        start = time.perf_counter()
        d = config.d_model

        # Character-level signal
        signal = [ord(c) / 128.0 for c in text[:d * 4]]

        # Helix projection (deterministic, reproducible)
        embedding = np.zeros(d, dtype=np.float64)
        for i, val in enumerate(signal):
            idx = i % d
            phase = math.sin(i * 0.1) * math.cos(idx * 0.1)
            embedding[idx] += val * phase

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return EmbeddingResult(
            vector=embedding,
            dimensions=d,
            model_name=config.model_name,
            latency_ms=elapsed_ms,
        )

    def health_check(self, config: BackendConfig) -> bool:
        """Sovereign backend is always available (zero deps)."""
        return True

    def _decode_sovereign_output(self, data: List[float], prompt: str) -> str:
        """Decode NeuroCore output into a text response.

        Uses activation patterns to select response templates
        for spectral analysis tasks.
        """
        # Activation-based classification
        mean_activation = sum(data) / len(data) if data else 0.0
        variance = sum((v - mean_activation) ** 2 for v in data) / len(data) if data else 0.0

        if variance > 0.5:
            category = "anomaly_detected"
        elif mean_activation > 0.3:
            category = "pattern_match"
        elif mean_activation < -0.1:
            category = "below_threshold"
        else:
            category = "stable"

        templates = {
            "anomaly_detected": f"Spectral anomaly detected. High variance ({variance:.4f}) "
                                f"indicates deviation from baseline patterns.",
            "pattern_match": f"Pattern recognized. Mean activation {mean_activation:.4f} "
                            f"correlates with known spectral signatures.",
            "below_threshold": f"Signal below detection threshold. Mean activation "
                              f"{mean_activation:.4f} suggests noise-dominated input.",
            "stable": f"Spectral pattern stable. Activation profile within normal "
                     f"operating range (mean={mean_activation:.4f}, var={variance:.4f}).",
        }
        return templates.get(category, templates["stable"])

    def _heuristic_response(
        self, prompt: str, context: Optional[SpectralInferenceContext]
    ) -> tuple[str, float]:
        """Fallback heuristic when NeuroCore unavailable."""
        if context and context.peak_frequencies:
            n_peaks = len(context.peak_frequencies)
            return (
                f"Spectral analysis: {n_peaks} peak frequencies detected in "
                f"{context.domain} domain. Record: {context.record_id}.",
                0.6,
            )
        return (
            "Sovereign inference: processing complete. "
            "No spectral context provided for detailed analysis.",
            0.5,
        )


# =============================================================================
# Ollama Backend — REST-based local LLM
# =============================================================================


class OllamaBackend(LocalModelBackend):
    """Ollama-based local inference via REST API.

    Connects to a locally-running Ollama server. Follows MESIE's
    zero-cloud-dependency principle: Ollama runs entirely on the
    user's machine.

    Requires:
        - Ollama installed and running (``ollama serve``)
        - A model pulled (e.g. ``ollama pull llama3``)
    """

    kind = BackendKind.OLLAMA

    def infer(
        self,
        prompt: str,
        config: BackendConfig,
        context: Optional[SpectralInferenceContext] = None,
    ) -> InferenceResult:
        """Generate text via Ollama's /api/generate endpoint."""
        import urllib.request
        import json

        start = time.perf_counter()

        full_prompt = prompt
        if context is not None:
            full_prompt = context.to_context_block() + "\n\n" + prompt

        payload = {
            "model": config.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            },
        }
        if config.seed is not None:
            payload["options"]["seed"] = config.seed

        url = f"http://{config.host}:{config.port}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return InferenceResult(
                text="",
                confidence=0.0,
                stop_condition=StopCondition.ERROR,
                latency_ms=elapsed_ms,
                model_name=config.model_name,
                metadata={"error": str(exc)},
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        text = body.get("response", "")
        tokens = body.get("eval_count", len(text.split()))

        # Derive confidence from eval duration (faster = more confident for the model)
        eval_ns = body.get("eval_duration", 1)
        confidence = min(1.0, 1.0 / (1.0 + eval_ns / 1e9))

        done_reason = body.get("done_reason", "stop")
        if done_reason == "length":
            stop = StopCondition.MAX_TOKENS
        else:
            stop = StopCondition.END_TOKEN

        return InferenceResult(
            text=text,
            confidence=confidence,
            stop_condition=stop,
            tokens_used=tokens,
            latency_ms=elapsed_ms,
            model_name=config.model_name,
            spectral_context=context,
            metadata={"raw_response": body},
        )

    def embed(self, text: str, config: BackendConfig) -> EmbeddingResult:
        """Compute embedding via Ollama's /api/embed endpoint."""
        import urllib.request
        import json

        start = time.perf_counter()
        payload = {"model": config.model_name, "input": text}
        url = f"http://{config.host}:{config.port}/api/embed"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return EmbeddingResult(
                vector=np.array([]),
                dimensions=0,
                model_name=config.model_name,
                latency_ms=elapsed_ms,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        embeddings = body.get("embeddings", [[]])
        vec = np.array(embeddings[0], dtype=np.float64) if embeddings else np.array([])

        return EmbeddingResult(
            vector=vec,
            dimensions=len(vec),
            model_name=config.model_name,
            latency_ms=elapsed_ms,
        )

    def health_check(self, config: BackendConfig) -> bool:
        """Ping Ollama server."""
        import urllib.request

        try:
            url = f"http://{config.host}:{config.port}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def stream(
        self,
        prompt: str,
        config: BackendConfig,
        context: Optional[SpectralInferenceContext] = None,
    ) -> Iterator[StreamChunk]:
        """Stream tokens from Ollama's /api/generate with stream=True."""
        import urllib.request
        import json

        full_prompt = prompt
        if context is not None:
            full_prompt = context.to_context_block() + "\n\n" + prompt

        payload = {
            "model": config.model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            },
        }
        if config.seed is not None:
            payload["options"]["seed"] = config.seed

        url = f"http://{config.host}:{config.port}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=config.timeout_s)
            tokens_seen = 0
            for line in resp:
                chunk = json.loads(line.decode("utf-8"))
                tokens_seen += 1
                yield StreamChunk(
                    text=chunk.get("response", ""),
                    done=chunk.get("done", False),
                    tokens_so_far=tokens_seen,
                )
                if chunk.get("done", False):
                    break
            resp.close()
        except Exception:
            yield StreamChunk(text="", done=True, tokens_so_far=0)


# =============================================================================
# LlamaCpp Backend — direct binding
# =============================================================================


class LlamaCppBackend(LocalModelBackend):
    """llama.cpp Python binding backend.

    Uses the ``llama-cpp-python`` package for direct model loading.
    Fully on-device, supports GGUF quantized models.

    Requires:
        - ``pip install llama-cpp-python``
        - A GGUF model file on disk
    """

    kind = BackendKind.LLAMA_CPP

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._model_path: str = ""

    def _ensure_model(self, config: BackendConfig) -> Any:
        """Lazy-load llama.cpp model."""
        if self._model is None or self._model_path != config.model_name:
            try:
                from llama_cpp import Llama  # type: ignore[import-untyped]

                self._model = Llama(
                    model_path=config.model_name,
                    n_ctx=config.context_length,
                    seed=config.seed or -1,
                    verbose=False,
                )
                self._model_path = config.model_name
            except ImportError:
                self._model = None
        return self._model

    def infer(
        self,
        prompt: str,
        config: BackendConfig,
        context: Optional[SpectralInferenceContext] = None,
    ) -> InferenceResult:
        """Generate via llama.cpp."""
        start = time.perf_counter()
        model = self._ensure_model(config)

        full_prompt = prompt
        if context is not None:
            full_prompt = context.to_context_block() + "\n\n" + prompt

        if model is None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return InferenceResult(
                text="",
                confidence=0.0,
                stop_condition=StopCondition.ERROR,
                latency_ms=elapsed_ms,
                model_name=config.model_name,
                metadata={"error": "llama-cpp-python not installed"},
            )

        output = model(
            full_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        choices = output.get("choices", [{}])
        text = choices[0].get("text", "") if choices else ""
        finish = choices[0].get("finish_reason", "stop") if choices else "stop"
        tokens = output.get("usage", {}).get("completion_tokens", len(text.split()))

        stop = StopCondition.MAX_TOKENS if finish == "length" else StopCondition.END_TOKEN

        return InferenceResult(
            text=text,
            confidence=0.7,  # llama.cpp doesn't expose logprobs easily
            stop_condition=stop,
            tokens_used=tokens,
            latency_ms=elapsed_ms,
            model_name=config.model_name,
            spectral_context=context,
        )

    def embed(self, text: str, config: BackendConfig) -> EmbeddingResult:
        """Compute embedding via llama.cpp."""
        start = time.perf_counter()
        model = self._ensure_model(config)

        if model is None:
            return EmbeddingResult(
                vector=np.array([]),
                dimensions=0,
                model_name=config.model_name,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        try:
            emb = model.embed(text)
            vec = np.array(emb, dtype=np.float64)
        except Exception:
            vec = np.array([])

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            vector=vec,
            dimensions=len(vec),
            model_name=config.model_name,
            latency_ms=elapsed_ms,
        )

    def health_check(self, config: BackendConfig) -> bool:
        """Check if llama.cpp model can be loaded."""
        return self._ensure_model(config) is not None


# =============================================================================
# Registry — mirrors EngineRegistry from mesie.engines.base
# =============================================================================


# Global backend registry (same pattern as engine registration)
_BACKEND_REGISTRY: Dict[BackendKind, type] = {
    BackendKind.SOVEREIGN: SovereignBackend,
    BackendKind.OLLAMA: OllamaBackend,
    BackendKind.LLAMA_CPP: LlamaCppBackend,
}


def register_backend(kind: BackendKind, backend_cls: type) -> None:
    """Register a custom backend class.

    Follows the same extensibility pattern as ``EngineRegistry.register()``.

    Args:
        kind: BackendKind enum value for this backend.
        backend_cls: Class implementing LocalModelBackend ABC.
    """
    _BACKEND_REGISTRY[kind] = backend_cls


class LocalModelRegistry:
    """Registry and dispatcher for local model backends.

    Mirrors ``mesie.engines.base.EngineRegistry`` — instantiates backends
    on demand and routes inference requests to the appropriate provider.

    Example:
        >>> registry = LocalModelRegistry()
        >>> result = registry.infer("Analyze spectrum", backend=BackendKind.SOVEREIGN)
        >>> print(result.text)
    """

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        self._config = config or BackendConfig()
        self._instances: Dict[BackendKind, LocalModelBackend] = {}

    @property
    def config(self) -> BackendConfig:
        """Current backend configuration."""
        return self._config

    @config.setter
    def config(self, value: BackendConfig) -> None:
        self._config = value

    def _get_backend(self, kind: Optional[BackendKind] = None) -> LocalModelBackend:
        """Get or create a backend instance."""
        kind = kind or self._config.kind
        if kind not in self._instances:
            cls = _BACKEND_REGISTRY.get(kind)
            if cls is None:
                raise ValueError(
                    f"No backend registered for {kind.value}. "
                    f"Available: {[k.value for k in _BACKEND_REGISTRY]}"
                )
            self._instances[kind] = cls()
        return self._instances[kind]

    def register(self, kind: BackendKind, backend_cls: Optional[type] = None) -> None:
        """Register a backend (class or from global registry).

        Args:
            kind: Backend kind to register/activate.
            backend_cls: Optional custom class. If None, uses global registry.
        """
        if backend_cls is not None:
            register_backend(kind, backend_cls)
        # Pre-instantiate
        self._get_backend(kind)

    def available_backends(self) -> List[BackendKind]:
        """List all registered backend kinds."""
        return list(_BACKEND_REGISTRY.keys())

    def health_check(self, backend: Optional[BackendKind] = None) -> bool:
        """Check if a backend is healthy.

        Args:
            backend: Specific backend to check. Defaults to configured backend.
        """
        b = self._get_backend(backend)
        return b.health_check(self._config)

    def infer(
        self,
        prompt: str,
        backend: Optional[BackendKind] = None,
        context: Optional[SpectralInferenceContext] = None,
        **overrides: Any,
    ) -> InferenceResult:
        """Run inference on the specified backend.

        Args:
            prompt: Input prompt text.
            backend: Which backend to use (defaults to config.kind).
            context: Optional spectral context for MESIE-aware inference.
            **overrides: Override config fields (temperature, max_tokens, etc).

        Returns:
            InferenceResult from the backend.
        """
        config = self._apply_overrides(overrides)
        b = self._get_backend(backend)
        return b.infer(prompt, config, context)

    def embed(
        self,
        text: str,
        backend: Optional[BackendKind] = None,
    ) -> EmbeddingResult:
        """Compute embedding via the specified backend.

        Args:
            text: Input text to embed.
            backend: Which backend to use.

        Returns:
            EmbeddingResult with the vector.
        """
        b = self._get_backend(backend)
        return b.embed(text, self._config)

    def stream(
        self,
        prompt: str,
        backend: Optional[BackendKind] = None,
        context: Optional[SpectralInferenceContext] = None,
    ) -> Iterator[StreamChunk]:
        """Stream inference from the specified backend.

        Args:
            prompt: Input prompt.
            backend: Which backend to use.
            context: Optional spectral context.

        Yields:
            StreamChunk instances.
        """
        b = self._get_backend(backend)
        return b.stream(prompt, self._config, context)

    def reason(
        self,
        prompt: str,
        record: Any = None,
        backend: Optional[BackendKind] = None,
    ) -> InferenceResult:
        """Structured spectral reasoning (convenience method).

        Builds SpectralInferenceContext from a MultiElementRecord and
        runs inference with spectral awareness.

        Args:
            prompt: Reasoning query.
            record: Optional MultiElementRecord for spectral context.
            backend: Which backend to use.

        Returns:
            InferenceResult with reasoning output.
        """
        context = None
        if record is not None:
            context = SpectralInferenceContext.from_record(record)
        return self.infer(prompt, backend=backend, context=context)

    def _apply_overrides(self, overrides: Dict[str, Any]) -> BackendConfig:
        """Create a config copy with overrides applied."""
        if not overrides:
            return self._config
        # Create a new config with overrides
        params = {
            "kind": self._config.kind,
            "model_name": self._config.model_name,
            "host": self._config.host,
            "port": self._config.port,
            "context_length": self._config.context_length,
            "d_model": self._config.d_model,
            "n_heads": self._config.n_heads,
            "temperature": self._config.temperature,
            "top_k": self._config.top_k,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "seed": self._config.seed,
            "timeout_s": self._config.timeout_s,
        }
        params.update(overrides)
        return BackendConfig(**params)
