"""MESIE Internal AI SDK — Unified interface for the spectral intelligence organism.

This module provides a production-grade facade for all foundation model capabilities,
serving as the single entrypoint for internal consumers of the spectral intelligence
engine. The SDK orchestrates tokenization, embedding, inference, training, and
cross-modal alignment through a coherent, stateful API.

Usage:
    from mesie.foundation.sdk import SpectralIntelligenceSDK

    sdk = SpectralIntelligenceSDK.from_config(config)
    sdk.initialize()

    # Tokenize spectral data
    tokens = sdk.tokenize(spectral_record)

    # Generate embeddings in the universal latent space
    embedding = sdk.embed(spectral_record, modality="seismic")

    # Run inference through the foundation model
    prediction = sdk.infer(tokens, task="reconstruction")

    # Train/finetune on domain-specific data
    sdk.train(dataset, objectives=["masked_spectral", "contrastive"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mesie.foundation.config.pretraining_config import (
    PretrainingConfig,
    ModelConfig,
    TokenizerConfig,
    DataConfig,
    TrainingConfig,
    LatentSpaceConfig,
    ObjectiveConfig,
    EvaluationConfig,
    ModalityType,
)
from mesie.foundation.models.spectral_gpt import SpectralGPT
from mesie.foundation.tokenizers.spectral_tokenizer import SpectralTokenizer
from mesie.foundation.latent.universal_latent_space import UniversalSpectralLatentSpace
from mesie.foundation.training.pretraining_engine import PretrainingEngine


class SDKState(str, Enum):
    """Lifecycle states of the SDK instance."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    TRAINING = "training"
    INFERENCE = "inference"
    ERROR = "error"


@dataclass
class InferenceResult:
    """Result container for foundation model inference.

    Attributes:
        output: Raw model output array.
        latent_embedding: Embedding in the universal latent space.
        confidence: Model confidence score in [0, 1].
        modality: Source modality type.
        metadata: Additional inference metadata.
    """

    output: np.ndarray
    latent_embedding: np.ndarray
    confidence: float = 0.0
    modality: str = "universal"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizationResult:
    """Result container for spectral tokenization.

    Attributes:
        token_ids: Discrete token indices (for VQ-based tokenizers).
        continuous_tokens: Continuous token representations.
        attention_mask: Mask indicating valid token positions.
        sequence_length: Number of tokens produced.
        metadata: Tokenization metadata (codebook usage, quantization error).
    """

    token_ids: Optional[np.ndarray] = None
    continuous_tokens: Optional[np.ndarray] = None
    attention_mask: Optional[np.ndarray] = None
    sequence_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result container for latent space embeddings.

    Attributes:
        embedding: Dense embedding vector in the universal latent space.
        modality_projection: Modality-specific projection of the embedding.
        alignment_score: Cross-modal alignment score.
        nearest_prototypes: Indices of nearest cluster prototypes.
        metadata: Additional embedding metadata.
    """

    embedding: np.ndarray
    modality_projection: Optional[np.ndarray] = None
    alignment_score: float = 0.0
    nearest_prototypes: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSession:
    """Tracks an active training session.

    Attributes:
        session_id: Unique training session identifier.
        step: Current training step.
        epoch: Current epoch.
        total_loss: Current total loss.
        objectives: Active training objectives.
        is_active: Whether the session is currently running.
    """

    session_id: str
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    objectives: List[str] = field(default_factory=list)
    is_active: bool = False


class SpectralIntelligenceSDK:
    """Unified internal SDK for the MESIE spectral intelligence organism.

    This class is the single entrypoint for all foundation model operations
    within the organism. It manages the lifecycle of the model, tokenizer,
    latent space, and training engine as a coherent system.

    The SDK follows an initialize-once, use-many pattern:
        1. Configure via from_config() or constructor
        2. Call initialize() to prepare all subsystems
        3. Use tokenize(), embed(), infer(), train() for operations
        4. Call shutdown() for clean resource release

    Attributes:
        state: Current lifecycle state of the SDK.
        config: Active pretraining configuration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
        latent_config: Optional[LatentSpaceConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        data_config: Optional[DataConfig] = None,
        objective_config: Optional[ObjectiveConfig] = None,
        evaluation_config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize SDK with individual component configurations.

        Args:
            model_config: Foundation model architecture configuration.
            tokenizer_config: Tokenizer pipeline configuration.
            latent_config: Universal latent space configuration.
            training_config: Training loop configuration.
            data_config: Data pipeline configuration.
            objective_config: Training objective configuration.
            evaluation_config: Evaluation and probing configuration.
        """
        self._model_config = model_config or ModelConfig()
        self._tokenizer_config = tokenizer_config or TokenizerConfig()
        self._latent_config = latent_config or LatentSpaceConfig()
        self._training_config = training_config or TrainingConfig()
        self._data_config = data_config or DataConfig()
        self._objective_config = objective_config or ObjectiveConfig()
        self._evaluation_config = evaluation_config or EvaluationConfig()

        self._model: Optional[SpectralGPT] = None
        self._tokenizer: Optional[SpectralTokenizer] = None
        self._latent_space: Optional[UniversalSpectralLatentSpace] = None
        self._engine: Optional[PretrainingEngine] = None
        self._training_session: Optional[TrainingSession] = None

        self.state = SDKState.UNINITIALIZED

    @classmethod
    def from_config(cls, config: PretrainingConfig) -> "SpectralIntelligenceSDK":
        """Create SDK instance from a unified pretraining configuration.

        Args:
            config: Complete pretraining configuration object.

        Returns:
            Configured SDK instance (still requires initialize() call).
        """
        return cls(
            model_config=config.model,
            tokenizer_config=config.tokenizer,
            latent_config=config.latent_space,
            training_config=config.training,
            data_config=config.data,
            objective_config=config.objectives,
            evaluation_config=config.evaluation,
        )

    @classmethod
    def default(cls) -> "SpectralIntelligenceSDK":
        """Create SDK with default production configuration.

        Returns:
            SDK instance with sensible defaults, ready for initialize().
        """
        return cls()

    def initialize(self) -> None:
        """Initialize all subsystems and transition to READY state.

        This prepares the model, tokenizer, latent space, and training engine
        for operation. Must be called before any inference or training methods.

        Raises:
            RuntimeError: If SDK is already initialized or in error state.
        """
        if self.state == SDKState.READY:
            return
        if self.state == SDKState.ERROR:
            raise RuntimeError(
                "SDK is in error state. Create a new instance to recover."
            )

        try:
            self._tokenizer = SpectralTokenizer(config=self._tokenizer_config)
            self._model = SpectralGPT(config=self._model_config)
            self._latent_space = UniversalSpectralLatentSpace(
                config=self._latent_config
            )
            self._engine = PretrainingEngine(
                model=self._model,
                tokenizer=self._tokenizer,
                latent_space=self._latent_space,
                config=self._training_config,
            )
            self.state = SDKState.READY
        except Exception as exc:
            self.state = SDKState.ERROR
            raise RuntimeError(f"SDK initialization failed: {exc}") from exc

    def shutdown(self) -> None:
        """Release resources and reset state.

        Safe to call multiple times. After shutdown, the SDK instance
        cannot be reused — create a new one.
        """
        self._model = None
        self._tokenizer = None
        self._latent_space = None
        self._engine = None
        self._training_session = None
        self.state = SDKState.UNINITIALIZED

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(
        self,
        spectral_data: np.ndarray,
        *,
        modality: str = "universal",
        patch_size: Optional[int] = None,
    ) -> TokenizationResult:
        """Tokenize raw spectral data into model-ready token sequences.

        Converts frequency-domain arrays into discrete or continuous tokens
        suitable for transformer processing through the foundation model.

        Args:
            spectral_data: Input spectral array (1D amplitude or 2D freq+amp).
            modality: Source modality hint for tokenizer routing.
            patch_size: Optional override for patch tokenization window.

        Returns:
            TokenizationResult with token IDs and/or continuous representations.

        Raises:
            RuntimeError: If SDK is not in READY state.
            ValueError: If spectral_data has invalid shape.
        """
        self._require_state(SDKState.READY)
        assert self._tokenizer is not None

        data = np.asarray(spectral_data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError(
                f"spectral_data must be 1D or 2D; got shape {data.shape}"
            )

        tokens = self._tokenizer.tokenize(
            data, modality=modality, patch_size=patch_size
        )

        return TokenizationResult(
            token_ids=tokens.get("token_ids"),
            continuous_tokens=tokens.get("continuous_tokens"),
            attention_mask=tokens.get("attention_mask"),
            sequence_length=tokens.get("sequence_length", data.shape[-1]),
            metadata={"modality": modality, "input_shape": data.shape},
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        spectral_data: np.ndarray,
        *,
        modality: str = "universal",
        normalize: bool = True,
    ) -> EmbeddingResult:
        """Generate universal latent space embeddings from spectral data.

        Maps input spectra into the shared representation space where all
        modalities are aligned, enabling cross-modal similarity and retrieval.

        Args:
            spectral_data: Input spectral array.
            modality: Source modality for modality-specific projection.
            normalize: Whether to L2-normalize the output embedding.

        Returns:
            EmbeddingResult with dense embedding and alignment metadata.

        Raises:
            RuntimeError: If SDK is not in READY state.
        """
        self._require_state(SDKState.READY)
        assert self._model is not None
        assert self._latent_space is not None

        tokens = self.tokenize(spectral_data, modality=modality)
        input_tokens = (
            tokens.continuous_tokens
            if tokens.continuous_tokens is not None
            else tokens.token_ids
        )

        if input_tokens is None:
            raise ValueError("Tokenization produced no valid tokens.")

        hidden = self._model.encode(input_tokens)
        embedding = self._latent_space.project(hidden, modality=modality)

        if normalize and embedding is not None:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return EmbeddingResult(
            embedding=embedding,
            modality_projection=self._latent_space.modality_project(
                embedding, modality=modality
            ),
            alignment_score=self._latent_space.alignment_score(
                embedding, modality=modality
            ),
            metadata={"modality": modality, "normalized": normalize},
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        spectral_data: np.ndarray,
        *,
        task: str = "reconstruction",
        modality: str = "universal",
        return_latent: bool = True,
    ) -> InferenceResult:
        """Run inference through the foundation model.

        Supports multiple tasks: reconstruction, next-window prediction,
        classification, and anomaly detection.

        Args:
            spectral_data: Input spectral array.
            task: Inference task type ('reconstruction', 'next_window',
                  'classification', 'anomaly').
            modality: Source modality.
            return_latent: Whether to include latent embedding in result.

        Returns:
            InferenceResult with model output and optional latent embedding.

        Raises:
            RuntimeError: If SDK is not in READY or INFERENCE state.
            ValueError: If task is not supported.
        """
        self._require_state(SDKState.READY, SDKState.INFERENCE)
        assert self._model is not None

        supported_tasks = {
            "reconstruction",
            "next_window",
            "classification",
            "anomaly",
        }
        if task not in supported_tasks:
            raise ValueError(
                f"Unsupported task '{task}'. Must be one of: {sorted(supported_tasks)}"
            )

        tokens = self.tokenize(spectral_data, modality=modality)
        input_tokens = (
            tokens.continuous_tokens
            if tokens.continuous_tokens is not None
            else tokens.token_ids
        )

        if input_tokens is None:
            raise ValueError("Tokenization produced no valid tokens.")

        output = self._model.forward(input_tokens, task=task)

        latent_embedding = np.array([])
        if return_latent and self._latent_space is not None:
            hidden = self._model.encode(input_tokens)
            latent_embedding = self._latent_space.project(
                hidden, modality=modality
            )

        return InferenceResult(
            output=output.get("prediction", np.array([])),
            latent_embedding=latent_embedding,
            confidence=float(output.get("confidence", 0.0)),
            modality=modality,
            metadata={"task": task, "sequence_length": tokens.sequence_length},
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: Any,
        *,
        objectives: Optional[List[str]] = None,
        num_steps: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingSession:
        """Start or resume a training session.

        Orchestrates pretraining or finetuning with configurable objectives,
        gradient accumulation, and distributed coordination.

        Args:
            dataset: Training data (iterable of spectral arrays or records).
            objectives: List of training objective names. If None, uses config.
            num_steps: Number of training steps. If None, uses config default.
            callbacks: Optional list of callback functions called per step.

        Returns:
            TrainingSession tracking the training run.

        Raises:
            RuntimeError: If SDK is not in READY state.
        """
        self._require_state(SDKState.READY)
        assert self._engine is not None

        self.state = SDKState.TRAINING
        session_id = f"train_{id(dataset)}_{id(self)}"

        self._training_session = TrainingSession(
            session_id=session_id,
            objectives=objectives or self._objective_config.active_objectives,
            is_active=True,
        )

        try:
            self._engine.train(
                dataset=dataset,
                objectives=self._training_session.objectives,
                num_steps=num_steps,
                callbacks=callbacks,
            )
            self._training_session.is_active = False
            self.state = SDKState.READY
        except Exception as exc:
            self._training_session.is_active = False
            self.state = SDKState.ERROR
            raise RuntimeError(f"Training failed: {exc}") from exc

        return self._training_session

    # ------------------------------------------------------------------
    # Cross-modal operations
    # ------------------------------------------------------------------

    def align(
        self,
        source_data: np.ndarray,
        target_data: np.ndarray,
        *,
        source_modality: str = "universal",
        target_modality: str = "universal",
    ) -> float:
        """Compute cross-modal alignment score between two spectral inputs.

        Maps both inputs to the universal latent space and measures their
        alignment quality, useful for transfer learning readiness assessment.

        Args:
            source_data: Source spectral array.
            target_data: Target spectral array.
            source_modality: Modality of source data.
            target_modality: Modality of target data.

        Returns:
            Alignment score in [0, 1] where 1.0 indicates perfect alignment.
        """
        self._require_state(SDKState.READY)

        source_emb = self.embed(source_data, modality=source_modality)
        target_emb = self.embed(target_data, modality=target_modality)

        dot = float(np.dot(source_emb.embedding, target_emb.embedding))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))

    def transfer(
        self,
        spectral_data: np.ndarray,
        *,
        source_modality: str,
        target_modality: str,
    ) -> np.ndarray:
        """Transfer spectral representation across modalities.

        Uses the universal latent space to project data from one modality
        into the representation space of another, enabling zero-shot
        cross-domain inference.

        Args:
            spectral_data: Input spectral array in source modality.
            source_modality: Original modality of the data.
            target_modality: Target modality for projection.

        Returns:
            Projected spectral representation in target modality space.
        """
        self._require_state(SDKState.READY)
        assert self._latent_space is not None

        source_emb = self.embed(spectral_data, modality=source_modality)
        projected = self._latent_space.cross_modal_transfer(
            source_emb.embedding,
            source_modality=source_modality,
            target_modality=target_modality,
        )
        return projected

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def model_parameters(self) -> int:
        """Total number of model parameters."""
        if self._model is None:
            return 0
        return self._model.num_parameters

    @property
    def is_ready(self) -> bool:
        """Whether the SDK is initialized and ready for operations."""
        return self.state == SDKState.READY

    @property
    def is_training(self) -> bool:
        """Whether a training session is currently active."""
        return self.state == SDKState.TRAINING

    @property
    def active_session(self) -> Optional[TrainingSession]:
        """Return the active training session, if any."""
        return self._training_session

    def diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about the SDK state.

        Returns:
            Dictionary with model stats, config summary, and health status.
        """
        return {
            "state": self.state.value,
            "model_parameters": self.model_parameters,
            "model_config": {
                "hidden_dim": self._model_config.hidden_dim,
                "num_layers": self._model_config.num_layers,
                "num_heads": self._model_config.num_heads,
            },
            "tokenizer_type": self._tokenizer_config.tokenizer_type,
            "latent_dim": self._latent_config.latent_dim,
            "supported_modalities": [m.value for m in ModalityType],
            "training_session": (
                {
                    "session_id": self._training_session.session_id,
                    "step": self._training_session.step,
                    "is_active": self._training_session.is_active,
                }
                if self._training_session
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_state(self, *allowed: SDKState) -> None:
        """Assert SDK is in one of the allowed states."""
        if self.state not in allowed:
            raise RuntimeError(
                f"Operation requires SDK state {[s.value for s in allowed]}, "
                f"but current state is '{self.state.value}'. "
                f"Call initialize() first."
            )
