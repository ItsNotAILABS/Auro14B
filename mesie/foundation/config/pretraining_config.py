"""Pretraining configuration for MESIE Foundation Model.

This module defines all configuration dataclasses for the spectral foundation
model pretraining pipeline. The configuration system is hierarchical and supports
JSON/YAML serialization for experiment tracking and reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Enumerations for configuration choices
# ---------------------------------------------------------------------------

class ModalityType(str, Enum):
    """Supported spectral data modalities."""
    SEISMIC = "seismic"
    VIBRATION = "vibration"
    EEG = "eeg"
    ECG = "ecg"
    AUDIO = "audio"
    RF = "rf"
    SYNTHETIC = "synthetic"
    UNIVERSAL = "universal"


class TokenizerType(str, Enum):
    """Tokenizer architecture types."""
    VQVAE = "vqvae"
    PATCH = "patch"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"
    LEARNED_CODEBOOK = "learned_codebook"


class PositionalEncodingType(str, Enum):
    """Positional encoding strategies."""
    SINUSOIDAL = "sinusoidal"
    ROTARY = "rotary"
    LEARNED = "learned"
    ALIBI = "alibi"
    FREQUENCY_AWARE = "frequency_aware"
    SPECTRAL_HARMONIC = "spectral_harmonic"


class AttentionType(str, Enum):
    """Attention mechanism types."""
    STANDARD = "standard"
    MULTI_SCALE = "multi_scale"
    FREQUENCY_BAND = "frequency_band"
    SPARSE = "sparse"
    LINEAR = "linear"
    FLASH = "flash"
    SPECTRAL_CROSS = "spectral_cross"


class ObjectiveType(str, Enum):
    """Pretraining objective types."""
    MASKED_SPECTRAL = "masked_spectral"
    NEXT_WINDOW = "next_window"
    CONTRASTIVE = "contrastive"
    RECONSTRUCTION = "reconstruction"
    DENOISING = "denoising"
    FREQUENCY_PREDICTION = "frequency_prediction"
    CROSS_MODAL = "cross_modal"
    PHYSICS_INFORMED = "physics_informed"
    MULTI_RESOLUTION = "multi_resolution"
    CAUSAL_SPECTRAL = "causal_spectral"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    LINEAR_WARMUP = "linear_warmup"
    COSINE_WARMUP = "cosine_warmup"
    POLYNOMIAL = "polynomial"
    ONE_CYCLE = "one_cycle"
    INVERSE_SQRT = "inverse_sqrt"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"
    SOPHIA = "sophia"
    ADAFACTOR = "adafactor"
    LAMB = "lamb"


class NormalizationType(str, Enum):
    """Normalization layer types."""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    GROUP_NORM = "group_norm"
    SPECTRAL_NORM = "spectral_norm"


class ActivationType(str, Enum):
    """Activation function types."""
    GELU = "gelu"
    SWIGLU = "swiglu"
    RELU = "relu"
    SILU = "silu"
    GEGLU = "geglu"
    MISH = "mish"


class LatentDistribution(str, Enum):
    """Latent space distribution types."""
    GAUSSIAN = "gaussian"
    VON_MISES_FISHER = "von_mises_fisher"
    HYPERBOLIC = "hyperbolic"
    MIXTURE = "mixture"
    FLOW_BASED = "flow_based"


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms.

    Attributes:
        attention_type: Type of attention mechanism.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        dropout: Attention dropout rate.
        use_rotary: Whether to use rotary positional embeddings.
        rotary_dim: Dimension for rotary embeddings.
        max_position_embeddings: Maximum sequence length for position embeddings.
        use_flash_attention: Whether to use flash attention optimization.
        sliding_window: Sliding window size for local attention (0 = global).
        num_kv_heads: Number of key-value heads for GQA (0 = standard MHA).
        frequency_bands: Number of frequency bands for band attention.
        band_overlap: Overlap ratio between frequency bands.
        causal: Whether attention is causal (autoregressive).
        cross_attention: Whether this is cross-attention.
        qk_norm: Whether to normalize queries and keys.
    """
    attention_type: str = AttentionType.STANDARD
    num_heads: int = 16
    head_dim: int = 64
    dropout: float = 0.0
    use_rotary: bool = True
    rotary_dim: int = 64
    max_position_embeddings: int = 8192
    use_flash_attention: bool = False
    sliding_window: int = 0
    num_kv_heads: int = 0
    frequency_bands: int = 8
    band_overlap: float = 0.25
    causal: bool = False
    cross_attention: bool = False
    qk_norm: bool = True


@dataclass
class FFNConfig:
    """Configuration for feed-forward networks.

    Attributes:
        hidden_dim: Hidden dimension in FFN.
        activation: Activation function type.
        dropout: FFN dropout rate.
        use_gated: Whether to use gated linear units.
        bias: Whether to include bias terms.
        multiple_of: Round hidden dim to this multiple for efficiency.
    """
    hidden_dim: int = 4096
    activation: str = ActivationType.SWIGLU
    dropout: float = 0.0
    use_gated: bool = True
    bias: bool = False
    multiple_of: int = 256


@dataclass
class TransformerBlockConfig:
    """Configuration for a transformer block.

    Attributes:
        attention: Attention configuration.
        ffn: Feed-forward configuration.
        normalization: Normalization type.
        pre_norm: Whether to use pre-normalization.
        residual_dropout: Dropout on residual connections.
        use_parallel_attention: Whether to compute attention and FFN in parallel.
    """
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    normalization: str = NormalizationType.RMS_NORM
    pre_norm: bool = True
    residual_dropout: float = 0.0
    use_parallel_attention: bool = False


@dataclass
class SpectralEncoderConfig:
    """Configuration for the spectral-specific encoder layers.

    Attributes:
        num_frequency_filters: Number of learned frequency-domain filters.
        filter_order: Order of the spectral filters.
        use_dft_layer: Whether to include learnable DFT layers.
        dft_dim: Dimension for learnable DFT.
        use_wavelet_decomposition: Whether to use wavelet multi-resolution.
        wavelet_levels: Number of wavelet decomposition levels.
        harmonic_attention: Whether to use harmonic-aware attention.
        max_harmonics: Maximum number of harmonics to track.
        phase_encoding: Whether to encode phase information.
        magnitude_log_scale: Whether to use log-scale for magnitudes.
        octave_pooling: Whether to use octave-band pooling.
        num_octave_bands: Number of octave bands for pooling.
    """
    num_frequency_filters: int = 128
    filter_order: int = 64
    use_dft_layer: bool = True
    dft_dim: int = 512
    use_wavelet_decomposition: bool = True
    wavelet_levels: int = 6
    harmonic_attention: bool = True
    max_harmonics: int = 16
    phase_encoding: bool = True
    magnitude_log_scale: bool = True
    octave_pooling: bool = True
    num_octave_bands: int = 10


@dataclass
class ModelConfig:
    """Complete model architecture configuration.

    Attributes:
        model_name: Name identifier for the model.
        hidden_dim: Model hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        vocab_size: Vocabulary size for discrete tokens.
        max_seq_len: Maximum sequence length.
        positional_encoding: Positional encoding type.
        transformer_block: Transformer block configuration.
        spectral_encoder: Spectral encoder configuration.
        embedding_dim: Embedding dimension (may differ from hidden_dim).
        tie_embeddings: Whether to tie input/output embeddings.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        init_std: Standard deviation for weight initialization.
        num_modality_experts: Number of modality-specific expert layers.
        use_mixture_of_experts: Whether to use MoE architecture.
        num_experts: Number of experts in MoE layers.
        top_k_experts: Number of active experts per token.
        expert_capacity_factor: Capacity factor for expert routing.
        use_cross_modal_attention: Whether to include cross-modal attention layers.
        cross_modal_layers: Which layers have cross-modal attention.
        output_heads: Dictionary of output head configurations.
    """
    model_name: str = "spectral-gpt-base"
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    head_dim: int = 64
    vocab_size: int = 32768
    max_seq_len: int = 8192
    positional_encoding: str = PositionalEncodingType.ROTARY
    transformer_block: TransformerBlockConfig = field(default_factory=TransformerBlockConfig)
    spectral_encoder: SpectralEncoderConfig = field(default_factory=SpectralEncoderConfig)
    embedding_dim: int = 1024
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = True
    init_std: float = 0.02
    num_modality_experts: int = 7
    use_mixture_of_experts: bool = True
    num_experts: int = 8
    top_k_experts: int = 2
    expert_capacity_factor: float = 1.25
    use_cross_modal_attention: bool = True
    cross_modal_layers: List[int] = field(default_factory=lambda: [6, 12, 18, 23])
    output_heads: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "spectral_reconstruction": {"dim": 1024, "activation": "linear"},
        "next_window": {"dim": 1024, "activation": "linear"},
        "contrastive": {"dim": 256, "activation": "linear"},
        "classification": {"dim": 512, "activation": "gelu"},
    })


# ---------------------------------------------------------------------------
# Tokenizer Configuration
# ---------------------------------------------------------------------------

@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE spectral tokenizer.

    Attributes:
        codebook_size: Number of codebook entries.
        codebook_dim: Dimension of codebook vectors.
        num_codebooks: Number of residual codebooks.
        commitment_loss_weight: Weight for commitment loss.
        decay: EMA decay rate for codebook updates.
        threshold_ema_dead_code: Threshold for resetting dead codes.
        use_cosine_similarity: Whether to use cosine similarity for quantization.
        orthogonal_init: Whether to initialize codebook orthogonally.
        temperature: Softmax temperature for soft quantization.
        straight_through: Whether to use straight-through estimator.
    """
    codebook_size: int = 8192
    codebook_dim: int = 256
    num_codebooks: int = 4
    commitment_loss_weight: float = 0.25
    decay: float = 0.99
    threshold_ema_dead_code: float = 2.0
    use_cosine_similarity: bool = True
    orthogonal_init: bool = True
    temperature: float = 1.0
    straight_through: bool = True


@dataclass
class PatchConfig:
    """Configuration for patch-based spectral tokenizer.

    Attributes:
        patch_size: Size of spectral patches in frequency bins.
        patch_stride: Stride between patches.
        time_patch_size: Size of patches in time dimension.
        time_patch_stride: Stride in time dimension.
        num_channels: Number of input channels.
        flatten_patches: Whether to flatten patches before projection.
        use_cls_token: Whether to use a CLS token.
        use_register_tokens: Whether to use register tokens.
        num_register_tokens: Number of register tokens.
    """
    patch_size: int = 16
    patch_stride: int = 16
    time_patch_size: int = 16
    time_patch_stride: int = 8
    num_channels: int = 1
    flatten_patches: bool = True
    use_cls_token: bool = True
    use_register_tokens: bool = True
    num_register_tokens: int = 4


@dataclass
class TokenizerConfig:
    """Complete tokenizer configuration.

    Attributes:
        tokenizer_type: Type of tokenizer architecture.
        vqvae: VQ-VAE configuration.
        patch: Patch tokenizer configuration.
        max_tokens: Maximum number of tokens per sequence.
        special_tokens: Special token definitions.
        domain_tokens: Domain-specific prefix tokens.
        continuous_dim: Dimension for continuous tokenization.
        use_frequency_binning: Whether to bin frequencies.
        num_frequency_bins: Number of frequency bins.
        frequency_scale: Frequency axis scale ('linear', 'log', 'mel', 'bark').
        min_frequency: Minimum frequency for tokenization (Hz).
        max_frequency: Maximum frequency for tokenization (Hz).
        amplitude_quantization_bits: Bits for amplitude quantization.
        phase_quantization_bits: Bits for phase quantization.
        use_delta_encoding: Whether to use delta encoding between frames.
        context_window: Context window for tokenization.
    """
    tokenizer_type: str = TokenizerType.HYBRID
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    max_tokens: int = 4096
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "PAD": 0,
        "BOS": 1,
        "EOS": 2,
        "MASK": 3,
        "SEP": 4,
        "UNK": 5,
        "SEISMIC": 6,
        "VIBRATION": 7,
        "EEG": 8,
        "ECG": 9,
        "AUDIO": 10,
        "RF": 11,
        "SYNTHETIC": 12,
    })
    domain_tokens: Dict[str, str] = field(default_factory=lambda: {
        "seismic": "[SEISMIC]",
        "vibration": "[VIBRATION]",
        "eeg": "[EEG]",
        "ecg": "[ECG]",
        "audio": "[AUDIO]",
        "rf": "[RF]",
        "synthetic": "[SYNTHETIC]",
    })
    continuous_dim: int = 256
    use_frequency_binning: bool = True
    num_frequency_bins: int = 512
    frequency_scale: str = "log"
    min_frequency: float = 0.01
    max_frequency: float = 100000.0
    amplitude_quantization_bits: int = 12
    phase_quantization_bits: int = 8
    use_delta_encoding: bool = True
    context_window: int = 64


# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModalityConfig:
    """Configuration for a single data modality.

    Attributes:
        name: Modality name.
        enabled: Whether this modality is enabled.
        weight: Sampling weight for this modality.
        data_paths: Paths to data directories.
        file_patterns: Glob patterns for file discovery.
        sampling_rate: Expected sampling rate (Hz).
        window_size: Window size in samples.
        hop_size: Hop size in samples.
        num_channels: Number of input channels.
        preprocessing: Preprocessing pipeline configuration.
        augmentation: Data augmentation configuration.
        max_samples: Maximum number of samples to use (0 = unlimited).
        cache_preprocessed: Whether to cache preprocessed data.
        stream_mode: Whether to stream data from disk.
    """
    name: str = ""
    enabled: bool = True
    weight: float = 1.0
    data_paths: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=lambda: ["*.npy", "*.npz", "*.h5"])
    sampling_rate: float = 100.0
    window_size: int = 1024
    hop_size: int = 256
    num_channels: int = 1
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "hann",
        "taper_fraction": 0.05,
    })
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "time_stretch": {"enabled": False, "range": [0.8, 1.2]},
        "frequency_shift": {"enabled": False, "range": [-0.1, 0.1]},
        "noise_injection": {"enabled": True, "snr_db": [10, 40]},
        "amplitude_scale": {"enabled": True, "range": [0.5, 2.0]},
        "phase_perturbation": {"enabled": False, "std": 0.1},
    })
    max_samples: int = 0
    cache_preprocessed: bool = True
    stream_mode: bool = False


@dataclass
class SeismicConfig(ModalityConfig):
    """Seismic-specific data configuration."""
    name: str = "seismic"
    sampling_rate: float = 100.0
    window_size: int = 2048
    hop_size: int = 512
    num_channels: int = 3
    file_patterns: List[str] = field(default_factory=lambda: [
        "*.mseed", "*.sac", "*.h5", "*.npy"
    ])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "hann",
        "taper_fraction": 0.05,
        "instrument_response_removal": True,
        "bandpass_filter": {"low": 0.1, "high": 40.0},
        "resample_rate": 100.0,
    })


@dataclass
class VibrationConfig(ModalityConfig):
    """Vibration-specific data configuration."""
    name: str = "vibration"
    sampling_rate: float = 25600.0
    window_size: int = 4096
    hop_size: int = 1024
    num_channels: int = 3
    file_patterns: List[str] = field(default_factory=lambda: [
        "*.csv", "*.tdms", "*.h5", "*.npy"
    ])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "hann",
        "taper_fraction": 0.02,
        "highpass_filter": {"cutoff": 10.0},
        "envelope_extraction": False,
    })


@dataclass
class EEGConfig(ModalityConfig):
    """EEG-specific data configuration."""
    name: str = "eeg"
    sampling_rate: float = 256.0
    window_size: int = 512
    hop_size: int = 128
    num_channels: int = 64
    file_patterns: List[str] = field(default_factory=lambda: [
        "*.edf", "*.bdf", "*.fif", "*.npy"
    ])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "hann",
        "taper_fraction": 0.02,
        "bandpass_filter": {"low": 0.5, "high": 100.0},
        "notch_filter": [50.0, 60.0],
        "artifact_rejection": True,
        "rereferencing": "average",
        "ica_artifact_removal": False,
    })


@dataclass
class ECGConfig(ModalityConfig):
    """ECG-specific data configuration."""
    name: str = "ecg"
    sampling_rate: float = 500.0
    window_size: int = 2500
    hop_size: int = 500
    num_channels: int = 12
    file_patterns: List[str] = field(default_factory=lambda: [
        "*.h5", "*.mat", "*.npy", "*.wfdb"
    ])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "hann",
        "taper_fraction": 0.01,
        "bandpass_filter": {"low": 0.5, "high": 150.0},
        "baseline_wander_removal": True,
        "powerline_removal": True,
        "r_peak_detection": True,
    })


@dataclass
class AudioConfig(ModalityConfig):
    """Audio spectrogram data configuration."""
    name: str = "audio"
    sampling_rate: float = 22050.0
    window_size: int = 2048
    hop_size: int = 512
    num_channels: int = 1
    file_patterns: List[str] = field(default_factory=lambda: [
        "*.wav", "*.flac", "*.mp3", "*.ogg", "*.npy"
    ])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": False,
        "taper": "hann",
        "taper_fraction": 0.0,
        "mel_spectrogram": True,
        "n_mels": 128,
        "log_scale": True,
        "fmin": 20.0,
        "fmax": 11025.0,
    })


@dataclass
class RFConfig(ModalityConfig):
    """RF sweep data configuration."""
    name: str = "rf"
    sampling_rate: float = 1e6
    window_size: int = 4096
    hop_size: int = 1024
    num_channels: int = 2
    file_patterns: List[str] = field(default_factory=lambda: [
        "*.iq", "*.sigmf", "*.h5", "*.npy"
    ])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "blackman",
        "taper_fraction": 0.05,
        "iq_to_power_spectrum": True,
        "frequency_correction": True,
        "dc_removal": True,
    })


@dataclass
class SyntheticConfig(ModalityConfig):
    """Synthetic physics simulation data configuration."""
    name: str = "synthetic"
    sampling_rate: float = 1000.0
    window_size: int = 2048
    hop_size: int = 512
    num_channels: int = 1
    file_patterns: List[str] = field(default_factory=lambda: ["*.npy", "*.npz", "*.h5"])
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "detrend": True,
        "taper": "hann",
        "taper_fraction": 0.02,
    })
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "parameter_variation": {"enabled": True, "range": 0.2},
        "noise_injection": {"enabled": True, "snr_db": [20, 60]},
        "damping_variation": {"enabled": True, "range": [0.01, 0.1]},
        "frequency_jitter": {"enabled": True, "std": 0.02},
    })


@dataclass
class DataConfig:
    """Complete data pipeline configuration.

    Attributes:
        modalities: Dictionary of modality configurations.
        batch_size: Global batch size.
        num_workers: Number of data loading workers.
        prefetch_factor: Number of batches to prefetch.
        pin_memory: Whether to pin memory for GPU transfer.
        shuffle_buffer_size: Size of the shuffle buffer.
        global_preprocessing: Global preprocessing settings.
        curriculum: Curriculum learning settings.
        sampling_strategy: How to sample across modalities.
        max_total_samples: Maximum total samples across all modalities.
        validation_split: Fraction of data for validation.
        seed: Random seed for reproducibility.
    """
    modalities: Dict[str, ModalityConfig] = field(default_factory=lambda: {
        "seismic": SeismicConfig(),
        "vibration": VibrationConfig(),
        "eeg": EEGConfig(),
        "ecg": ECGConfig(),
        "audio": AudioConfig(),
        "rf": RFConfig(),
        "synthetic": SyntheticConfig(),
    })
    batch_size: int = 256
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    shuffle_buffer_size: int = 100000
    global_preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "global_normalization": "per_sample",
        "clip_amplitude": 10.0,
        "nan_handling": "zero",
        "inf_handling": "clip",
    })
    curriculum: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "phases": [
            {"epoch_start": 0, "modalities": ["synthetic"], "difficulty": "easy"},
            {"epoch_start": 5, "modalities": ["seismic", "vibration", "synthetic"], "difficulty": "medium"},
            {"epoch_start": 15, "modalities": "all", "difficulty": "hard"},
        ],
    })
    sampling_strategy: str = "proportional"
    max_total_samples: int = 0
    validation_split: float = 0.05
    seed: int = 42


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Complete training configuration.

    Attributes:
        max_epochs: Maximum number of training epochs.
        max_steps: Maximum number of training steps (0 = use epochs).
        optimizer: Optimizer type.
        learning_rate: Peak learning rate.
        min_learning_rate: Minimum learning rate for scheduling.
        weight_decay: Weight decay coefficient.
        beta1: Adam beta1.
        beta2: Adam beta2.
        epsilon: Adam epsilon.
        max_grad_norm: Maximum gradient norm for clipping.
        scheduler: Learning rate scheduler type.
        warmup_steps: Number of warmup steps.
        warmup_ratio: Warmup ratio (alternative to warmup_steps).
        accumulation_steps: Gradient accumulation steps.
        mixed_precision: Whether to use mixed precision training.
        precision_type: Precision type ('fp16', 'bf16', 'fp32').
        compile_model: Whether to use torch.compile.
        distributed: Distributed training settings.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_interval: Steps between checkpoints.
        log_interval: Steps between logging.
        eval_interval: Steps between evaluations.
        save_top_k: Number of top checkpoints to keep.
        early_stopping_patience: Patience for early stopping (0 = disabled).
        early_stopping_metric: Metric for early stopping.
        resume_from: Path to resume training from.
        seed: Training random seed.
        deterministic: Whether to use deterministic operations.
    """
    max_epochs: int = 100
    max_steps: int = 0
    optimizer: str = OptimizerType.ADAMW
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler: str = SchedulerType.COSINE_WARMUP
    warmup_steps: int = 2000
    warmup_ratio: float = 0.0
    accumulation_steps: int = 4
    mixed_precision: bool = True
    precision_type: str = "bf16"
    compile_model: bool = False
    distributed: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "backend": "nccl",
        "strategy": "ddp",
        "num_nodes": 1,
        "devices_per_node": 1,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
    })
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 5000
    log_interval: int = 100
    eval_interval: int = 1000
    save_top_k: int = 5
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    resume_from: Optional[str] = None
    seed: int = 42
    deterministic: bool = False


# ---------------------------------------------------------------------------
# Latent Space Configuration
# ---------------------------------------------------------------------------

@dataclass
class LatentSpaceConfig:
    """Configuration for the universal spectral latent space.

    Attributes:
        latent_dim: Dimension of the latent space.
        distribution: Latent distribution type.
        num_components: Number of mixture components (for mixture distribution).
        use_hierarchical: Whether to use hierarchical latent space.
        hierarchy_levels: Number of hierarchy levels.
        hierarchy_dims: Dimensions at each hierarchy level.
        alignment_method: Method for cross-modal alignment.
        alignment_weight: Weight for alignment loss.
        regularization: Regularization settings.
        projection_heads: Modality-specific projection head settings.
        temperature: Temperature for similarity computation.
        use_momentum_encoder: Whether to use momentum encoder.
        momentum: Momentum for EMA updates.
        queue_size: Size of negative sample queue.
        manifold_type: Type of manifold for latent space.
        curvature: Curvature for hyperbolic spaces.
    """
    latent_dim: int = 512
    distribution: str = LatentDistribution.GAUSSIAN
    num_components: int = 16
    use_hierarchical: bool = True
    hierarchy_levels: int = 3
    hierarchy_dims: List[int] = field(default_factory=lambda: [128, 256, 512])
    alignment_method: str = "contrastive"
    alignment_weight: float = 0.1
    regularization: Dict[str, Any] = field(default_factory=lambda: {
        "kl_weight": 0.001,
        "orthogonality_weight": 0.01,
        "uniformity_weight": 0.1,
        "spectral_smoothness": 0.05,
    })
    projection_heads: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "seismic": {"layers": [1024, 512], "activation": "gelu"},
        "vibration": {"layers": [1024, 512], "activation": "gelu"},
        "eeg": {"layers": [1024, 512], "activation": "gelu"},
        "ecg": {"layers": [1024, 512], "activation": "gelu"},
        "audio": {"layers": [1024, 512], "activation": "gelu"},
        "rf": {"layers": [1024, 512], "activation": "gelu"},
        "synthetic": {"layers": [1024, 512], "activation": "gelu"},
    })
    temperature: float = 0.07
    use_momentum_encoder: bool = True
    momentum: float = 0.999
    queue_size: int = 65536
    manifold_type: str = "euclidean"
    curvature: float = 1.0


# ---------------------------------------------------------------------------
# Objective Configuration
# ---------------------------------------------------------------------------

@dataclass
class MaskedSpectralConfig:
    """Configuration for masked spectral modeling objective.

    Attributes:
        mask_ratio: Fraction of tokens to mask.
        mask_strategy: Masking strategy ('random', 'structured', 'frequency_band').
        mask_token_prob: Probability of replacing with mask token.
        random_token_prob: Probability of replacing with random token.
        keep_token_prob: Probability of keeping original token.
        predict_mean: Whether to predict mean of masked region.
        predict_variance: Whether to predict variance of masked region.
        spectral_weighting: Whether to weight loss by spectral importance.
        band_masking_bands: Number of contiguous frequency bands to mask.
        temporal_masking_ratio: Ratio of temporal positions to mask.
    """
    mask_ratio: float = 0.3
    mask_strategy: str = "structured"
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_token_prob: float = 0.1
    predict_mean: bool = True
    predict_variance: bool = True
    spectral_weighting: bool = True
    band_masking_bands: int = 4
    temporal_masking_ratio: float = 0.15


@dataclass
class NextWindowConfig:
    """Configuration for next-window prediction objective.

    Attributes:
        prediction_steps: Number of future windows to predict.
        use_autoregressive: Whether to use autoregressive prediction.
        use_teacher_forcing: Whether to use teacher forcing.
        teacher_forcing_ratio: Ratio for teacher forcing schedule.
        prediction_type: What to predict ('tokens', 'embeddings', 'spectral').
        use_speculative_decoding: Whether to use speculative decoding.
    """
    prediction_steps: int = 4
    use_autoregressive: bool = True
    use_teacher_forcing: bool = True
    teacher_forcing_ratio: float = 0.5
    prediction_type: str = "embeddings"
    use_speculative_decoding: bool = False


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning objective.

    Attributes:
        temperature: Temperature for InfoNCE loss.
        negative_samples: Number of negative samples.
        hard_negative_ratio: Ratio of hard negatives.
        use_cross_modal: Whether to use cross-modal contrastive learning.
        augmentation_pairs: Types of augmentation for positive pairs.
        similarity_metric: Similarity metric ('cosine', 'bilinear', 'euclidean').
        use_prototypes: Whether to use prototypical contrastive learning.
        num_prototypes: Number of prototypes.
        sinkhorn_iterations: Number of Sinkhorn iterations for assignment.
    """
    temperature: float = 0.07
    negative_samples: int = 65536
    hard_negative_ratio: float = 0.3
    use_cross_modal: bool = True
    augmentation_pairs: List[str] = field(default_factory=lambda: [
        "time_stretch", "frequency_shift", "noise", "masking", "channel_drop"
    ])
    similarity_metric: str = "cosine"
    use_prototypes: bool = True
    num_prototypes: int = 3000
    sinkhorn_iterations: int = 3


@dataclass
class PhysicsInformedConfig:
    """Configuration for physics-informed pretraining objectives.

    Attributes:
        conservation_laws: List of conservation laws to enforce.
        symmetry_constraints: Symmetry constraints to enforce.
        pde_residual_weight: Weight for PDE residual loss.
        boundary_condition_weight: Weight for boundary condition loss.
        spectral_decay_prior: Prior on spectral decay rate.
        causality_enforcement: Whether to enforce causality.
        energy_conservation_weight: Weight for energy conservation.
        dispersion_relation_weight: Weight for dispersion relation.
        parseval_weight: Weight for Parseval's theorem consistency.
    """
    conservation_laws: List[str] = field(default_factory=lambda: [
        "energy", "momentum", "parseval"
    ])
    symmetry_constraints: List[str] = field(default_factory=lambda: [
        "time_reversal", "frequency_conjugate"
    ])
    pde_residual_weight: float = 0.01
    boundary_condition_weight: float = 0.01
    spectral_decay_prior: float = -2.0
    causality_enforcement: bool = True
    energy_conservation_weight: float = 0.1
    dispersion_relation_weight: float = 0.05
    parseval_weight: float = 0.1


@dataclass
class ObjectiveConfig:
    """Complete pretraining objectives configuration.

    Attributes:
        objectives: List of active objectives.
        objective_weights: Weights for each objective.
        masked_spectral: Masked spectral modeling config.
        next_window: Next window prediction config.
        contrastive: Contrastive learning config.
        physics_informed: Physics-informed config.
        auxiliary_objectives: Auxiliary objective settings.
        multi_task_strategy: Multi-task balancing strategy.
        loss_scaling: Per-objective loss scaling factors.
    """
    objectives: List[str] = field(default_factory=lambda: [
        "masked_spectral", "next_window", "contrastive", "physics_informed"
    ])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "masked_spectral": 1.0,
        "next_window": 0.5,
        "contrastive": 0.3,
        "physics_informed": 0.1,
        "denoising": 0.2,
        "frequency_prediction": 0.2,
        "cross_modal": 0.3,
        "multi_resolution": 0.1,
    })
    masked_spectral: MaskedSpectralConfig = field(default_factory=MaskedSpectralConfig)
    next_window: NextWindowConfig = field(default_factory=NextWindowConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    physics_informed: PhysicsInformedConfig = field(default_factory=PhysicsInformedConfig)
    auxiliary_objectives: Dict[str, Any] = field(default_factory=lambda: {
        "denoising": {"noise_schedule": "linear", "max_noise_level": 0.5},
        "frequency_prediction": {"num_bins": 128, "classification": True},
        "multi_resolution": {"scales": [1, 2, 4, 8], "loss_type": "l1"},
    })
    multi_task_strategy: str = "uncertainty_weighted"
    loss_scaling: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Configuration for evaluation and benchmarking.

    Attributes:
        metrics: List of evaluation metrics.
        downstream_tasks: Downstream evaluation tasks.
        probing_layers: Layers to probe for linear evaluation.
        few_shot_settings: Few-shot evaluation settings.
        zero_shot_settings: Zero-shot evaluation settings.
        retrieval_settings: Retrieval evaluation settings.
        generation_settings: Generation evaluation settings.
        visualization: Visualization settings.
    """
    metrics: List[str] = field(default_factory=lambda: [
        "reconstruction_mse",
        "spectral_coherence",
        "frequency_accuracy",
        "phase_accuracy",
        "latent_uniformity",
        "latent_alignment",
        "perplexity",
        "codebook_utilization",
    ])
    downstream_tasks: Dict[str, Any] = field(default_factory=lambda: {
        "seismic_event_detection": {"type": "classification", "num_classes": 5},
        "vibration_fault_diagnosis": {"type": "classification", "num_classes": 10},
        "eeg_sleep_staging": {"type": "classification", "num_classes": 5},
        "ecg_arrhythmia": {"type": "classification", "num_classes": 12},
        "audio_scene_classification": {"type": "classification", "num_classes": 10},
        "rf_modulation_recognition": {"type": "classification", "num_classes": 11},
        "spectral_super_resolution": {"type": "regression"},
        "cross_modal_retrieval": {"type": "retrieval"},
        "anomaly_detection": {"type": "anomaly", "contamination": 0.05},
    })
    probing_layers: List[int] = field(default_factory=lambda: [0, 6, 12, 18, 23])
    few_shot_settings: Dict[str, Any] = field(default_factory=lambda: {
        "shots": [1, 5, 10, 50],
        "episodes": 100,
        "metric": "accuracy",
    })
    zero_shot_settings: Dict[str, Any] = field(default_factory=lambda: {
        "prompt_templates": True,
        "spectral_prototypes": True,
    })
    retrieval_settings: Dict[str, Any] = field(default_factory=lambda: {
        "top_k": [1, 5, 10, 50],
        "metrics": ["recall", "ndcg", "map"],
    })
    generation_settings: Dict[str, Any] = field(default_factory=lambda: {
        "num_samples": 1000,
        "metrics": ["fid", "is", "spectral_distance"],
    })
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        "tsne": True,
        "umap": True,
        "pca": True,
        "attention_maps": True,
        "frequency_response": True,
    })


# ---------------------------------------------------------------------------
# Master Configuration
# ---------------------------------------------------------------------------

@dataclass
class PretrainingConfig:
    """Master configuration for MESIE Foundation Model pretraining.

    This is the top-level configuration that contains all sub-configurations
    needed for the complete pretraining pipeline.

    Attributes:
        experiment_name: Name for this experiment run.
        model: Model architecture configuration.
        tokenizer: Tokenizer configuration.
        data: Data pipeline configuration.
        training: Training hyperparameter configuration.
        latent_space: Latent space configuration.
        objectives: Pretraining objectives configuration.
        evaluation: Evaluation and benchmarking configuration.
        logging: Logging configuration.
        wandb: Weights & Biases configuration.
    """
    experiment_name: str = "spectral-gpt-pretrain"
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    latent_space: LatentSpaceConfig = field(default_factory=LatentSpaceConfig)
    objectives: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "log_dir": "./logs",
        "tensorboard": True,
        "console": True,
        "file": True,
    })
    wandb: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "project": "mesie-foundation",
        "entity": None,
        "tags": ["pretraining", "spectral-gpt"],
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PretrainingConfig":
        """Create configuration from dictionary.

        Note: This performs a shallow reconstruction. Nested dataclasses
        are reconstructed from their dictionary representations.
        """
        config = cls()
        if "experiment_name" in data:
            config.experiment_name = data["experiment_name"]
        if "model" in data:
            config.model = ModelConfig(**{
                k: v for k, v in data["model"].items()
                if k in ModelConfig.__dataclass_fields__
            })
        if "training" in data:
            config.training = TrainingConfig(**{
                k: v for k, v in data["training"].items()
                if k in TrainingConfig.__dataclass_fields__
            })
        if "logging" in data:
            config.logging = data["logging"]
        if "wandb" in data:
            config.wandb = data["wandb"]
        return config

    @classmethod
    def from_json(cls, json_str: str) -> "PretrainingConfig":
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> List[str]:
        """Validate configuration consistency.

        Returns:
            List of validation warnings/errors.
        """
        issues: List[str] = []

        # Check model dimensions consistency
        if self.model.hidden_dim != self.model.num_heads * self.model.head_dim:
            issues.append(
                f"hidden_dim ({self.model.hidden_dim}) != "
                f"num_heads ({self.model.num_heads}) * head_dim ({self.model.head_dim})"
            )

        # Check tokenizer consistency
        if self.tokenizer.max_tokens > self.model.max_seq_len:
            issues.append(
                f"tokenizer max_tokens ({self.tokenizer.max_tokens}) > "
                f"model max_seq_len ({self.model.max_seq_len})"
            )

        # Check latent space hierarchy
        if self.latent_space.use_hierarchical:
            if len(self.latent_space.hierarchy_dims) != self.latent_space.hierarchy_levels:
                issues.append(
                    f"hierarchy_dims length ({len(self.latent_space.hierarchy_dims)}) != "
                    f"hierarchy_levels ({self.latent_space.hierarchy_levels})"
                )

        # Check training configuration
        if self.training.warmup_steps > 0 and self.training.warmup_ratio > 0:
            issues.append("Both warmup_steps and warmup_ratio are set; warmup_steps takes precedence")

        # Check objective weights
        for obj in self.objectives.objectives:
            if obj not in self.objectives.objective_weights:
                issues.append(f"Objective '{obj}' missing from objective_weights")

        return issues


# ---------------------------------------------------------------------------
# Preset Configurations
# ---------------------------------------------------------------------------

def spectral_gpt_tiny() -> PretrainingConfig:
    """Tiny configuration for testing and development."""
    config = PretrainingConfig(
        experiment_name="spectral-gpt-tiny",
    )
    config.model.hidden_dim = 256
    config.model.num_layers = 4
    config.model.num_heads = 4
    config.model.head_dim = 64
    config.model.max_seq_len = 512
    config.model.vocab_size = 4096
    config.model.use_mixture_of_experts = False
    config.data.batch_size = 32
    config.training.learning_rate = 1e-3
    config.training.max_epochs = 10
    config.latent_space.latent_dim = 128
    config.latent_space.use_hierarchical = False
    return config


def spectral_gpt_small() -> PretrainingConfig:
    """Small configuration for single-GPU training."""
    config = PretrainingConfig(
        experiment_name="spectral-gpt-small",
    )
    config.model.hidden_dim = 512
    config.model.num_layers = 12
    config.model.num_heads = 8
    config.model.head_dim = 64
    config.model.max_seq_len = 2048
    config.model.vocab_size = 16384
    config.data.batch_size = 64
    config.training.learning_rate = 5e-4
    config.training.max_epochs = 50
    config.latent_space.latent_dim = 256
    return config


def spectral_gpt_base() -> PretrainingConfig:
    """Base configuration for multi-GPU training."""
    return PretrainingConfig(experiment_name="spectral-gpt-base")


def spectral_gpt_large() -> PretrainingConfig:
    """Large configuration for cluster training."""
    config = PretrainingConfig(
        experiment_name="spectral-gpt-large",
    )
    config.model.hidden_dim = 2048
    config.model.num_layers = 36
    config.model.num_heads = 32
    config.model.head_dim = 64
    config.model.max_seq_len = 16384
    config.model.vocab_size = 65536
    config.model.num_experts = 16
    config.model.top_k_experts = 4
    config.data.batch_size = 512
    config.training.learning_rate = 1.5e-4
    config.training.max_epochs = 200
    config.training.accumulation_steps = 8
    config.latent_space.latent_dim = 1024
    config.latent_space.hierarchy_dims = [256, 512, 1024]
    return config


def spectral_gpt_xl() -> PretrainingConfig:
    """Extra-large configuration for large-scale pretraining."""
    config = PretrainingConfig(
        experiment_name="spectral-gpt-xl",
    )
    config.model.hidden_dim = 4096
    config.model.num_layers = 48
    config.model.num_heads = 64
    config.model.head_dim = 64
    config.model.max_seq_len = 32768
    config.model.vocab_size = 131072
    config.model.num_experts = 32
    config.model.top_k_experts = 4
    config.data.batch_size = 1024
    config.training.learning_rate = 1e-4
    config.training.max_epochs = 300
    config.training.accumulation_steps = 16
    config.training.distributed = {
        "enabled": True,
        "backend": "nccl",
        "strategy": "fsdp",
        "num_nodes": 8,
        "devices_per_node": 8,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
    }
    config.latent_space.latent_dim = 2048
    config.latent_space.hierarchy_dims = [512, 1024, 2048]
    return config
