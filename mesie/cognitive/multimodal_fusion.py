"""Multi-Modal Spectral Fusion System.

Provides cross-domain fusion capabilities for integrating multiple
spectral modalities (vibration, acoustic, electromagnetic, thermal, etc.)
into unified representations. Enables multi-sensor intelligence by
learning cross-modal attention and shared latent spaces.

Key Components:
    - ModalityEncoder: Per-modality feature extraction
    - CrossModalAttention: Attention between different modalities
    - FusionGate: Learned gating for modality contribution
    - LatentSpaceAligner: Shared latent space alignment
    - MultiModalFusionPipeline: Full multi-modal processing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Configuration and Data Structures
# =============================================================================


@dataclass
class ModalityConfig:
    """Configuration for a single modality.

    Args:
        modality_id: Unique identifier for this modality.
        input_dim: Input dimension for this modality.
        description: Human-readable description.
        sampling_rate: Sampling rate in Hz.
        frequency_range: Tuple of (min_freq, max_freq) in Hz.
        noise_floor: Expected noise floor level.
    """

    modality_id: str
    input_dim: int = 128
    description: str = ""
    sampling_rate: float = 1000.0
    frequency_range: Tuple[float, float] = (0.0, 500.0)
    noise_floor: float = 0.001


@dataclass
class FusionConfig:
    """Configuration for the multi-modal fusion system.

    Args:
        d_latent: Shared latent space dimension.
        n_attention_heads: Number of cross-modal attention heads.
        fusion_strategy: Strategy for combining modalities.
        alignment_weight: Weight for latent space alignment loss.
        gate_temperature: Temperature for fusion gate softmax.
        max_modalities: Maximum number of modalities supported.
    """

    d_latent: int = 128
    n_attention_heads: int = 4
    fusion_strategy: str = "attention"  # 'attention', 'concatenate', 'gated', 'average'
    alignment_weight: float = 0.1
    gate_temperature: float = 1.0
    max_modalities: int = 8


@dataclass
class FusionResult:
    """Result from multi-modal fusion.

    Args:
        fused_embedding: Combined multi-modal embedding.
        per_modality_embeddings: Individual modality embeddings.
        cross_attention_maps: Cross-modal attention weights.
        gate_values: Fusion gate values per modality.
        alignment_score: Cross-modal alignment quality.
        metadata: Processing metadata.
    """

    fused_embedding: np.ndarray
    per_modality_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    cross_attention_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    gate_values: Dict[str, float] = field(default_factory=dict)
    alignment_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Modality Encoder
# =============================================================================


class ModalityEncoder:
    """Encoder for a single modality.

    Transforms modality-specific input into a standardized
    latent representation suitable for cross-modal fusion.

    Args:
        config: Modality configuration.
        d_latent: Output latent dimension.
        n_layers: Number of encoding layers.
    """

    def __init__(
        self,
        config: ModalityConfig,
        d_latent: int = 128,
        n_layers: int = 2,
    ) -> None:
        self.config = config
        self.d_latent = d_latent
        self.n_layers = n_layers

        # Build encoding layers
        self._layers: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []

        prev_dim = config.input_dim
        for i in range(n_layers - 1):
            hidden_dim = max(d_latent, (prev_dim + d_latent) // 2)
            self._layers.append(
                np.random.randn(prev_dim, hidden_dim) * (2.0 / (prev_dim + hidden_dim)) ** 0.5
            )
            self._biases.append(np.zeros(hidden_dim))
            prev_dim = hidden_dim

        # Final projection to latent space
        self._layers.append(
            np.random.randn(prev_dim, d_latent) * (2.0 / (prev_dim + d_latent)) ** 0.5
        )
        self._biases.append(np.zeros(d_latent))

        # Layer normalization parameters
        self._scale = np.ones(d_latent)
        self._bias_norm = np.zeros(d_latent)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode modality-specific data to latent space.

        Args:
            data: Input data of shape (input_dim,) or (seq_len, input_dim).

        Returns:
            Latent representation of shape (d_latent,) or (seq_len, d_latent).
        """
        data = np.atleast_1d(data).astype(float)
        is_sequence = data.ndim == 2

        if not is_sequence:
            data = data.reshape(1, -1)

        # Resize if needed
        if data.shape[1] != self.config.input_dim:
            resized = np.zeros((data.shape[0], self.config.input_dim))
            for i in range(data.shape[0]):
                resized[i] = np.interp(
                    np.linspace(0, 1, self.config.input_dim),
                    np.linspace(0, 1, data.shape[1]),
                    data[i],
                )
            data = resized

        # Forward through layers
        x = data
        for i, (layer, bias) in enumerate(zip(self._layers, self._biases)):
            x = x @ layer + bias
            if i < len(self._layers) - 1:
                # GELU activation
                x = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

        # Layer normalization
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        x = (x - mean) / std * self._scale + self._bias_norm

        if not is_sequence:
            return x[0]
        return x


# =============================================================================
# Cross-Modal Attention
# =============================================================================


class CrossModalAttention:
    """Cross-modal attention mechanism.

    Computes attention between representations from different modalities,
    allowing each modality to attend to relevant features in other modalities.

    Args:
        d_latent: Latent space dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_latent: int = 128, n_heads: int = 4) -> None:
        self.d_latent = d_latent
        self.n_heads = n_heads
        self.head_dim = d_latent // n_heads

        scale = 0.02
        self.W_q = np.random.randn(d_latent, d_latent) * scale
        self.W_k = np.random.randn(d_latent, d_latent) * scale
        self.W_v = np.random.randn(d_latent, d_latent) * scale
        self.W_o = np.random.randn(d_latent, d_latent) * scale

    def attend(
        self,
        query_modality: np.ndarray,
        key_modality: np.ndarray,
        value_modality: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-modal attention.

        Args:
            query_modality: Query modality embedding(s) (d_latent,) or (n, d_latent).
            key_modality: Key modality embedding(s).
            value_modality: Value modality embeddings (defaults to key_modality).

        Returns:
            Tuple of (attended_output, attention_weights).
        """
        if value_modality is None:
            value_modality = key_modality

        # Ensure 2D
        q = np.atleast_2d(query_modality)
        k = np.atleast_2d(key_modality)
        v = np.atleast_2d(value_modality)

        q_len = q.shape[0]
        k_len = k.shape[0]

        Q = q @ self.W_q
        K = k @ self.W_k
        V = v @ self.W_v

        # Multi-head
        Q = Q.reshape(q_len, self.n_heads, self.head_dim)
        K = K.reshape(k_len, self.n_heads, self.head_dim)
        V = V.reshape(k_len, self.n_heads, self.head_dim)

        # Attention scores
        scores = np.einsum("qhd,khd->hqk", Q, K) / np.sqrt(self.head_dim)

        # Softmax
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / (np.sum(attn, axis=-1, keepdims=True) + 1e-10)

        # Apply to values
        output = np.einsum("hqk,khd->qhd", attn, V)
        output = output.reshape(q_len, self.d_latent)
        output = output @ self.W_o

        avg_attn = np.mean(attn, axis=0)

        if q_len == 1:
            return output[0], avg_attn[0]
        return output, avg_attn


# =============================================================================
# Fusion Gate
# =============================================================================


class FusionGate:
    """Learned gating mechanism for modality fusion.

    Learns to weight the contribution of each modality based on
    the current input context. Can suppress noisy or irrelevant
    modalities dynamically.

    Args:
        d_latent: Latent space dimension.
        max_modalities: Maximum number of modalities.
        temperature: Softmax temperature for gate values.
    """

    def __init__(
        self,
        d_latent: int = 128,
        max_modalities: int = 8,
        temperature: float = 1.0,
    ) -> None:
        self.d_latent = d_latent
        self.max_modalities = max_modalities
        self.temperature = temperature

        # Gate network
        self.W_gate = np.random.randn(d_latent, max_modalities) * 0.02
        self.gate_bias = np.zeros(max_modalities)

    def compute_gates(
        self,
        modality_embeddings: List[np.ndarray],
    ) -> np.ndarray:
        """Compute gating values for each modality.

        Args:
            modality_embeddings: List of modality latent vectors.

        Returns:
            Gate values summing to 1 across modalities.
        """
        n_modalities = len(modality_embeddings)
        if n_modalities == 0:
            return np.array([])

        # Concatenate and compute context
        context = np.mean(modality_embeddings, axis=0)

        # Compute gate logits
        logits = context @ self.W_gate[:, :n_modalities] + self.gate_bias[:n_modalities]
        logits /= self.temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        gates = exp_logits / (np.sum(exp_logits) + 1e-12)

        return gates

    def apply_gates(
        self,
        modality_embeddings: List[np.ndarray],
        gates: np.ndarray,
    ) -> np.ndarray:
        """Apply gates to produce fused output.

        Args:
            modality_embeddings: List of modality embeddings.
            gates: Gate values per modality.

        Returns:
            Gated fusion output.
        """
        output = np.zeros(self.d_latent)
        for i, (emb, gate) in enumerate(zip(modality_embeddings, gates)):
            output += gate * emb
        return output


# =============================================================================
# Latent Space Aligner
# =============================================================================


class LatentSpaceAligner:
    """Align multiple modality latent spaces for cross-modal transfer.

    Learns alignment transformations that map different modality
    representations into a shared semantic space where similar
    content across modalities is nearby.

    Args:
        d_latent: Latent space dimension.
        alignment_strength: Strength of alignment regularization.
    """

    def __init__(
        self,
        d_latent: int = 128,
        alignment_strength: float = 0.1,
    ) -> None:
        self.d_latent = d_latent
        self.alignment_strength = alignment_strength
        self._alignment_transforms: Dict[str, np.ndarray] = {}
        self._reference_modality: Optional[str] = None

    def register_modality(self, modality_id: str) -> None:
        """Register a modality for alignment.

        Args:
            modality_id: Unique modality identifier.
        """
        if modality_id not in self._alignment_transforms:
            # Initialize with identity-like transform
            self._alignment_transforms[modality_id] = (
                np.eye(self.d_latent) + np.random.randn(self.d_latent, self.d_latent) * 0.01
            )

        if self._reference_modality is None:
            self._reference_modality = modality_id

    def align(self, embedding: np.ndarray, modality_id: str) -> np.ndarray:
        """Align a modality embedding to the shared space.

        Args:
            embedding: Modality-specific embedding.
            modality_id: Source modality.

        Returns:
            Aligned embedding in shared space.
        """
        if modality_id not in self._alignment_transforms:
            self.register_modality(modality_id)

        transform = self._alignment_transforms[modality_id]
        aligned = embedding @ transform

        # L2 normalize for alignment
        norm = np.linalg.norm(aligned) + 1e-12
        return aligned / norm * np.linalg.norm(embedding)

    def compute_alignment_score(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> float:
        """Compute how well different modality embeddings are aligned.

        Args:
            embeddings: Dictionary of modality_id -> embedding.

        Returns:
            Alignment score (0-1, higher is better).
        """
        if len(embeddings) < 2:
            return 1.0

        # Align all embeddings
        aligned = {
            mid: self.align(emb, mid) for mid, emb in embeddings.items()
        }

        # Compute pairwise cosine similarities
        modalities = list(aligned.keys())
        similarities = []
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                a = aligned[modalities[i]]
                b = aligned[modalities[j]]
                sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0


# =============================================================================
# Modality-Specific Analyzers
# =============================================================================


class VibrationModalityAnalyzer:
    """Specialized analyzer for vibration spectral data.

    Extracts vibration-specific features including natural frequencies,
    damping ratios, mode shapes, and operational deflection shapes.

    Args:
        d_latent: Output dimension.
        n_modes: Number of modes to extract.
    """

    def __init__(self, d_latent: int = 128, n_modes: int = 10) -> None:
        self.d_latent = d_latent
        self.n_modes = n_modes

    def extract_modal_features(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """Extract modal analysis features from vibration spectrum.

        Args:
            spectrum: Vibration spectrum.

        Returns:
            Dictionary of modal features.
        """
        spectrum = np.atleast_1d(spectrum).flatten()

        # Find natural frequencies (peaks)
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                if spectrum[i] > np.mean(spectrum) + 0.5 * np.std(spectrum):
                    peaks.append(i)

        natural_frequencies = peaks[:self.n_modes]

        # Estimate damping ratios (half-power bandwidth method)
        damping_ratios = []
        for peak_idx in natural_frequencies:
            half_power = spectrum[peak_idx] / np.sqrt(2)
            # Find half-power points
            left = peak_idx
            while left > 0 and spectrum[left] > half_power:
                left -= 1
            right = peak_idx
            while right < len(spectrum) - 1 and spectrum[right] > half_power:
                right += 1
            bandwidth = right - left
            if peak_idx > 0:
                damping = bandwidth / (2 * peak_idx)
            else:
                damping = 0.0
            damping_ratios.append(float(np.clip(damping, 0, 1)))

        # Total RMS
        rms = float(np.sqrt(np.mean(spectrum ** 2)))

        # Crest factor
        crest_factor = float(np.max(np.abs(spectrum)) / (rms + 1e-12))

        return {
            "natural_frequencies": natural_frequencies,
            "damping_ratios": damping_ratios,
            "n_modes_detected": len(natural_frequencies),
            "rms": rms,
            "crest_factor": crest_factor,
            "total_energy": float(np.sum(spectrum ** 2)),
        }


class AcousticModalityAnalyzer:
    """Specialized analyzer for acoustic spectral data.

    Extracts acoustic-specific features including loudness,
    brightness, roughness, and spectral tilt.

    Args:
        d_latent: Output dimension.
        sample_rate: Audio sample rate.
    """

    def __init__(self, d_latent: int = 128, sample_rate: float = 44100.0) -> None:
        self.d_latent = d_latent
        self.sample_rate = sample_rate

    def extract_acoustic_features(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """Extract acoustic features from spectrum.

        Args:
            spectrum: Acoustic spectrum (magnitude).

        Returns:
            Dictionary of acoustic features.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        spectrum = np.abs(spectrum) + 1e-12

        # Spectral centroid (brightness)
        freqs = np.linspace(0, self.sample_rate / 2, len(spectrum))
        total_energy = np.sum(spectrum)
        centroid = float(np.sum(freqs * spectrum) / total_energy)

        # Spectral spread
        spread = float(np.sqrt(np.sum((freqs - centroid) ** 2 * spectrum) / total_energy))

        # Spectral rolloff (95th percentile)
        cumulative = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumulative, 0.95 * total_energy)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral flatness (tonality)
        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = float(geo_mean / (arith_mean + 1e-12))

        # Spectral tilt (slope of log spectrum)
        log_spectrum = np.log(spectrum)
        x = np.arange(len(log_spectrum))
        if len(x) > 1:
            tilt = float(np.polyfit(x, log_spectrum, 1)[0])
        else:
            tilt = 0.0

        # Loudness estimate (dB)
        loudness = float(10 * np.log10(total_energy + 1e-12))

        # Roughness estimate (energy in critical band differences)
        n_bands = min(24, len(spectrum) // 4)
        band_size = len(spectrum) // max(1, n_bands)
        band_energies = [
            np.sum(spectrum[i * band_size:(i + 1) * band_size] ** 2)
            for i in range(n_bands)
        ]
        roughness = float(np.std(band_energies) / (np.mean(band_energies) + 1e-12))

        return {
            "centroid": centroid,
            "spread": spread,
            "rolloff": rolloff,
            "flatness": flatness,
            "tilt": tilt,
            "loudness": loudness,
            "roughness": roughness,
            "total_energy": float(total_energy),
        }


class ElectromagneticModalityAnalyzer:
    """Specialized analyzer for electromagnetic spectral data.

    Extracts EM-specific features including band power, interference
    patterns, and signal-to-noise ratios.

    Args:
        d_latent: Output dimension.
        center_frequency: Center frequency of interest.
    """

    def __init__(self, d_latent: int = 128, center_frequency: float = 1e9) -> None:
        self.d_latent = d_latent
        self.center_frequency = center_frequency

    def extract_em_features(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """Extract electromagnetic features from spectrum.

        Args:
            spectrum: EM spectrum (power spectral density).

        Returns:
            Dictionary of EM features.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        spectrum = np.abs(spectrum) + 1e-12

        # Signal power (around center)
        center_idx = len(spectrum) // 2
        signal_width = max(1, len(spectrum) // 10)
        signal_region = spectrum[
            max(0, center_idx - signal_width):min(len(spectrum), center_idx + signal_width)
        ]
        signal_power = float(np.sum(signal_region))

        # Noise power (outside signal region)
        noise_spectrum = spectrum.copy()
        noise_spectrum[
            max(0, center_idx - signal_width):min(len(spectrum), center_idx + signal_width)
        ] = 0
        noise_power = float(np.sum(noise_spectrum))

        # SNR
        snr = float(10 * np.log10(signal_power / (noise_power + 1e-12)))

        # Bandwidth (3dB)
        peak_power = np.max(spectrum)
        half_power = peak_power / 2
        above_half = np.where(spectrum > half_power)[0]
        bandwidth = float(len(above_half)) / len(spectrum)

        # Interference patterns (periodicity in spectrum)
        autocorr = np.correlate(spectrum - np.mean(spectrum), spectrum - np.mean(spectrum), mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr /= autocorr[0] + 1e-12

        # Find first significant peak in autocorrelation
        interference_period = 0.0
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > 0.3:
                    interference_period = float(i)
                    break

        return {
            "signal_power": signal_power,
            "noise_power": noise_power,
            "snr_db": snr,
            "bandwidth": bandwidth,
            "interference_period": interference_period,
            "peak_power": float(peak_power),
            "spectral_efficiency": float(signal_power / (np.sum(spectrum) + 1e-12)),
        }


# =============================================================================
# Multi-Modal Fusion Pipeline
# =============================================================================


class MultiModalFusionPipeline:
    """Complete multi-modal spectral fusion pipeline.

    Orchestrates per-modality encoding, cross-modal attention,
    gated fusion, and latent space alignment to produce unified
    multi-modal spectral representations.

    Args:
        config: Fusion system configuration.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()

        # Components
        self._encoders: Dict[str, ModalityEncoder] = {}
        self._cross_attention = CrossModalAttention(
            d_latent=self.config.d_latent,
            n_heads=self.config.n_attention_heads,
        )
        self._fusion_gate = FusionGate(
            d_latent=self.config.d_latent,
            max_modalities=self.config.max_modalities,
            temperature=self.config.gate_temperature,
        )
        self._aligner = LatentSpaceAligner(
            d_latent=self.config.d_latent,
            alignment_strength=self.config.alignment_weight,
        )

        # Specialized analyzers
        self._vibration_analyzer = VibrationModalityAnalyzer(d_latent=self.config.d_latent)
        self._acoustic_analyzer = AcousticModalityAnalyzer(d_latent=self.config.d_latent)
        self._em_analyzer = ElectromagneticModalityAnalyzer(d_latent=self.config.d_latent)

        # State
        self._processing_count: int = 0
        self._modality_history: Dict[str, int] = {}

    def register_modality(self, modality_config: ModalityConfig) -> None:
        """Register a new modality with the fusion system.

        Args:
            modality_config: Configuration for the modality.
        """
        self._encoders[modality_config.modality_id] = ModalityEncoder(
            config=modality_config,
            d_latent=self.config.d_latent,
        )
        self._aligner.register_modality(modality_config.modality_id)
        self._modality_history[modality_config.modality_id] = 0

    def fuse(
        self,
        modality_data: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None,
    ) -> FusionResult:
        """Fuse multi-modal spectral data into unified representation.

        Args:
            modality_data: Dictionary of modality_id -> spectral data.
            context: Optional contextual information.

        Returns:
            FusionResult with fused embedding and diagnostics.
        """
        self._processing_count += 1

        # Auto-register unknown modalities
        for mid in modality_data:
            if mid not in self._encoders:
                self.register_modality(ModalityConfig(
                    modality_id=mid,
                    input_dim=len(np.atleast_1d(modality_data[mid]).flatten()),
                ))
            self._modality_history[mid] = self._modality_history.get(mid, 0) + 1

        # Encode each modality
        per_modality_embeddings = {}
        for mid, data in modality_data.items():
            encoder = self._encoders[mid]
            embedding = encoder.encode(data)
            per_modality_embeddings[mid] = embedding

        # Align to shared space
        aligned_embeddings = {
            mid: self._aligner.align(emb, mid)
            for mid, emb in per_modality_embeddings.items()
        }

        # Fusion based on strategy
        modality_list = list(aligned_embeddings.values())
        modality_ids = list(aligned_embeddings.keys())

        cross_attention_maps = {}
        gate_values = {}

        if self.config.fusion_strategy == "attention":
            fused, cross_attention_maps = self._attention_fusion(
                aligned_embeddings
            )
        elif self.config.fusion_strategy == "gated":
            fused, gate_values = self._gated_fusion(modality_list, modality_ids)
        elif self.config.fusion_strategy == "concatenate":
            fused = self._concatenation_fusion(modality_list)
        else:  # average
            fused = np.mean(modality_list, axis=0) if modality_list else np.zeros(self.config.d_latent)

        # Compute alignment score
        alignment_score = self._aligner.compute_alignment_score(per_modality_embeddings)

        return FusionResult(
            fused_embedding=fused,
            per_modality_embeddings=per_modality_embeddings,
            cross_attention_maps=cross_attention_maps,
            gate_values=gate_values,
            alignment_score=alignment_score,
            metadata={
                "n_modalities": len(modality_data),
                "modality_ids": list(modality_data.keys()),
                "fusion_strategy": self.config.fusion_strategy,
                "processing_id": self._processing_count,
            },
        )

    def _attention_fusion(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Fuse using cross-modal attention."""
        modality_ids = list(embeddings.keys())
        emb_list = [embeddings[mid] for mid in modality_ids]

        if len(emb_list) == 0:
            return np.zeros(self.config.d_latent), {}

        if len(emb_list) == 1:
            return emb_list[0], {}

        # Stack for cross-attention
        stacked = np.vstack([e.reshape(1, -1) for e in emb_list])

        # Each modality attends to all others
        attention_maps = {}
        attended_outputs = []

        for i, mid in enumerate(modality_ids):
            query = emb_list[i].reshape(1, -1)
            # Keys/values are all other modalities
            others = np.vstack([
                emb_list[j].reshape(1, -1)
                for j in range(len(emb_list)) if j != i
            ])
            output, attn = self._cross_attention.attend(query, others)
            attended_outputs.append(output.flatten())
            attention_maps[mid] = attn

        # Average attended outputs
        fused = np.mean(attended_outputs, axis=0)
        return fused, attention_maps

    def _gated_fusion(
        self,
        embeddings: List[np.ndarray],
        modality_ids: List[str],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fuse using learned gates."""
        gates = self._fusion_gate.compute_gates(embeddings)
        fused = self._fusion_gate.apply_gates(embeddings, gates)
        gate_dict = {mid: float(g) for mid, g in zip(modality_ids, gates)}
        return fused, gate_dict

    def _concatenation_fusion(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Fuse by concatenation and projection."""
        if not embeddings:
            return np.zeros(self.config.d_latent)

        concatenated = np.concatenate(embeddings)
        # Project back to d_latent
        proj = np.random.randn(len(concatenated), self.config.d_latent) * 0.02
        return concatenated @ proj

    def get_modality_analysis(self, modality_id: str, data: np.ndarray) -> Dict[str, Any]:
        """Get specialized analysis for a specific modality type.

        Args:
            modality_id: Modality identifier.
            data: Spectral data.

        Returns:
            Modality-specific analysis results.
        """
        data = np.atleast_1d(data).flatten()

        if "vibration" in modality_id.lower():
            return self._vibration_analyzer.extract_modal_features(data)
        elif "acoustic" in modality_id.lower() or "audio" in modality_id.lower():
            return self._acoustic_analyzer.extract_acoustic_features(data)
        elif "em" in modality_id.lower() or "electromagnetic" in modality_id.lower():
            return self._em_analyzer.extract_em_features(data)
        else:
            # Generic analysis
            return {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "energy": float(np.sum(data ** 2)),
                "max": float(np.max(data)),
                "min": float(np.min(data)),
                "n_samples": len(data),
            }

    @property
    def registered_modalities(self) -> List[str]:
        """List of registered modality IDs."""
        return list(self._encoders.keys())

    @property
    def processing_count(self) -> int:
        """Total fusion operations performed."""
        return self._processing_count
