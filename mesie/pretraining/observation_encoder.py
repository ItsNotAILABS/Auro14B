"""Observation encoder for agent-level spectral reasoning.

Implements the observation encoding pattern that turns MESIE into a
"sensory cortex" for agents:

    Raw world → spectra → MESIE embedding z_t
    Agent observation: o_t = [z_t, state_t, semantic_t]
    Policy / controller consumes o_t

This module provides the ObservationEncoder that concatenates spectral
embeddings with optional state and semantic modalities to produce unified
agent observations suitable for RL, IL, or planning policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ModalityConfig:
    """Configuration for a single observation modality.

    Attributes
    ----------
    name : str
        Modality name (e.g., 'spectral', 'state', 'semantic').
    dim : int
        Dimensionality of this modality's representation.
    normalize : bool
        Whether to L2-normalize this modality before concatenation.
    weight : float
        Relative weight (scales the modality in the final observation).
    """

    name: str = "spectral"
    dim: int = 64
    normalize: bool = True
    weight: float = 1.0


class SpectralTransform:
    """Transform raw sensor signals into spectral tensors.

    Applies FFT, STFT, or wavelet transforms to convert time-domain
    signals into frequency-domain representations.

    Parameters
    ----------
    method : str
        Transform method: 'fft', 'stft', or 'welch'.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT (only used with 'stft').
    window : str
        Window function name.
    """

    def __init__(
        self,
        method: str = "fft",
        n_fft: int = 256,
        hop_length: int = 128,
        window: str = "hann",
    ) -> None:
        if method not in ("fft", "stft", "welch"):
            raise ValueError(
                f"Unknown transform method: {method}. Use 'fft', 'stft', or 'welch'."
            )
        self.method = method
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

    def _get_window(self, size: int) -> np.ndarray:
        """Get window function array."""
        if self.window == "hann":
            return np.hanning(size)
        elif self.window == "hamming":
            return np.hamming(size)
        elif self.window == "rectangular":
            return np.ones(size)
        else:
            return np.hanning(size)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply spectral transform to a time-domain signal.

        Parameters
        ----------
        signal : ndarray, shape (n_samples,) or (n_channels, n_samples)
            Raw time-domain signal.

        Returns
        -------
        ndarray
            Spectral representation:
            - FFT: shape (n_fft // 2,) magnitude spectrum
            - STFT: shape (n_frames, n_fft // 2) spectrogram
            - Welch: shape (n_fft // 2,) PSD estimate
        """
        signal = np.atleast_1d(signal)

        if signal.ndim == 2:
            # Multi-channel: transform each and average
            results = [self.transform(signal[ch]) for ch in range(signal.shape[0])]
            return np.mean(results, axis=0)

        if self.method == "fft":
            return self._fft(signal)
        elif self.method == "stft":
            return self._stft(signal)
        elif self.method == "welch":
            return self._welch(signal)
        else:
            return self._fft(signal)

    def _fft(self, signal: np.ndarray) -> np.ndarray:
        """Compute FFT magnitude spectrum."""
        n = min(len(signal), self.n_fft)
        windowed = signal[:n] * self._get_window(n)
        spectrum = np.abs(np.fft.rfft(windowed, n=self.n_fft))
        # Exclude DC and Nyquist
        return spectrum[1: self.n_fft // 2 + 1]

    def _stft(self, signal: np.ndarray) -> np.ndarray:
        """Compute STFT magnitude spectrogram."""
        n_frames = max(1, (len(signal) - self.n_fft) // self.hop_length + 1)
        n_bins = self.n_fft // 2

        spectrogram = np.zeros((n_frames, n_bins))
        window = self._get_window(self.n_fft)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end > len(signal):
                break
            frame = signal[start:end] * window
            spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))
            spectrogram[i, :] = spectrum[1: n_bins + 1]

        return spectrogram

    def _welch(self, signal: np.ndarray) -> np.ndarray:
        """Compute Welch PSD estimate."""
        # Use overlapping segments and average
        n_segments = max(1, (len(signal) - self.n_fft) // self.hop_length + 1)
        n_bins = self.n_fft // 2
        psd = np.zeros(n_bins)
        window = self._get_window(self.n_fft)
        window_power = np.sum(window ** 2)

        count = 0
        for i in range(n_segments):
            start = i * self.hop_length
            end = start + self.n_fft
            if end > len(signal):
                break
            frame = signal[start:end] * window
            spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft)) ** 2
            psd += spectrum[1: n_bins + 1]
            count += 1

        if count > 0:
            psd /= count * max(window_power, 1e-8)

        return psd


class ObservationEncoder:
    """Unified observation encoder for agent-level spectral reasoning.

    Combines MESIE spectral embeddings with state and semantic modalities
    to produce the observation vector consumed by agent policies:

        o_t = [z_t, state_t, semantic_t]

    Where:
    - z_t = MESIE(s_t) is the spectral embedding from the MESIE backbone
    - state_t = additional state features (position, velocity, etc.)
    - semantic_t = semantic context (labels, categories, etc.)

    Parameters
    ----------
    spectral_encoder : callable or None
        Function mapping spectral tensor -> embedding z_t.
        If None, a default feature-based encoder is used.
    spectral_transform : SpectralTransform or None
        Transform for converting raw signals to spectra.
    modalities : list of ModalityConfig or None
        Configuration for each observation modality.
    output_dim : int or None
        If set, project final observation to this dimensionality.
    """

    def __init__(
        self,
        spectral_encoder: Optional[Callable] = None,
        spectral_transform: Optional[SpectralTransform] = None,
        modalities: Optional[List[ModalityConfig]] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        self.spectral_encoder = spectral_encoder or self._default_encoder
        self.spectral_transform = spectral_transform or SpectralTransform()
        self.modalities = modalities or [
            ModalityConfig(name="spectral", dim=64, normalize=True, weight=1.0),
            ModalityConfig(name="state", dim=16, normalize=False, weight=1.0),
            ModalityConfig(name="semantic", dim=8, normalize=False, weight=0.5),
        ]
        self.output_dim = output_dim
        self._projection_matrix: Optional[np.ndarray] = None

    @property
    def observation_dim(self) -> int:
        """Total dimension of the observation vector."""
        if self.output_dim is not None:
            return self.output_dim
        return sum(m.dim for m in self.modalities)

    @property
    def modality_names(self) -> List[str]:
        """Names of configured modalities."""
        return [m.name for m in self.modalities]

    def encode_spectral(self, spectrum: np.ndarray) -> np.ndarray:
        """Encode a spectral tensor into the MESIE embedding z_t.

        Parameters
        ----------
        spectrum : ndarray
            Spectral data (output of SpectralTransform or raw spectrum).

        Returns
        -------
        ndarray, shape (spectral_dim,)
            Spectral embedding.
        """
        embedding = self.spectral_encoder(spectrum)
        embedding = np.atleast_1d(embedding).astype(float)

        # Ensure correct dimensionality
        spectral_config = next(
            (m for m in self.modalities if m.name == "spectral"), None
        )
        if spectral_config is not None:
            target_dim = spectral_config.dim
            if len(embedding) > target_dim:
                embedding = embedding[:target_dim]
            elif len(embedding) < target_dim:
                padding = np.zeros(target_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])

        return embedding

    def encode_observation(
        self,
        spectral_input: np.ndarray,
        state: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
        additional_modalities: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Create a full agent observation by combining modalities.

        Parameters
        ----------
        spectral_input : ndarray
            Raw spectrum or time-domain signal.
        state : ndarray or None
            Additional state features (e.g., position, velocity, load).
        semantic : ndarray or None
            Semantic context vector (e.g., encoded labels, categories).
        additional_modalities : dict or None
            Extra named modalities to include.

        Returns
        -------
        ndarray, shape (observation_dim,)
            Concatenated and optionally projected observation vector.
        """
        parts = {}

        # Spectral modality
        z_t = self.encode_spectral(spectral_input)
        parts["spectral"] = z_t

        # State modality
        if state is not None:
            parts["state"] = np.atleast_1d(state).astype(float)

        # Semantic modality
        if semantic is not None:
            parts["semantic"] = np.atleast_1d(semantic).astype(float)

        # Additional modalities
        if additional_modalities:
            for name, vec in additional_modalities.items():
                parts[name] = np.atleast_1d(vec).astype(float)

        # Build observation vector
        observation = self._assemble_observation(parts)

        # Optional projection
        if self.output_dim is not None:
            observation = self._project(observation)

        return observation

    def encode_from_raw_signal(
        self,
        raw_signal: np.ndarray,
        state: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Full pipeline: raw signal → spectrum → observation.

        Parameters
        ----------
        raw_signal : ndarray
            Raw time-domain sensor signal.
        state : ndarray or None
            State features.
        semantic : ndarray or None
            Semantic context.

        Returns
        -------
        ndarray
            Agent observation vector.
        """
        spectrum = self.spectral_transform.transform(raw_signal)
        return self.encode_observation(spectrum, state=state, semantic=semantic)

    def batch_encode(
        self,
        spectral_batch: np.ndarray,
        states: Optional[np.ndarray] = None,
        semantics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch encode multiple observations.

        Parameters
        ----------
        spectral_batch : ndarray, shape (n_samples, n_freq)
            Batch of spectra.
        states : ndarray or None, shape (n_samples, state_dim)
            Batch of state vectors.
        semantics : ndarray or None, shape (n_samples, semantic_dim)
            Batch of semantic vectors.

        Returns
        -------
        ndarray, shape (n_samples, observation_dim)
            Batch of observation vectors.
        """
        n = spectral_batch.shape[0]
        observations = []

        for i in range(n):
            state_i = states[i] if states is not None else None
            sem_i = semantics[i] if semantics is not None else None
            obs = self.encode_observation(spectral_batch[i], state=state_i, semantic=sem_i)
            observations.append(obs)

        return np.array(observations)

    def _assemble_observation(self, parts: Dict[str, np.ndarray]) -> np.ndarray:
        """Assemble modality parts into observation vector."""
        components = []

        for modality in self.modalities:
            if modality.name in parts:
                vec = parts[modality.name]
                # Pad or truncate to expected dim
                if len(vec) > modality.dim:
                    vec = vec[: modality.dim]
                elif len(vec) < modality.dim:
                    padding = np.zeros(modality.dim - len(vec))
                    vec = np.concatenate([vec, padding])

                # Normalize if configured
                if modality.normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 1e-8:
                        vec = vec / norm

                # Apply weight
                vec = vec * modality.weight
                components.append(vec)
            else:
                # Zero-fill missing modalities
                components.append(np.zeros(modality.dim))

        return np.concatenate(components)

    def _project(self, observation: np.ndarray) -> np.ndarray:
        """Project observation to output_dim using random projection."""
        if self.output_dim is None:
            return observation

        input_dim = len(observation)
        if self._projection_matrix is None or self._projection_matrix.shape != (
            input_dim,
            self.output_dim,
        ):
            # Initialize random projection (fixed once created)
            rng = np.random.default_rng(42)
            self._projection_matrix = rng.normal(
                0, 1.0 / np.sqrt(self.output_dim), size=(input_dim, self.output_dim)
            )

        return observation @ self._projection_matrix

    def _default_encoder(self, spectrum: np.ndarray) -> np.ndarray:
        """Default spectral encoder: statistical features."""
        spectrum = np.atleast_1d(spectrum)
        if spectrum.ndim > 1:
            # Flatten STFT to 1D by averaging over time
            spectrum = np.mean(spectrum, axis=0)

        n = len(spectrum)
        if n == 0:
            return np.zeros(16)

        # Statistical features
        mean_amp = np.mean(spectrum)
        std_amp = np.std(spectrum)
        max_amp = np.max(spectrum)
        min_amp = np.min(spectrum)

        # Spectral features
        total_energy = np.sum(spectrum ** 2)
        freqs = np.arange(n, dtype=float)
        amp_sum = np.sum(spectrum)
        if amp_sum > 1e-8:
            centroid = np.sum(freqs * spectrum) / amp_sum
            spread = np.sqrt(max(0.0, np.sum(((freqs - centroid) ** 2) * spectrum) / amp_sum))
        else:
            centroid = 0.0
            spread = 0.0

        peak_idx = np.argmax(spectrum) / max(n - 1, 1)

        # Entropy
        prob = np.abs(spectrum) + 1e-10
        prob = prob / prob.sum()
        entropy = -np.sum(prob * np.log(prob))

        # Band energies (4 bands)
        band_size = max(1, n // 4)
        bands = [
            np.sum(spectrum[i * band_size: (i + 1) * band_size] ** 2)
            for i in range(4)
        ]

        features = np.array([
            mean_amp, std_amp, max_amp, min_amp,
            total_energy, centroid, spread, peak_idx,
            entropy, *bands,
            max_amp / max(mean_amp, 1e-8),  # peak-to-mean ratio
            std_amp / max(mean_amp, 1e-8),  # coefficient of variation
            float(np.median(spectrum)),
        ])

        return features


class LineageConditionedEncoder(ObservationEncoder):
    """Observation encoder with lineage conditioning.

    Extends ObservationEncoder to condition on retrieved past spectral
    memories, enabling agents to reason about temporal lineage:

        o_t = [z_t, z_retrieved, state_t, semantic_t]

    Parameters
    ----------
    memory_store : object or None
        Spectral memory store supporting query(embedding) -> context.
    lineage_dim : int
        Dimensionality of the lineage/retrieved context.
    **kwargs
        Additional arguments passed to ObservationEncoder.
    """

    def __init__(
        self,
        memory_store=None,
        lineage_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.memory_store = memory_store
        self.lineage_dim = lineage_dim

        # Add lineage as an additional modality
        self.modalities.append(
            ModalityConfig(name="lineage", dim=lineage_dim, normalize=True, weight=0.8)
        )

    def encode_with_lineage(
        self,
        spectral_input: np.ndarray,
        state: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode observation with lineage-conditioned context.

        Parameters
        ----------
        spectral_input : ndarray
            Spectral data.
        state : ndarray or None
            State features.
        semantic : ndarray or None
            Semantic context.

        Returns
        -------
        ndarray
            Observation vector including retrieved lineage context.
        """
        z_t = self.encode_spectral(spectral_input)

        # Retrieve relevant past context
        lineage_context = self._retrieve_lineage(z_t)

        return self.encode_observation(
            spectral_input,
            state=state,
            semantic=semantic,
            additional_modalities={"lineage": lineage_context},
        )

    def _retrieve_lineage(self, embedding: np.ndarray) -> np.ndarray:
        """Retrieve lineage context from memory store.

        Parameters
        ----------
        embedding : ndarray
            Current spectral embedding to query against.

        Returns
        -------
        ndarray, shape (lineage_dim,)
            Retrieved lineage context.
        """
        if self.memory_store is None:
            return np.zeros(self.lineage_dim)

        # Use memory store's lineage retrieval if available
        if hasattr(self.memory_store, "get_lineage"):
            context = self.memory_store.get_lineage(embedding)
            # Truncate or pad to lineage_dim
            if len(context) > self.lineage_dim:
                return context[: self.lineage_dim]
            elif len(context) < self.lineage_dim:
                padding = np.zeros(self.lineage_dim - len(context))
                return np.concatenate([context, padding])
            return context
        elif hasattr(self.memory_store, "query_simple"):
            results = self.memory_store.query_simple(embedding, top_k=3)
            if results:
                # Average retrieved embeddings
                retrieved = np.mean(
                    [r.embedding for r in results if r.embedding is not None],
                    axis=0,
                )
                if len(retrieved) > self.lineage_dim:
                    return retrieved[: self.lineage_dim]
                elif len(retrieved) < self.lineage_dim:
                    padding = np.zeros(self.lineage_dim - len(retrieved))
                    return np.concatenate([retrieved, padding])
                return retrieved

        return np.zeros(self.lineage_dim)
