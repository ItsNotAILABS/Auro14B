"""Multi-modal spectral datasets for foundation model pretraining.

This module provides dataset classes for each supported spectral modality,
as well as a unified multi-modal dataset that handles cross-domain sampling
and curriculum learning for the SpectralGPT pretraining pipeline.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class SpectralSample:
    """A single spectral sample with metadata.

    Attributes:
        data: Raw spectral data array.
        modality: Source modality name.
        sampling_rate: Original sampling rate (Hz).
        metadata: Additional metadata.
        label: Optional label for supervised tasks.
        window_id: Window index within the source signal.
        source_file: Source file path.
    """

    def __init__(
        self,
        data: np.ndarray,
        modality: str = "unknown",
        sampling_rate: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        label: Optional[int] = None,
        window_id: int = 0,
        source_file: str = "",
    ):
        """Initialize spectral sample.

        Args:
            data: Spectral data array.
            modality: Modality name.
            sampling_rate: Sampling rate in Hz.
            metadata: Additional metadata.
            label: Optional classification label.
            window_id: Window index.
            source_file: Source file path.
        """
        self.data = data
        self.modality = modality
        self.sampling_rate = sampling_rate
        self.metadata = metadata or {}
        self.label = label
        self.window_id = window_id
        self.source_file = source_file

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get data shape."""
        return self.data.shape

    @property
    def duration(self) -> float:
        """Get sample duration in seconds."""
        return self.data.shape[-1] / self.sampling_rate

    @property
    def num_channels(self) -> int:
        """Get number of channels."""
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]


class SpectralDataset:
    """Base class for spectral datasets.

    Provides common functionality for loading, windowing, and
    iterating over spectral data from various sources.

    Attributes:
        name: Dataset name.
        modality: Data modality.
        window_size: Window size in samples.
        hop_size: Hop size between windows.
        sampling_rate: Sampling rate (Hz).
        transform: Optional data transformation.
        max_samples: Maximum number of samples.
    """

    def __init__(
        self,
        name: str = "spectral_dataset",
        modality: str = "generic",
        data_paths: Optional[List[str]] = None,
        window_size: int = 1024,
        hop_size: int = 256,
        sampling_rate: float = 100.0,
        num_channels: int = 1,
        transform: Optional[Callable] = None,
        max_samples: int = 0,
        seed: int = 42,
    ):
        """Initialize spectral dataset.

        Args:
            name: Dataset identifier.
            modality: Data modality name.
            data_paths: Paths to data files/directories.
            window_size: Window size for segmentation.
            hop_size: Hop between windows.
            sampling_rate: Expected sampling rate.
            num_channels: Number of data channels.
            transform: Optional transformation function.
            max_samples: Maximum samples (0=unlimited).
            seed: Random seed.
        """
        self.name = name
        self.modality = modality
        self.data_paths = data_paths or []
        self.window_size = window_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.transform = transform
        self.max_samples = max_samples
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Sample storage
        self._samples: List[SpectralSample] = []
        self._index = 0
        self._initialized = False

    def _generate_synthetic_data(self, num_samples: int = 1000) -> None:
        """Generate synthetic data for testing when no real data is available.

        Args:
            num_samples: Number of synthetic samples to generate.
        """
        for i in range(num_samples):
            if self.max_samples > 0 and len(self._samples) >= self.max_samples:
                break

            # Generate random spectral pattern
            data = self._generate_sample(i)
            sample = SpectralSample(
                data=data,
                modality=self.modality,
                sampling_rate=self.sampling_rate,
                window_id=i,
                source_file=f"synthetic_{self.modality}_{i}",
                metadata={"synthetic": True, "sample_idx": i},
            )
            self._samples.append(sample)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate a single synthetic sample. Override in subclasses.

        Args:
            idx: Sample index for reproducibility.

        Returns:
            Generated data array.
        """
        rng = np.random.RandomState(self.seed + idx)
        if self.num_channels > 1:
            return rng.randn(self.num_channels, self.window_size).astype(np.float32)
        return rng.randn(self.window_size).astype(np.float32)

    def initialize(self, generate_if_empty: bool = True) -> None:
        """Initialize dataset by loading or generating data.

        Args:
            generate_if_empty: Whether to generate synthetic data if no files found.
        """
        if self._initialized:
            return

        # Try to find data files
        files_found = self._discover_files()

        if not files_found and generate_if_empty:
            self._generate_synthetic_data()

        self._initialized = True

    def _discover_files(self) -> bool:
        """Discover data files in configured paths.

        Returns:
            True if files were found.
        """
        # In production, this would scan paths for matching files
        # For the framework, we generate synthetic data
        return False

    def __len__(self) -> int:
        """Get dataset length."""
        if not self._initialized:
            self.initialize()
        return len(self._samples)

    def __getitem__(self, idx: int) -> SpectralSample:
        """Get sample by index.

        Args:
            idx: Sample index.

        Returns:
            SpectralSample instance.
        """
        if not self._initialized:
            self.initialize()

        sample = self._samples[idx % len(self._samples)]

        if self.transform is not None:
            transformed_data = self.transform(sample.data)
            return SpectralSample(
                data=transformed_data,
                modality=sample.modality,
                sampling_rate=sample.sampling_rate,
                metadata=sample.metadata,
                label=sample.label,
                window_id=sample.window_id,
                source_file=sample.source_file,
            )

        return sample

    def __iter__(self) -> Iterator[SpectralSample]:
        """Iterate over dataset."""
        if not self._initialized:
            self.initialize()
        self._index = 0
        return self

    def __next__(self) -> SpectralSample:
        """Get next sample."""
        if self._index >= len(self._samples):
            raise StopIteration
        sample = self[self._index]
        self._index += 1
        return sample

    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[SpectralSample]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples.
            shuffle: Whether to shuffle selection.

        Returns:
            List of samples.
        """
        if not self._initialized:
            self.initialize()

        if shuffle:
            indices = self._rng.choice(len(self._samples), batch_size, replace=True)
        else:
            indices = list(range(min(batch_size, len(self._samples))))

        return [self[i] for i in indices]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self._initialized:
            self.initialize()

        if not self._samples:
            return {"num_samples": 0}

        all_data = np.array([s.data.flatten()[:100] for s in self._samples[:100]])

        return {
            "name": self.name,
            "modality": self.modality,
            "num_samples": len(self._samples),
            "window_size": self.window_size,
            "sampling_rate": self.sampling_rate,
            "num_channels": self.num_channels,
            "data_mean": float(np.mean(all_data)),
            "data_std": float(np.std(all_data)),
            "data_min": float(np.min(all_data)),
            "data_max": float(np.max(all_data)),
        }


class SeismicDataset(SpectralDataset):
    """Dataset for seismic waveform windows.

    Generates realistic seismic-like signals with P-waves, S-waves,
    and surface waves at various magnitudes and distances.
    """

    def __init__(self, **kwargs):
        """Initialize seismic dataset."""
        defaults = {
            "name": "seismic",
            "modality": "seismic",
            "window_size": 2048,
            "hop_size": 512,
            "sampling_rate": 100.0,
            "num_channels": 3,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic seismic waveform."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)

        channels = []
        for ch in range(self.num_channels):
            # P-wave arrival
            p_time = rng.uniform(0.1, 0.3) * t[-1]
            p_freq = rng.uniform(1.0, 15.0)
            p_amp = rng.uniform(0.1, 1.0)
            p_wave = p_amp * np.exp(-3.0 * (t - p_time) ** 2 / (0.1 * t[-1]) ** 2) * \
                np.sin(2 * np.pi * p_freq * (t - p_time))

            # S-wave arrival
            s_time = p_time + rng.uniform(0.1, 0.3) * t[-1]
            s_freq = rng.uniform(0.5, 8.0)
            s_amp = rng.uniform(0.3, 2.0)
            s_wave = s_amp * np.exp(-2.0 * (t - s_time) ** 2 / (0.15 * t[-1]) ** 2) * \
                np.sin(2 * np.pi * s_freq * (t - s_time))

            # Surface wave (low frequency, late arrival)
            surf_time = s_time + rng.uniform(0.05, 0.2) * t[-1]
            surf_freq = rng.uniform(0.1, 2.0)
            surf_amp = rng.uniform(0.5, 3.0)
            surf_wave = surf_amp * np.exp(-1.0 * np.maximum(0, t - surf_time) / (0.3 * t[-1])) * \
                np.sin(2 * np.pi * surf_freq * (t - surf_time)) * (t >= surf_time)

            # Background noise
            noise = rng.randn(self.window_size) * 0.05

            # Coda waves
            coda_start = surf_time + 0.1 * t[-1]
            coda_decay = rng.uniform(1.0, 5.0)
            coda = 0.1 * rng.randn(self.window_size) * np.exp(-coda_decay * np.maximum(0, t - coda_start))

            signal = p_wave + s_wave + surf_wave + noise + coda
            channels.append(signal.astype(np.float32))

        return np.array(channels)


class VibrationDataset(SpectralDataset):
    """Dataset for machine vibration signals.

    Generates realistic vibration patterns including bearing faults,
    gear mesh frequencies, shaft imbalance, and resonances.
    """

    def __init__(self, **kwargs):
        """Initialize vibration dataset."""
        defaults = {
            "name": "vibration",
            "modality": "vibration",
            "window_size": 4096,
            "hop_size": 1024,
            "sampling_rate": 25600.0,
            "num_channels": 3,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic vibration signal."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)

        channels = []
        for ch in range(self.num_channels):
            signal = np.zeros(self.window_size)

            # Shaft rotation frequency
            shaft_freq = rng.uniform(20.0, 120.0)
            shaft_amp = rng.uniform(0.1, 0.5)
            signal += shaft_amp * np.sin(2 * np.pi * shaft_freq * t)

            # Harmonics of shaft frequency
            num_harmonics = rng.randint(3, 8)
            for h in range(2, num_harmonics + 2):
                harm_amp = shaft_amp * rng.uniform(0.1, 0.5) / h
                signal += harm_amp * np.sin(2 * np.pi * h * shaft_freq * t + rng.uniform(0, 2 * np.pi))

            # Bearing fault frequencies (BPFO, BPFI, BSF, FTF)
            if rng.random() > 0.5:
                fault_type = rng.choice(["bpfo", "bpfi", "bsf"])
                if fault_type == "bpfo":
                    fault_freq = shaft_freq * rng.uniform(3.0, 5.0)
                elif fault_type == "bpfi":
                    fault_freq = shaft_freq * rng.uniform(5.0, 8.0)
                else:
                    fault_freq = shaft_freq * rng.uniform(1.5, 3.0)

                fault_amp = rng.uniform(0.05, 0.3)
                # Impulsive modulation
                impulse_env = np.abs(np.sin(2 * np.pi * fault_freq * t))
                carrier_freq = rng.uniform(2000.0, 8000.0)
                signal += fault_amp * impulse_env * np.sin(2 * np.pi * carrier_freq * t)

            # Gear mesh frequency
            if rng.random() > 0.3:
                num_teeth = rng.randint(20, 60)
                mesh_freq = shaft_freq * num_teeth
                mesh_amp = rng.uniform(0.05, 0.2)
                signal += mesh_amp * np.sin(2 * np.pi * mesh_freq * t)
                # Sidebands
                for sb in range(1, 4):
                    sb_amp = mesh_amp * 0.3 / sb
                    signal += sb_amp * np.sin(2 * np.pi * (mesh_freq + sb * shaft_freq) * t)
                    signal += sb_amp * np.sin(2 * np.pi * (mesh_freq - sb * shaft_freq) * t)

            # Structural resonances
            num_resonances = rng.randint(1, 4)
            for _ in range(num_resonances):
                res_freq = rng.uniform(500.0, 5000.0)
                res_amp = rng.uniform(0.01, 0.1)
                res_damping = rng.uniform(0.001, 0.01)
                signal += res_amp * np.exp(-res_damping * 2 * np.pi * res_freq * t) * \
                    np.sin(2 * np.pi * res_freq * t)

            # Background noise
            noise_level = rng.uniform(0.01, 0.05)
            signal += noise_level * rng.randn(self.window_size)

            channels.append(signal.astype(np.float32))

        return np.array(channels)


class EEGDataset(SpectralDataset):
    """Dataset for EEG (electroencephalography) signals.

    Generates realistic EEG-like signals with alpha, beta, theta,
    delta, and gamma rhythms.
    """

    def __init__(self, **kwargs):
        """Initialize EEG dataset."""
        defaults = {
            "name": "eeg",
            "modality": "eeg",
            "window_size": 512,
            "hop_size": 128,
            "sampling_rate": 256.0,
            "num_channels": 64,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic EEG signal."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)

        # Brain rhythm parameters
        rhythms = {
            "delta": {"freq_range": (0.5, 4.0), "amp_range": (20, 100)},
            "theta": {"freq_range": (4.0, 8.0), "amp_range": (10, 50)},
            "alpha": {"freq_range": (8.0, 13.0), "amp_range": (20, 80)},
            "beta": {"freq_range": (13.0, 30.0), "amp_range": (5, 30)},
            "gamma": {"freq_range": (30.0, 100.0), "amp_range": (2, 10)},
        }

        # Simulate different brain states
        state = rng.choice(["awake_eyes_open", "awake_eyes_closed", "drowsy", "sleep_n2", "sleep_n3"])

        # Adjust rhythm amplitudes based on state
        state_modifiers = {
            "awake_eyes_open": {"delta": 0.2, "theta": 0.3, "alpha": 0.5, "beta": 1.5, "gamma": 1.2},
            "awake_eyes_closed": {"delta": 0.3, "theta": 0.4, "alpha": 2.0, "beta": 0.8, "gamma": 0.5},
            "drowsy": {"delta": 0.5, "theta": 1.5, "alpha": 1.0, "beta": 0.5, "gamma": 0.3},
            "sleep_n2": {"delta": 1.0, "theta": 1.0, "alpha": 0.3, "beta": 0.3, "gamma": 0.2},
            "sleep_n3": {"delta": 2.0, "theta": 0.5, "alpha": 0.1, "beta": 0.1, "gamma": 0.1},
        }

        modifiers = state_modifiers[state]

        channels = []
        for ch in range(min(self.num_channels, 64)):
            signal = np.zeros(self.window_size)

            # Add each brain rhythm
            for rhythm_name, params in rhythms.items():
                freq = rng.uniform(*params["freq_range"])
                amp = rng.uniform(*params["amp_range"]) * modifiers[rhythm_name]
                amp *= 1e-6  # Convert to microvolts scale normalized

                # Add some amplitude modulation
                am_freq = rng.uniform(0.1, 1.0)
                am = 1.0 + 0.3 * np.sin(2 * np.pi * am_freq * t)

                # Phase varies by channel (spatial pattern)
                phase = ch * rng.uniform(0, 0.5)
                signal += amp * am * np.sin(2 * np.pi * freq * t + phase)

            # Add 1/f noise (pink noise component)
            white = rng.randn(self.window_size)
            freqs = np.fft.rfftfreq(self.window_size, 1.0 / self.sampling_rate)
            freqs[0] = 1
            spectrum = np.fft.rfft(white)
            pink_spectrum = spectrum / np.sqrt(freqs)
            pink_noise = np.fft.irfft(pink_spectrum, n=self.window_size)
            signal += pink_noise * 5e-6

            # Occasional artifacts (eye blinks, muscle)
            if rng.random() < 0.1 and ch < 4:  # Frontal channels
                blink_time = rng.uniform(0.2, 0.8) * t[-1]
                blink_amp = rng.uniform(50, 200) * 1e-6
                blink = blink_amp * np.exp(-((t - blink_time) / 0.1) ** 2)
                signal += blink

            # Channel-specific noise
            signal += rng.randn(self.window_size) * 2e-6

            channels.append(signal.astype(np.float32))

        return np.array(channels)


class ECGDataset(SpectralDataset):
    """Dataset for ECG (electrocardiography) signals.

    Generates realistic ECG waveforms with P-QRS-T morphology
    at various heart rates and conditions.
    """

    def __init__(self, **kwargs):
        """Initialize ECG dataset."""
        defaults = {
            "name": "ecg",
            "modality": "ecg",
            "window_size": 2500,
            "hop_size": 500,
            "sampling_rate": 500.0,
            "num_channels": 12,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic ECG signal."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)

        # Heart rate
        hr = rng.uniform(50, 120)  # beats per minute
        rr_interval = 60.0 / hr  # seconds

        channels = []
        for ch in range(min(self.num_channels, 12)):
            signal = np.zeros(self.window_size)

            # Lead-specific amplitude scaling
            lead_scales = [1.0, 0.8, 0.6, 0.3, 0.4, 0.5, 1.0, 0.9, 0.7, 0.5, 0.6, 0.4]
            lead_scale = lead_scales[ch % len(lead_scales)]

            # Generate beats
            beat_times = np.arange(0, t[-1], rr_interval)
            # Add RR variability (HRV)
            hrv = rng.randn(len(beat_times)) * 0.02 * rr_interval
            beat_times = beat_times + np.cumsum(hrv)

            for beat_time in beat_times:
                if beat_time > t[-1]:
                    break

                # P wave
                p_amp = rng.uniform(0.1, 0.25) * lead_scale
                p_width = rng.uniform(0.06, 0.1)
                p_offset = -0.16
                p_wave = p_amp * np.exp(-((t - beat_time - p_offset) / p_width) ** 2)

                # QRS complex
                q_amp = -rng.uniform(0.05, 0.15) * lead_scale
                q_width = 0.02
                q_offset = -0.04
                q_wave = q_amp * np.exp(-((t - beat_time - q_offset) / q_width) ** 2)

                r_amp = rng.uniform(0.8, 1.5) * lead_scale
                r_width = rng.uniform(0.02, 0.04)
                r_wave = r_amp * np.exp(-((t - beat_time) / r_width) ** 2)

                s_amp = -rng.uniform(0.1, 0.3) * lead_scale
                s_width = 0.025
                s_offset = 0.04
                s_wave = s_amp * np.exp(-((t - beat_time - s_offset) / s_width) ** 2)

                # T wave
                t_amp = rng.uniform(0.2, 0.5) * lead_scale
                t_width = rng.uniform(0.08, 0.14)
                t_offset = rng.uniform(0.18, 0.3)
                t_wave = t_amp * np.exp(-((t - beat_time - t_offset) / t_width) ** 2)

                signal += p_wave + q_wave + r_wave + s_wave + t_wave

            # Baseline wander (low frequency drift)
            bw_freq = rng.uniform(0.05, 0.5)
            bw_amp = rng.uniform(0.01, 0.1)
            signal += bw_amp * np.sin(2 * np.pi * bw_freq * t)

            # Noise
            signal += rng.randn(self.window_size) * 0.02

            channels.append(signal.astype(np.float32))

        return np.array(channels)


class AudioSpectrogramDataset(SpectralDataset):
    """Dataset for audio spectrograms.

    Generates synthetic audio-like spectrograms with harmonic structures,
    formants, transients, and noise characteristics.
    """

    def __init__(self, **kwargs):
        """Initialize audio spectrogram dataset."""
        defaults = {
            "name": "audio",
            "modality": "audio",
            "window_size": 2048,
            "hop_size": 512,
            "sampling_rate": 22050.0,
            "num_channels": 1,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic audio signal."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)

        signal = np.zeros(self.window_size)

        # Choose sound type
        sound_type = rng.choice(["tonal", "speech_like", "percussive", "environmental"])

        if sound_type == "tonal":
            # Harmonic series (musical instrument-like)
            fundamental = rng.uniform(80.0, 800.0)
            num_harmonics = rng.randint(5, 20)

            for h in range(1, num_harmonics + 1):
                amp = rng.uniform(0.1, 1.0) / (h ** rng.uniform(0.5, 1.5))
                freq = fundamental * h
                if freq > self.sampling_rate / 2:
                    break
                # Slight inharmonicity
                freq *= (1 + rng.uniform(-0.001, 0.001) * h ** 2)
                phase = rng.uniform(0, 2 * np.pi)
                signal += amp * np.sin(2 * np.pi * freq * t + phase)

            # Amplitude envelope (ADSR-like)
            attack = int(rng.uniform(0.01, 0.1) * self.window_size)
            decay = int(rng.uniform(0.05, 0.2) * self.window_size)
            sustain_level = rng.uniform(0.3, 0.8)
            release = int(rng.uniform(0.1, 0.3) * self.window_size)

            envelope = np.ones(self.window_size) * sustain_level
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
            if release < self.window_size:
                envelope[-release:] = np.linspace(sustain_level, 0, release)

            signal *= envelope

        elif sound_type == "speech_like":
            # Formant-based synthesis
            formant_freqs = [
                rng.uniform(200, 900),
                rng.uniform(800, 2500),
                rng.uniform(2000, 3500),
                rng.uniform(3000, 4500),
            ]
            formant_bandwidths = [80, 100, 120, 150]

            # Glottal pulse train
            f0 = rng.uniform(80, 300)
            glottal = np.zeros(self.window_size)
            pulse_period = int(self.sampling_rate / f0)
            for i in range(0, self.window_size, max(1, pulse_period)):
                if i < self.window_size:
                    # Glottal pulse (simplified)
                    pulse_len = min(pulse_period // 2, self.window_size - i)
                    pulse = np.sin(np.pi * np.arange(pulse_len) / pulse_len) ** 2
                    glottal[i:i + pulse_len] += pulse

            # Apply formant filters (simplified via additive)
            for freq, bw in zip(formant_freqs, formant_bandwidths):
                amp = rng.uniform(0.2, 1.0) * np.exp(-freq / 3000)
                formant_signal = amp * np.sin(2 * np.pi * freq * t)
                # Modulate by glottal
                signal += formant_signal * glottal * 0.5

            signal += glottal * 0.1

        elif sound_type == "percussive":
            # Transient + resonance
            # Attack transient
            attack_len = int(rng.uniform(0.001, 0.01) * self.sampling_rate)
            transient = np.zeros(self.window_size)
            transient[:attack_len] = rng.randn(attack_len) * 2.0

            # Resonant modes
            num_modes = rng.randint(3, 10)
            for _ in range(num_modes):
                mode_freq = rng.uniform(100, 8000)
                mode_amp = rng.uniform(0.1, 0.5)
                mode_decay = rng.uniform(5, 50)
                mode = mode_amp * np.exp(-mode_decay * t) * np.sin(2 * np.pi * mode_freq * t)
                signal += mode

            signal += transient

        else:  # environmental
            # Broadband noise with spectral shape
            white = rng.randn(self.window_size)
            # Shape the spectrum
            n_fft = self.window_size
            spectrum = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sampling_rate)

            # Random spectral envelope
            num_peaks = rng.randint(2, 6)
            spectral_env = np.ones(len(freqs)) * 0.1
            for _ in range(num_peaks):
                peak_freq = rng.uniform(100, 8000)
                peak_width = rng.uniform(100, 2000)
                peak_amp = rng.uniform(0.5, 2.0)
                spectral_env += peak_amp * np.exp(-((freqs - peak_freq) / peak_width) ** 2)

            spectrum *= spectral_env
            signal = np.fft.irfft(spectrum, n=n_fft)

        # Final normalization
        max_val = np.max(np.abs(signal)) + 1e-10
        signal = signal / max_val * rng.uniform(0.3, 0.9)

        return signal.astype(np.float32)


class RFSweepDataset(SpectralDataset):
    """Dataset for RF (Radio Frequency) sweep signals.

    Generates synthetic RF signals with various modulation schemes,
    carrier frequencies, and interference patterns.
    """

    def __init__(self, **kwargs):
        """Initialize RF sweep dataset."""
        defaults = {
            "name": "rf",
            "modality": "rf",
            "window_size": 4096,
            "hop_size": 1024,
            "sampling_rate": 1e6,
            "num_channels": 2,  # I/Q channels
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic RF signal (I/Q)."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)

        # Choose modulation type
        mod_type = rng.choice(["am", "fm", "psk", "qam", "ofdm", "chirp"])

        # Carrier frequency (relative to sampling rate)
        carrier_freq = rng.uniform(0.05, 0.4) * self.sampling_rate

        if mod_type == "am":
            # AM signal
            mod_freq = rng.uniform(100, 10000)
            mod_depth = rng.uniform(0.3, 0.9)
            modulator = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
            i_channel = modulator * np.cos(2 * np.pi * carrier_freq * t)
            q_channel = modulator * np.sin(2 * np.pi * carrier_freq * t)

        elif mod_type == "fm":
            # FM signal
            mod_freq = rng.uniform(100, 5000)
            deviation = rng.uniform(1000, 50000)
            phase = 2 * np.pi * carrier_freq * t + \
                (deviation / mod_freq) * np.sin(2 * np.pi * mod_freq * t)
            i_channel = np.cos(phase)
            q_channel = np.sin(phase)

        elif mod_type == "psk":
            # PSK signal
            symbol_rate = rng.uniform(1000, 100000)
            m_ary = rng.choice([2, 4, 8])
            symbols_per_sample = int(self.window_size * symbol_rate / self.sampling_rate)
            symbols = rng.randint(0, m_ary, symbols_per_sample)
            phases = symbols * (2 * np.pi / m_ary)

            # Upsample symbols
            samples_per_symbol = max(1, self.window_size // symbols_per_sample)
            phase_signal = np.repeat(phases, samples_per_symbol)[:self.window_size]
            if len(phase_signal) < self.window_size:
                phase_signal = np.pad(phase_signal, (0, self.window_size - len(phase_signal)))

            total_phase = 2 * np.pi * carrier_freq * t + phase_signal
            i_channel = np.cos(total_phase)
            q_channel = np.sin(total_phase)

        elif mod_type == "qam":
            # QAM signal
            symbol_rate = rng.uniform(1000, 50000)
            m_ary = rng.choice([4, 16, 64])
            constellation_size = int(np.sqrt(m_ary))
            symbols_per_sample = max(1, int(self.window_size * symbol_rate / self.sampling_rate))

            i_symbols = rng.randint(0, constellation_size, symbols_per_sample) * 2 - constellation_size + 1
            q_symbols = rng.randint(0, constellation_size, symbols_per_sample) * 2 - constellation_size + 1

            samples_per_symbol = max(1, self.window_size // symbols_per_sample)
            i_baseband = np.repeat(i_symbols.astype(float), samples_per_symbol)[:self.window_size]
            q_baseband = np.repeat(q_symbols.astype(float), samples_per_symbol)[:self.window_size]

            if len(i_baseband) < self.window_size:
                i_baseband = np.pad(i_baseband, (0, self.window_size - len(i_baseband)))
                q_baseband = np.pad(q_baseband, (0, self.window_size - len(q_baseband)))

            i_channel = i_baseband * np.cos(2 * np.pi * carrier_freq * t) - \
                q_baseband * np.sin(2 * np.pi * carrier_freq * t)
            q_channel = i_baseband * np.sin(2 * np.pi * carrier_freq * t) + \
                q_baseband * np.cos(2 * np.pi * carrier_freq * t)

        elif mod_type == "chirp":
            # Linear frequency chirp
            f_start = rng.uniform(0.01, 0.2) * self.sampling_rate
            f_end = rng.uniform(0.2, 0.45) * self.sampling_rate
            phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / t[-1] * t ** 2)
            i_channel = np.cos(phase)
            q_channel = np.sin(phase)

        else:  # ofdm
            # Simplified OFDM
            num_subcarriers = rng.randint(32, 128)
            subcarrier_spacing = self.sampling_rate / self.window_size
            i_channel = np.zeros(self.window_size)
            q_channel = np.zeros(self.window_size)

            for sc in range(num_subcarriers):
                sc_freq = carrier_freq + (sc - num_subcarriers // 2) * subcarrier_spacing
                sc_phase = rng.uniform(0, 2 * np.pi)
                sc_amp = rng.uniform(0.5, 1.0)
                i_channel += sc_amp * np.cos(2 * np.pi * sc_freq * t + sc_phase)
                q_channel += sc_amp * np.sin(2 * np.pi * sc_freq * t + sc_phase)

        # Add noise
        snr_db = rng.uniform(5, 30)
        signal_power = np.mean(i_channel ** 2 + q_channel ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        i_channel += rng.randn(self.window_size) * np.sqrt(noise_power / 2)
        q_channel += rng.randn(self.window_size) * np.sqrt(noise_power / 2)

        # Normalize
        max_val = max(np.max(np.abs(i_channel)), np.max(np.abs(q_channel)), 1e-10)
        i_channel /= max_val
        q_channel /= max_val

        return np.array([i_channel, q_channel], dtype=np.float32)


class SyntheticPhysicsDataset(SpectralDataset):
    """Dataset for synthetic physics simulation signals.

    Generates signals from various physical systems including:
    - Damped harmonic oscillators
    - Wave equations
    - Coupled oscillators
    - Chaotic systems (Lorenz, Duffing)
    - Fluid dynamics modes
    """

    def __init__(self, **kwargs):
        """Initialize synthetic physics dataset."""
        defaults = {
            "name": "synthetic",
            "modality": "synthetic",
            "window_size": 2048,
            "hop_size": 512,
            "sampling_rate": 1000.0,
            "num_channels": 1,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def _generate_sample(self, idx: int) -> np.ndarray:
        """Generate synthetic physics signal."""
        rng = np.random.RandomState(self.seed + idx)
        t = np.linspace(0, self.window_size / self.sampling_rate, self.window_size)
        dt = t[1] - t[0]

        # Choose physical system
        system = rng.choice([
            "damped_oscillator", "coupled_oscillators", "wave_modes",
            "duffing", "van_der_pol", "string_vibration"
        ])

        if system == "damped_oscillator":
            # Damped harmonic oscillator
            omega = rng.uniform(10, 200) * 2 * np.pi
            zeta = rng.uniform(0.001, 0.1)
            omega_d = omega * np.sqrt(1 - zeta ** 2)
            amp = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            signal = amp * np.exp(-zeta * omega * t) * np.cos(omega_d * t + phase)

        elif system == "coupled_oscillators":
            # Two coupled oscillators
            k1 = rng.uniform(1, 10)
            k2 = rng.uniform(1, 10)
            kc = rng.uniform(0.1, 2.0)  # coupling
            m1 = rng.uniform(0.5, 2.0)
            m2 = rng.uniform(0.5, 2.0)

            # Normal mode frequencies
            omega1_sq = (k1 + kc) / m1
            omega2_sq = (k2 + kc) / m2
            omega1 = np.sqrt(max(omega1_sq, 0.01)) * 2 * np.pi
            omega2 = np.sqrt(max(omega2_sq, 0.01)) * 2 * np.pi

            amp1 = rng.uniform(0.3, 1.0)
            amp2 = rng.uniform(0.3, 1.0)
            signal = amp1 * np.cos(omega1 * t) + amp2 * np.cos(omega2 * t)

            # Beat pattern
            signal *= (1 + 0.3 * np.cos((omega1 - omega2) * t / 2))

        elif system == "wave_modes":
            # Standing wave modes
            num_modes = rng.randint(3, 10)
            L = rng.uniform(1, 10)  # length
            signal = np.zeros(self.window_size)

            for n in range(1, num_modes + 1):
                mode_freq = n * rng.uniform(5, 50)
                mode_amp = rng.uniform(0.1, 1.0) / n
                decay = rng.uniform(0.1, 2.0) * n
                signal += mode_amp * np.exp(-decay * t) * np.sin(2 * np.pi * mode_freq * t)

        elif system == "duffing":
            # Duffing oscillator (nonlinear)
            alpha = rng.uniform(-1, 1)
            beta = rng.uniform(0.1, 2.0)
            delta = rng.uniform(0.1, 0.5)
            gamma = rng.uniform(0.1, 1.5)
            omega_drive = rng.uniform(0.5, 3.0)

            x = rng.uniform(-1, 1)
            v = rng.uniform(-1, 1)
            trajectory = np.zeros(self.window_size)

            for i in range(self.window_size):
                # dx/dt = v
                # dv/dt = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t[i])
                acc = -delta * v - alpha * x - beta * x ** 3 + gamma * np.cos(omega_drive * t[i])
                v += acc * dt
                x += v * dt
                trajectory[i] = x

            signal = trajectory

        elif system == "van_der_pol":
            # Van der Pol oscillator
            mu = rng.uniform(0.1, 5.0)
            omega0 = rng.uniform(1, 20) * 2 * np.pi

            x = rng.uniform(0.1, 2.0)
            v = 0.0
            trajectory = np.zeros(self.window_size)

            for i in range(self.window_size):
                acc = mu * (1 - x ** 2) * v - omega0 ** 2 * x
                v += acc * dt
                x += v * dt
                trajectory[i] = x

            signal = trajectory

        else:  # string_vibration
            # Vibrating string with multiple modes
            num_modes = rng.randint(5, 15)
            fundamental = rng.uniform(50, 500)
            inharmonicity = rng.uniform(0, 0.001)

            signal = np.zeros(self.window_size)
            for n in range(1, num_modes + 1):
                # Slightly inharmonic partials
                freq = fundamental * n * np.sqrt(1 + inharmonicity * n ** 2)
                amp = rng.uniform(0.1, 1.0) / n ** rng.uniform(0.5, 1.5)
                decay = rng.uniform(0.5, 5.0) * n ** 0.5
                phase = rng.uniform(0, 2 * np.pi)
                signal += amp * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase)

        # Normalize
        max_val = np.max(np.abs(signal)) + 1e-10
        signal = signal / max_val

        if self.num_channels > 1:
            channels = [signal]
            for ch in range(1, self.num_channels):
                # Slightly phase-shifted versions
                phase_shift = rng.uniform(0, 0.1) * self.window_size
                shift_int = int(phase_shift)
                shifted = np.roll(signal, shift_int) * rng.uniform(0.7, 1.0)
                channels.append(shifted)
            return np.array(channels, dtype=np.float32)

        return signal.astype(np.float32)


class MultiModalSpectralDataset:
    """Unified multi-modal dataset for foundation model pretraining.

    Combines all modality-specific datasets into a single interface
    with configurable sampling weights, curriculum learning, and
    cross-modal pairing for contrastive objectives.

    Attributes:
        datasets: Dictionary of modality-specific datasets.
        sampling_weights: Per-modality sampling weights.
        total_samples: Total number of available samples.
        curriculum_phase: Current curriculum learning phase.
    """

    def __init__(
        self,
        modality_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        sampling_strategy: str = "proportional",
        batch_size: int = 256,
        seed: int = 42,
        max_samples_per_modality: int = 1000,
    ):
        """Initialize multi-modal dataset.

        Args:
            modality_configs: Per-modality configurations.
            sampling_strategy: How to sample across modalities.
            batch_size: Default batch size.
            seed: Random seed.
            max_samples_per_modality: Max samples per modality.
        """
        self.sampling_strategy = sampling_strategy
        self.batch_size = batch_size
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Default modality configurations
        if modality_configs is None:
            modality_configs = {
                "seismic": {"weight": 1.0},
                "vibration": {"weight": 1.0},
                "eeg": {"weight": 1.0},
                "ecg": {"weight": 1.0},
                "audio": {"weight": 1.0},
                "rf": {"weight": 1.0},
                "synthetic": {"weight": 1.0},
            }

        # Create datasets
        self.datasets: Dict[str, SpectralDataset] = {}
        self.sampling_weights: Dict[str, float] = {}

        dataset_classes = {
            "seismic": SeismicDataset,
            "vibration": VibrationDataset,
            "eeg": EEGDataset,
            "ecg": ECGDataset,
            "audio": AudioSpectrogramDataset,
            "rf": RFSweepDataset,
            "synthetic": SyntheticPhysicsDataset,
        }

        for modality, config in modality_configs.items():
            if modality in dataset_classes:
                ds_class = dataset_classes[modality]
                self.datasets[modality] = ds_class(
                    max_samples=max_samples_per_modality,
                    seed=seed,
                )
                self.sampling_weights[modality] = config.get("weight", 1.0)

        # Normalize weights
        total_weight = sum(self.sampling_weights.values()) + 1e-10
        self.sampling_weights = {
            k: v / total_weight for k, v in self.sampling_weights.items()
        }

        # Curriculum state
        self.curriculum_phase = 0
        self.epoch = 0

    def initialize(self) -> None:
        """Initialize all datasets."""
        for ds in self.datasets.values():
            ds.initialize()

    def get_batch(
        self,
        batch_size: Optional[int] = None,
        modality: Optional[str] = None,
    ) -> List[SpectralSample]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples.
            modality: Optional specific modality (None = mixed).

        Returns:
            List of spectral samples.
        """
        bs = batch_size or self.batch_size

        if modality is not None and modality in self.datasets:
            return self.datasets[modality].get_batch(bs)

        # Sample from multiple modalities according to weights
        modalities = list(self.datasets.keys())
        weights = [self.sampling_weights[m] for m in modalities]

        batch: List[SpectralSample] = []
        for _ in range(bs):
            selected_modality = self._rng.choice(modalities, p=weights)
            sample = self.datasets[selected_modality].get_batch(1)[0]
            batch.append(sample)

        return batch

    def get_cross_modal_batch(
        self, batch_size: int = 32
    ) -> List[Tuple[SpectralSample, SpectralSample]]:
        """Get pairs of samples from different modalities.

        Args:
            batch_size: Number of pairs.

        Returns:
            List of (sample1, sample2) tuples from different modalities.
        """
        modalities = list(self.datasets.keys())
        pairs: List[Tuple[SpectralSample, SpectralSample]] = []

        for _ in range(batch_size):
            mod1, mod2 = self._rng.choice(modalities, 2, replace=False)
            sample1 = self.datasets[mod1].get_batch(1)[0]
            sample2 = self.datasets[mod2].get_batch(1)[0]
            pairs.append((sample1, sample2))

        return pairs

    def advance_curriculum(self, epoch: int) -> Dict[str, float]:
        """Advance curriculum learning phase.

        Args:
            epoch: Current epoch number.

        Returns:
            Updated sampling weights.
        """
        self.epoch = epoch

        # Simple curriculum: start with synthetic, gradually add more modalities
        if epoch < 5:
            self.curriculum_phase = 0
            active = {"synthetic": 1.0}
        elif epoch < 15:
            self.curriculum_phase = 1
            active = {"synthetic": 0.3, "seismic": 0.3, "vibration": 0.4}
        else:
            self.curriculum_phase = 2
            active = {m: 1.0 for m in self.datasets.keys()}

        # Normalize
        total = sum(active.values())
        self.sampling_weights = {
            m: active.get(m, 0.0) / total for m in self.datasets.keys()
        }

        return self.sampling_weights

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics across all modalities."""
        stats: Dict[str, Any] = {
            "num_modalities": len(self.datasets),
            "sampling_weights": self.sampling_weights,
            "curriculum_phase": self.curriculum_phase,
            "epoch": self.epoch,
            "modality_stats": {},
        }

        for name, ds in self.datasets.items():
            stats["modality_stats"][name] = ds.get_statistics()

        stats["total_samples"] = sum(
            s.get("num_samples", 0) for s in stats["modality_stats"].values()
        )

        return stats
