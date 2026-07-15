"""Time-frequency transforms (STFT / spectrogram / spectral-to-TF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record


@dataclass
class TimeFrequencyMap:
    """2D time-frequency representation (rows=frequency, cols=time/frame)."""

    matrix: np.ndarray
    frequencies_hz: np.ndarray
    times_s: np.ndarray
    method: str
    sample_rate_hz: float = 1.0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.matrix.shape


class TimeFrequencyTransform:
    """Transform signals or spectral records into time-frequency maps."""

    def __init__(
        self,
        n_fft: int = 128,
        hop_length: int = 32,
        n_freq_rows: int = 32,
        n_time_cols: int = 48,
    ) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_rows = n_freq_rows
        self.n_time_cols = n_time_cols

    def from_time_series(
        self,
        signal: np.ndarray,
        sample_rate_hz: float,
    ) -> TimeFrequencyMap:
        """STFT magnitude spectrogram from 1D time series."""
        x = np.asarray(signal, dtype=np.float64).ravel()
        n = len(x)
        if n < self.n_fft:
            x = np.pad(x, (0, self.n_fft - n))

        try:
            from scipy.signal import stft as scipy_stft

            f, t, z = scipy_stft(
                x,
                fs=sample_rate_hz,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
            )
            mag = np.abs(z)
            return TimeFrequencyMap(
                matrix=mag,
                frequencies_hz=f,
                times_s=t,
                method="scipy_stft",
                sample_rate_hz=sample_rate_hz,
            )
        except ImportError:
            return self._stft_numpy(x, sample_rate_hz)

    def _stft_numpy(self, x: np.ndarray, sample_rate_hz: float) -> TimeFrequencyMap:
        window = np.hanning(self.n_fft)
        frames = []
        times = []
        for start in range(0, len(x) - self.n_fft + 1, self.hop_length):
            seg = x[start : start + self.n_fft] * window
            spec = np.fft.rfft(seg)
            frames.append(np.abs(spec))
            times.append(start / sample_rate_hz)
        mag = np.array(frames, dtype=np.float64).T
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sample_rate_hz)
        return TimeFrequencyMap(
            matrix=mag,
            frequencies_hz=freqs,
            times_s=np.array(times, dtype=np.float64),
            method="numpy_stft",
            sample_rate_hz=sample_rate_hz,
        )

    def from_record(self, record: RecordInput) -> TimeFrequencyMap:
        """Build a pseudo time-frequency map from a frequency-domain spectral record.

        Rows: log-spaced frequency bands. Cols: sliding windows along the spectrum
        (treated as a spatial axis for salient-point detection).
        """
        rec = load_record(record)
        comp = rec.components[0]
        freq = np.asarray(comp.frequency, dtype=np.float64)
        amp = np.maximum(np.abs(np.asarray(comp.amplitude, dtype=np.float64)), 1e-12)

        order = np.argsort(freq)
        freq = freq[order]
        amp = amp[order]

        f_min, f_max = freq[0], freq[-1]
        if f_max <= f_min:
            f_max = f_min + 1.0

        band_edges = np.geomspace(f_min, f_max, self.n_freq_rows + 1)
        rows = np.zeros((self.n_freq_rows, len(freq)), dtype=np.float64)
        for i in range(self.n_freq_rows):
            mask = (freq >= band_edges[i]) & (freq < band_edges[i + 1])
            if np.any(mask):
                rows[i, mask] = amp[mask]

        win = max(len(freq) // self.n_time_cols, 4)
        cols = []
        band_centers = []
        for i in range(self.n_freq_rows):
            lo, hi = band_edges[i], band_edges[i + 1]
            band_centers.append(np.sqrt(lo * hi))
        for start in range(0, max(len(freq) - win + 1, 1), max(win // 2, 1)):
            chunk = rows[:, start : start + win]
            cols.append(np.max(chunk, axis=1) if chunk.size else np.zeros(self.n_freq_rows))
            if len(cols) >= self.n_time_cols:
                break
        while len(cols) < self.n_time_cols:
            cols.append(cols[-1] if cols else np.zeros(self.n_freq_rows))

        matrix = np.stack(cols[: self.n_time_cols], axis=1)
        times = np.linspace(0, 1.0, matrix.shape[1])
        return TimeFrequencyMap(
            matrix=matrix,
            frequencies_hz=np.array(band_centers, dtype=np.float64),
            times_s=times,
            method="spectral_pseudo_tf",
            sample_rate_hz=1.0 / max(times[-1] - times[0], 1e-9),
        )

    def synthetic_signal_from_record(
        self,
        record: RecordInput,
        *,
        duration_s: float = 2.0,
        sample_rate_hz: float = 256.0,
    ) -> TimeFrequencyMap:
        """Synthesize narrowband signal from spectrum peaks, then STFT."""
        rec = load_record(record)
        comp = rec.components[0]
        freq = np.asarray(comp.frequency, dtype=np.float64)
        amp = np.maximum(np.abs(np.asarray(comp.amplitude, dtype=np.float64)), 0.0)
        n = int(duration_s * sample_rate_hz)
        t = np.arange(n, dtype=np.float64) / sample_rate_hz
        x = np.zeros(n, dtype=np.float64)
        top = np.argsort(amp)[-min(16, len(amp)) :]
        for idx in top:
            x += amp[idx] * np.sin(2 * np.pi * freq[idx] * t)
        x /= max(np.max(np.abs(x)), 1e-12)
        return self.from_time_series(x, sample_rate_hz)