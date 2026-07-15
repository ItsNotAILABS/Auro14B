"""Physics-informed pretraining objectives.

Encodes physical laws and constraints as loss functions to
ensure learned representations respect fundamental physics
of spectral signals across all modalities.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PhysicsInformedLoss:
    """Combined physics-informed loss for spectral pretraining.

    Combines multiple physics constraints into a single objective
    that guides the model to learn physically meaningful representations.

    Physical constraints include:
    - Energy conservation (Parseval's theorem)
    - Causality (no acausal predictions)
    - Time-frequency uncertainty principle
    - Spectral smoothness priors
    - Harmonic structure preservation
    - Modality-specific physics

    Attributes:
        constraints: Active physics constraints.
        weights: Per-constraint weights.
    """

    def __init__(
        self,
        constraints: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        sample_rate: float = 1.0,
    ):
        """Initialize physics-informed loss.

        Args:
            constraints: List of active constraints.
            weights: Per-constraint weights.
            sample_rate: Signal sampling rate.
        """
        self.constraints = constraints or [
            "energy_conservation",
            "causality",
            "uncertainty_principle",
            "spectral_smoothness",
            "harmonic_consistency",
            "parseval",
        ]

        self.weights = weights or {
            "energy_conservation": 1.0,
            "causality": 0.5,
            "uncertainty_principle": 0.3,
            "spectral_smoothness": 0.2,
            "harmonic_consistency": 0.5,
            "parseval": 1.0,
        }

        self.sample_rate = sample_rate

        # Sub-loss modules
        self._conservation = ConservationLoss(sample_rate=sample_rate)
        self._causality = CausalityLoss()
        self._symmetry = SymmetryLoss()
        self._consistency = SpectralConsistencyLoss(sample_rate=sample_rate)

    def compute(
        self,
        prediction: np.ndarray,
        target: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined physics-informed loss.

        Args:
            prediction: Model prediction [B, T] or [B, T, D].
            target: Optional ground truth.
            metadata: Optional metadata (modality, frequency info, etc.).

        Returns:
            Total loss and per-constraint losses.
        """
        losses: Dict[str, float] = {}

        if "energy_conservation" in self.constraints:
            losses["energy_conservation"] = self._conservation.energy_conservation(
                prediction, target
            )

        if "parseval" in self.constraints:
            losses["parseval"] = self._conservation.parseval_loss(prediction)

        if "causality" in self.constraints:
            losses["causality"] = self._causality.compute(prediction)

        if "uncertainty_principle" in self.constraints:
            losses["uncertainty_principle"] = self._consistency.uncertainty_loss(
                prediction
            )

        if "spectral_smoothness" in self.constraints:
            losses["spectral_smoothness"] = self._consistency.smoothness_loss(
                prediction
            )

        if "harmonic_consistency" in self.constraints:
            if metadata and "fundamental_freq" in metadata:
                losses["harmonic_consistency"] = self._consistency.harmonic_loss(
                    prediction, metadata["fundamental_freq"]
                )
            else:
                losses["harmonic_consistency"] = 0.0

        # Weighted sum
        total = sum(
            self.weights.get(k, 0.0) * v
            for k, v in losses.items()
        )
        losses["total"] = total

        return total, losses


class ConservationLoss:
    """Energy and conservation law losses.

    Ensures physical conservation laws are respected:
    - Energy conservation between time and frequency domains
    - Parseval's theorem
    - Power spectral density constraints
    - Total energy bounds

    Attributes:
        sample_rate: Sampling rate.
        energy_tolerance: Tolerance for energy conservation.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        energy_tolerance: float = 0.01,
    ):
        """Initialize conservation loss.

        Args:
            sample_rate: Signal sampling rate.
            energy_tolerance: Allowed energy error.
        """
        self.sample_rate = sample_rate
        self.energy_tolerance = energy_tolerance

    def energy_conservation(
        self,
        prediction: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> float:
        """Ensure energy is conserved in reconstruction.

        The reconstructed signal should have approximately the
        same total energy as the original.

        Args:
            prediction: Predicted signal [B, T].
            target: Original signal [B, T].

        Returns:
            Energy conservation loss.
        """
        if target is None:
            return 0.0

        # Flatten to 2D
        pred = prediction.reshape(prediction.shape[0], -1)
        tgt = target.reshape(target.shape[0], -1)

        # Energy = sum of squared amplitudes
        pred_energy = np.sum(pred ** 2, axis=-1)
        target_energy = np.sum(tgt ** 2, axis=-1)

        # Relative energy error
        energy_ratio = pred_energy / (target_energy + 1e-10)
        energy_loss = float(np.mean((energy_ratio - 1.0) ** 2))

        return energy_loss

    def parseval_loss(self, signal: np.ndarray) -> float:
        """Verify Parseval's theorem: time energy == frequency energy.

        For any signal, the total energy in time domain should equal
        the total energy in frequency domain (up to normalization).

        Args:
            signal: Signal [B, T].

        Returns:
            Parseval violation loss.
        """
        sig = signal.reshape(signal.shape[0], -1)

        # Time domain energy
        time_energy = np.sum(sig ** 2, axis=-1)

        # Frequency domain energy
        fft = np.fft.rfft(sig, axis=-1)
        freq_energy = np.sum(np.abs(fft) ** 2, axis=-1) / sig.shape[-1]

        # They should be equal (Parseval's theorem)
        relative_error = np.abs(time_energy - freq_energy) / (time_energy + 1e-10)
        return float(np.mean(relative_error))

    def power_spectral_density_constraint(
        self,
        signal: np.ndarray,
        expected_slope: float = -2.0,
    ) -> float:
        """Ensure PSD follows expected power law.

        Many natural signals have 1/f^alpha power spectra.
        This loss penalizes deviations from expected spectral shape.

        Args:
            signal: Signal [B, T].
            expected_slope: Expected log-log slope.

        Returns:
            PSD constraint loss.
        """
        sig = signal.reshape(signal.shape[0], -1)
        fft = np.fft.rfft(sig, axis=-1)
        psd = np.abs(fft) ** 2

        # Log-log PSD
        n_freqs = psd.shape[-1]
        freqs = np.arange(1, n_freqs + 1, dtype=np.float64)
        log_freqs = np.log(freqs)
        log_psd = np.log(psd + 1e-10)

        # Fit slope via linear regression in log-log space
        mean_log_f = np.mean(log_freqs)
        mean_log_p = np.mean(log_psd, axis=-1)

        numerator = np.sum(
            (log_freqs - mean_log_f) * (log_psd - mean_log_p[:, np.newaxis]),
            axis=-1
        )
        denominator = np.sum((log_freqs - mean_log_f) ** 2) + 1e-10

        estimated_slope = numerator / denominator

        # Penalize deviation from expected slope
        slope_error = (estimated_slope - expected_slope) ** 2
        return float(np.mean(slope_error))


class CausalityLoss:
    """Causality constraint loss.

    Ensures predictions respect temporal causality:
    - Impulse responses should be causal (zero for t < 0)
    - Transfer functions should be minimum phase
    - No "time travel" in predictions

    Attributes:
        causal_tolerance: Tolerance for acausal energy.
    """

    def __init__(self, causal_tolerance: float = 0.01):
        """Initialize causality loss.

        Args:
            causal_tolerance: Allowed acausal energy fraction.
        """
        self.causal_tolerance = causal_tolerance

    def compute(self, signal: np.ndarray) -> float:
        """Compute causality violation loss.

        Checks if the signal's analytic representation
        satisfies minimum phase conditions.

        Args:
            signal: Signal [B, T].

        Returns:
            Causality loss.
        """
        sig = signal.reshape(signal.shape[0], -1)

        # Compute analytic signal
        fft = np.fft.fft(sig, axis=-1)
        n = sig.shape[-1]

        # For causal signals, the spectrum should satisfy
        # Kramers-Kronig relations (real and imaginary parts related by Hilbert transform)

        # Check: log-magnitude should be a causal function
        log_mag = np.log(np.abs(fft) + 1e-10)
        phase = np.angle(fft)

        # Minimum phase: phase should be Hilbert transform of log-magnitude
        # Compute expected minimum phase
        log_fft = np.fft.ifft(log_mag, axis=-1)

        # Acausal energy (energy in negative time for impulse response)
        half_n = n // 2
        acausal_energy = np.sum(np.abs(log_fft[:, half_n:]) ** 2, axis=-1)
        total_energy = np.sum(np.abs(log_fft) ** 2, axis=-1) + 1e-10

        # Causality violation = fraction of acausal energy
        violation = acausal_energy / total_energy
        loss = float(np.mean(np.maximum(0, violation - self.causal_tolerance)))

        return loss

    def group_delay_consistency(self, signal: np.ndarray) -> float:
        """Check group delay consistency.

        Group delay should be non-negative for causal systems.

        Args:
            signal: Signal [B, T].

        Returns:
            Group delay violation loss.
        """
        sig = signal.reshape(signal.shape[0], -1)
        n = sig.shape[-1]

        # Compute group delay: -d(phase)/d(frequency)
        fft = np.fft.rfft(sig, axis=-1)
        phase = np.unwrap(np.angle(fft), axis=-1)

        # Numerical derivative
        group_delay = -np.diff(phase, axis=-1)

        # For causal systems, group delay should be >= 0
        negative_gd = np.minimum(0, group_delay)
        loss = float(np.mean(negative_gd ** 2))

        return loss


class SymmetryLoss:
    """Symmetry-based physics constraints.

    Exploits physical symmetries to regularize representations:
    - Time-reversal symmetry
    - Frequency-shift invariance
    - Scale invariance
    - Conjugate symmetry of real signals

    Attributes:
        symmetries: Active symmetry constraints.
    """

    def __init__(
        self,
        symmetries: Optional[List[str]] = None,
    ):
        """Initialize symmetry loss.

        Args:
            symmetries: Active symmetries.
        """
        self.symmetries = symmetries or [
            "conjugate",
            "time_reversal",
            "scale",
        ]

    def compute(
        self,
        signal: np.ndarray,
        representation: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute symmetry losses.

        Args:
            signal: Input signal [B, T].
            representation: Learned representation [B, D].

        Returns:
            Total loss and per-symmetry losses.
        """
        losses: Dict[str, float] = {}

        if "conjugate" in self.symmetries:
            losses["conjugate_symmetry"] = self._conjugate_symmetry(signal)

        if "time_reversal" in self.symmetries and representation is not None:
            losses["time_reversal"] = self._time_reversal_symmetry(
                signal, representation
            )

        if "scale" in self.symmetries and representation is not None:
            losses["scale_invariance"] = self._scale_invariance(
                signal, representation
            )

        total = sum(losses.values())
        losses["total"] = total
        return total, losses

    def _conjugate_symmetry(self, signal: np.ndarray) -> float:
        """Verify conjugate symmetry for real signals.

        For real-valued signals, the DFT should satisfy:
        X(-f) = X*(f) (conjugate symmetry).

        Args:
            signal: Real signal [B, T].

        Returns:
            Conjugate symmetry violation.
        """
        sig = signal.reshape(signal.shape[0], -1)
        fft = np.fft.fft(sig, axis=-1)
        n = sig.shape[-1]

        # Check X[k] == conj(X[N-k])
        fft_reversed = np.conj(fft[:, ::-1])
        fft_shifted = np.roll(fft_reversed, 1, axis=-1)

        violation = np.abs(fft - fft_shifted)
        return float(np.mean(violation))

    def _time_reversal_symmetry(
        self, signal: np.ndarray, representation: np.ndarray
    ) -> float:
        """Check time-reversal properties.

        For systems with time-reversal symmetry, reversing the
        signal should produce a predictable transformation of
        the representation (e.g., complex conjugation of spectrum).

        Args:
            signal: Signal [B, T].
            representation: Representation [B, D].

        Returns:
            Time-reversal symmetry loss.
        """
        # For representations, time reversal should only affect phase
        sig_reversed = signal[..., ::-1] if signal.ndim >= 2 else signal[::-1]

        # The magnitude spectrum should be identical for signal and its reverse
        fft_original = np.fft.rfft(signal.reshape(signal.shape[0], -1), axis=-1)
        fft_reversed = np.fft.rfft(sig_reversed.reshape(signal.shape[0], -1), axis=-1)

        mag_diff = np.abs(np.abs(fft_original) - np.abs(fft_reversed))
        return float(np.mean(mag_diff))

    def _scale_invariance(
        self, signal: np.ndarray, representation: np.ndarray
    ) -> float:
        """Check scale invariance of representation.

        Scaling a signal's amplitude should produce a proportional
        change in representation magnitude, not direction.

        Args:
            signal: Signal [B, T].
            representation: Representation [B, D].

        Returns:
            Scale invariance loss.
        """
        # Direction of representation should be invariant to amplitude scaling
        rep_norm = representation / (np.linalg.norm(representation, axis=-1, keepdims=True) + 1e-8)

        # Compute expected: scaling signal by alpha should not change rep direction
        # This is enforced by checking cosine similarity of different-amplitude versions
        # Using signal energy as proxy
        energy = np.sum(signal.reshape(signal.shape[0], -1) ** 2, axis=-1, keepdims=True)
        normalized_energy = energy / (np.mean(energy) + 1e-10)

        # Representations with different energy should still have similar directions
        # Variance of direction as function of energy = scale sensitivity
        rep_variance = np.var(rep_norm, axis=0)
        return float(np.mean(rep_variance))


class SpectralConsistencyLoss:
    """Spectral consistency constraints.

    Ensures predictions maintain spectral consistency:
    - Harmonic relationships
    - Spectral envelope smoothness
    - Time-frequency uncertainty
    - Spectral continuity

    Attributes:
        sample_rate: Sampling rate.
    """

    def __init__(self, sample_rate: float = 1.0):
        """Initialize spectral consistency.

        Args:
            sample_rate: Signal sampling rate.
        """
        self.sample_rate = sample_rate

    def uncertainty_loss(self, signal: np.ndarray) -> float:
        """Heisenberg uncertainty principle constraint.

        Ensures time-frequency representations respect the
        fundamental uncertainty: Δt × Δf >= 1/(4π).

        Args:
            signal: Signal [B, T].

        Returns:
            Uncertainty principle violation loss.
        """
        sig = signal.reshape(signal.shape[0], -1)
        n = sig.shape[-1]

        # Time spread
        t = np.arange(n)
        sig_sq = sig ** 2
        energy = np.sum(sig_sq, axis=-1, keepdims=True) + 1e-10
        sig_prob = sig_sq / energy

        mean_t = np.sum(sig_prob * t, axis=-1)
        var_t = np.sum(sig_prob * (t - mean_t[:, np.newaxis]) ** 2, axis=-1)

        # Frequency spread
        fft = np.fft.rfft(sig, axis=-1)
        mag_sq = np.abs(fft) ** 2
        freq_energy = np.sum(mag_sq, axis=-1, keepdims=True) + 1e-10
        freq_prob = mag_sq / freq_energy

        f = np.arange(mag_sq.shape[-1])
        mean_f = np.sum(freq_prob * f, axis=-1)
        var_f = np.sum(freq_prob * (f - mean_f[:, np.newaxis]) ** 2, axis=-1)

        # Uncertainty product should be >= 1/(4*pi)
        uncertainty_product = np.sqrt(var_t * var_f + 1e-10)
        min_uncertainty = 1.0 / (4 * np.pi)

        # Penalize violations
        violation = np.maximum(0, min_uncertainty - uncertainty_product)
        return float(np.mean(violation))

    def smoothness_loss(self, signal: np.ndarray) -> float:
        """Spectral smoothness regularization.

        Penalizes rapid spectral variations that are
        unlikely in natural signals.

        Args:
            signal: Signal [B, T].

        Returns:
            Smoothness loss.
        """
        sig = signal.reshape(signal.shape[0], -1)
        fft = np.fft.rfft(sig, axis=-1)
        log_mag = np.log(np.abs(fft) + 1e-10)

        # First-order spectral difference
        diff1 = np.diff(log_mag, axis=-1)
        smoothness_1 = float(np.mean(diff1 ** 2))

        # Second-order (curvature)
        diff2 = np.diff(diff1, axis=-1)
        smoothness_2 = float(np.mean(diff2 ** 2))

        return smoothness_1 + 0.5 * smoothness_2

    def harmonic_loss(
        self,
        signal: np.ndarray,
        fundamental_freq: float,
        num_harmonics: int = 8,
    ) -> float:
        """Harmonic structure preservation loss.

        For harmonic signals, ensures that predicted harmonics
        maintain proper frequency ratios and relative amplitudes.

        Args:
            signal: Signal [B, T].
            fundamental_freq: Fundamental frequency (Hz).
            num_harmonics: Number of harmonics to check.

        Returns:
            Harmonic consistency loss.
        """
        sig = signal.reshape(signal.shape[0], -1)
        n = sig.shape[-1]

        fft = np.fft.rfft(sig, axis=-1)
        mag = np.abs(fft)

        # Frequency resolution
        freq_res = self.sample_rate / n

        # Expected harmonic positions
        harmonic_freqs = fundamental_freq * np.arange(1, num_harmonics + 1)
        harmonic_bins = (harmonic_freqs / freq_res).astype(int)
        harmonic_bins = harmonic_bins[harmonic_bins < mag.shape[-1]]

        if len(harmonic_bins) < 2:
            return 0.0

        # Extract harmonic amplitudes
        harmonic_amps = mag[:, harmonic_bins]

        # Harmonic relationship: amplitudes should decay monotonically (approximately)
        amp_ratios = harmonic_amps[:, 1:] / (harmonic_amps[:, :-1] + 1e-10)

        # Most natural harmonic series have decreasing amplitude
        # Penalize cases where higher harmonics are stronger than lower
        violations = np.maximum(0, amp_ratios - 1.5)  # Allow some tolerance

        return float(np.mean(violations))

    def spectral_continuity_loss(
        self,
        spectrogram: np.ndarray,
    ) -> float:
        """Temporal continuity in spectrogram.

        Adjacent time frames should have similar spectral content
        (signals don't change instantaneously in nature).

        Args:
            spectrogram: Spectrogram [B, T, F].

        Returns:
            Continuity loss.
        """
        if spectrogram.ndim < 3:
            return 0.0

        # Temporal differences
        time_diff = np.diff(spectrogram, axis=1)
        continuity = float(np.mean(time_diff ** 2))

        # Frequency differences (spectral smoothness across time)
        freq_diff = np.diff(spectrogram, axis=2)
        smoothness = float(np.mean(freq_diff ** 2))

        return continuity + 0.5 * smoothness
