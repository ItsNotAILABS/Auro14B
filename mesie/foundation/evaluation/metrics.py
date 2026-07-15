"""Evaluation metrics for spectral foundation models.

Comprehensive metrics for assessing quality of spectral
representations and reconstructions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SpectralMetrics:
    """Core spectral evaluation metrics.

    Provides standard metrics for comparing spectral signals
    including both time-domain and frequency-domain measures.

    Attributes:
        sample_rate: Signal sampling rate.
    """

    def __init__(self, sample_rate: float = 1.0):
        """Initialize spectral metrics.

        Args:
            sample_rate: Sampling rate.
        """
        self.sample_rate = sample_rate

    def signal_to_noise_ratio(
        self, signal: np.ndarray, noise: np.ndarray
    ) -> float:
        """Compute SNR in dB.

        Args:
            signal: Clean signal.
            noise: Noise component.

        Returns:
            SNR in dB.
        """
        signal_power = np.mean(signal ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        return float(10 * np.log10(signal_power / noise_power))

    def signal_to_distortion_ratio(
        self, reference: np.ndarray, estimate: np.ndarray
    ) -> float:
        """Compute SDR (Signal-to-Distortion Ratio).

        Standard metric for source separation quality.

        Args:
            reference: Reference signal.
            estimate: Estimated signal.

        Returns:
            SDR in dB.
        """
        # Project estimate onto reference
        ref_flat = reference.flatten()
        est_flat = estimate.flatten()

        # s_target = <ref, est> / <ref, ref> * ref
        alpha = np.dot(ref_flat, est_flat) / (np.dot(ref_flat, ref_flat) + 1e-10)
        s_target = alpha * ref_flat

        # e_noise = est - s_target
        e_noise = est_flat - s_target

        # SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        sdr = 10 * np.log10(
            (np.sum(s_target ** 2) + 1e-10) /
            (np.sum(e_noise ** 2) + 1e-10)
        )
        return float(sdr)

    def log_spectral_distance(
        self, reference: np.ndarray, estimate: np.ndarray
    ) -> float:
        """Compute Log-Spectral Distance.

        Standard measure of spectral envelope difference.

        Args:
            reference: Reference signal.
            estimate: Estimated signal.

        Returns:
            LSD value.
        """
        ref_fft = np.fft.rfft(reference.flatten())
        est_fft = np.fft.rfft(estimate.flatten())

        ref_mag = np.abs(ref_fft) + 1e-10
        est_mag = np.abs(est_fft) + 1e-10

        log_diff = (20 * np.log10(ref_mag) - 20 * np.log10(est_mag)) ** 2
        lsd = float(np.sqrt(np.mean(log_diff)))

        return lsd

    def itakura_saito_distance(
        self, reference: np.ndarray, estimate: np.ndarray
    ) -> float:
        """Compute Itakura-Saito divergence.

        Perceptually motivated spectral distance measure.

        Args:
            reference: Reference spectrum.
            estimate: Estimated spectrum.

        Returns:
            IS distance.
        """
        ref_psd = np.abs(np.fft.rfft(reference.flatten())) ** 2 + 1e-10
        est_psd = np.abs(np.fft.rfft(estimate.flatten())) ** 2 + 1e-10

        ratio = ref_psd / est_psd
        is_dist = float(np.mean(ratio - np.log(ratio) - 1))
        return is_dist

    def spectral_convergence(
        self, reference: np.ndarray, estimate: np.ndarray
    ) -> float:
        """Compute spectral convergence.

        Args:
            reference: Reference signal.
            estimate: Estimated signal.

        Returns:
            Spectral convergence (lower is better).
        """
        ref_mag = np.abs(np.fft.rfft(reference.flatten()))
        est_mag = np.abs(np.fft.rfft(estimate.flatten()))

        sc = float(
            np.linalg.norm(ref_mag - est_mag) /
            (np.linalg.norm(ref_mag) + 1e-10)
        )
        return sc

    def frequency_weighted_snr(
        self, reference: np.ndarray, estimate: np.ndarray,
        num_bands: int = 8
    ) -> Dict[str, float]:
        """Compute frequency-weighted SNR per band.

        Args:
            reference: Reference signal.
            estimate: Estimated signal.
            num_bands: Number of frequency bands.

        Returns:
            Per-band SNR values.
        """
        ref_fft = np.fft.rfft(reference.flatten())
        est_fft = np.fft.rfft(estimate.flatten())
        n_freqs = len(ref_fft)

        band_size = n_freqs // num_bands
        results: Dict[str, float] = {}

        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else n_freqs

            ref_band = ref_fft[start:end]
            est_band = est_fft[start:end]

            ref_power = np.sum(np.abs(ref_band) ** 2) + 1e-10
            error_power = np.sum(np.abs(ref_band - est_band) ** 2) + 1e-10

            band_snr = 10 * np.log10(ref_power / error_power)
            results[f"band_{i}_snr_db"] = float(band_snr)

        results["mean_band_snr_db"] = float(np.mean(list(results.values())))
        return results


class ReconstructionMetrics:
    """Metrics for evaluating signal reconstruction quality.

    Attributes:
        spectral_metrics: SpectralMetrics instance.
    """

    def __init__(self, sample_rate: float = 1.0):
        self.spectral_metrics = SpectralMetrics(sample_rate)

    def evaluate(
        self, reference: np.ndarray, reconstruction: np.ndarray
    ) -> Dict[str, float]:
        """Compute all reconstruction metrics.

        Args:
            reference: Original signal.
            reconstruction: Reconstructed signal.

        Returns:
            Dictionary of all metrics.
        """
        error = reference - reconstruction

        metrics = {
            "mse": float(np.mean(error ** 2)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "snr_db": self.spectral_metrics.signal_to_noise_ratio(reference, error),
            "sdr_db": self.spectral_metrics.signal_to_distortion_ratio(
                reference, reconstruction
            ),
            "lsd": self.spectral_metrics.log_spectral_distance(
                reference, reconstruction
            ),
            "spectral_convergence": self.spectral_metrics.spectral_convergence(
                reference, reconstruction
            ),
            "correlation": float(np.corrcoef(
                reference.flatten(), reconstruction.flatten()
            )[0, 1]) if reference.size > 1 else 0.0,
        }

        # Envelope correlation
        ref_env = np.abs(self._hilbert(reference.flatten()))
        rec_env = np.abs(self._hilbert(reconstruction.flatten()))
        if len(ref_env) > 1:
            metrics["envelope_correlation"] = float(
                np.corrcoef(ref_env, rec_env)[0, 1]
            )

        return metrics

    def _hilbert(self, x: np.ndarray) -> np.ndarray:
        """Compute analytic signal via Hilbert transform."""
        n = len(x)
        fft = np.fft.fft(x)
        h = np.zeros(n)
        if n > 0:
            h[0] = 1
            if n % 2 == 0:
                h[n//2] = 1
                h[1:n//2] = 2
            else:
                h[1:(n+1)//2] = 2
        return np.fft.ifft(fft * h)


class RepresentationMetrics:
    """Metrics for evaluating learned representations.

    Assesses the quality of the latent space including
    alignment, uniformity, and downstream utility.
    """

    def __init__(self, latent_dim: int = 768):
        self.latent_dim = latent_dim

    def alignment(
        self, z_a: np.ndarray, z_b: np.ndarray
    ) -> float:
        """Compute alignment of positive pairs.

        Lower is better (positive pairs should be close).

        Args:
            z_a: First representations [B, D].
            z_b: Paired representations [B, D].

        Returns:
            Alignment score.
        """
        z_a_norm = z_a / (np.linalg.norm(z_a, axis=-1, keepdims=True) + 1e-8)
        z_b_norm = z_b / (np.linalg.norm(z_b, axis=-1, keepdims=True) + 1e-8)
        return float(np.mean(np.sum((z_a_norm - z_b_norm) ** 2, axis=-1)))

    def uniformity(self, z: np.ndarray, t: float = 2.0) -> float:
        """Compute uniformity of representation distribution.

        Lower is better (points spread uniformly on hypersphere).

        Args:
            z: Representations [B, D].
            t: Temperature.

        Returns:
            Uniformity score.
        """
        z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        sq_dists = np.sum(
            (z_norm[:, None] - z_norm[None, :]) ** 2, axis=-1
        )
        return float(np.log(np.mean(np.exp(-t * sq_dists)) + 1e-10))

    def isotropy(self, z: np.ndarray) -> float:
        """Compute isotropy of representation space.

        Higher is better (representations use all dimensions).

        Args:
            z: Representations [B, D].

        Returns:
            Isotropy score (0 = anisotropic, 1 = isotropic).
        """
        # Compute singular values
        z_centered = z - np.mean(z, axis=0)
        if z_centered.shape[0] < z_centered.shape[1]:
            cov = np.dot(z_centered, z_centered.T) / z_centered.shape[0]
            eigenvalues = np.linalg.eigvalsh(cov)
        else:
            cov = np.dot(z_centered.T, z_centered) / z_centered.shape[0]
            eigenvalues = np.linalg.eigvalsh(cov)

        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-10)

        # Isotropy = exponential of entropy / max entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        max_entropy = np.log(len(eigenvalues))

        return float(np.exp(entropy - max_entropy))

    def intrinsic_dimensionality(self, z: np.ndarray) -> float:
        """Estimate intrinsic dimensionality.

        Args:
            z: Representations [B, D].

        Returns:
            Estimated intrinsic dimension.
        """
        z_centered = z - np.mean(z, axis=0)
        n = min(z_centered.shape)
        cov = np.dot(z_centered.T, z_centered) / z_centered.shape[0]
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 0)

        # Participation ratio
        total = np.sum(eigenvalues) + 1e-10
        participation_ratio = total ** 2 / (np.sum(eigenvalues ** 2) + 1e-10)

        return float(participation_ratio)

    def modality_separability(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Measure separability between modalities.

        Args:
            embeddings: Per-modality embeddings.

        Returns:
            Separability metrics.
        """
        modalities = list(embeddings.keys())
        metrics: Dict[str, float] = {}

        # Inter-modality distances
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i+1:]:
                z1 = embeddings[m1]
                z2 = embeddings[m2]

                # Normalize
                z1_norm = z1 / (np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
                z2_norm = z2 / (np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)

                # Mean cosine similarity
                cross_sim = np.mean(np.dot(z1_norm, z2_norm.T))
                metrics[f"{m1}-{m2}_similarity"] = float(cross_sim)

        # Within-modality cohesion
        for m, z in embeddings.items():
            z_norm = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            within_sim = np.mean(np.dot(z_norm, z_norm.T))
            metrics[f"{m}_cohesion"] = float(within_sim)

        return metrics

    def evaluate_all(
        self,
        embeddings: np.ndarray,
        positive_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Run all representation metrics.

        Args:
            embeddings: Representations [B, D].
            positive_pairs: Optional (z_a, z_b) positive pairs.

        Returns:
            All metrics.
        """
        metrics = {
            "uniformity": self.uniformity(embeddings),
            "isotropy": self.isotropy(embeddings),
            "intrinsic_dim": self.intrinsic_dimensionality(embeddings),
        }

        if positive_pairs is not None:
            metrics["alignment"] = self.alignment(*positive_pairs)

        return metrics


class DownstreamMetrics:
    """Metrics for downstream task evaluation.

    Assesses how well pretrained representations transfer
    to downstream tasks.
    """

    def accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(predictions == labels))

    def f1_score(
        self, predictions: np.ndarray, labels: np.ndarray, average: str = "macro"
    ) -> float:
        """Compute F1 score.

        Args:
            predictions: Predicted labels.
            labels: True labels.
            average: Averaging method.

        Returns:
            F1 score.
        """
        classes = np.unique(np.concatenate([predictions, labels]))
        f1_scores = []

        for cls in classes:
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)

        if average == "macro":
            return float(np.mean(f1_scores))
        elif average == "micro":
            tp_total = np.sum(predictions == labels)
            return float(tp_total / len(labels))
        return float(np.mean(f1_scores))

    def mean_average_precision(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Compute mean average precision for retrieval.

        Args:
            scores: Similarity scores [N, N].
            labels: Class labels [N].

        Returns:
            mAP score.
        """
        n = len(labels)
        aps = []

        for i in range(n):
            # Sort by similarity (descending)
            sorted_indices = np.argsort(scores[i])[::-1]
            # Remove self
            sorted_indices = sorted_indices[sorted_indices != i]

            # Compute AP
            relevant = (labels[sorted_indices] == labels[i])
            if not np.any(relevant):
                continue

            precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = np.sum(precision_at_k * relevant) / (np.sum(relevant) + 1e-10)
            aps.append(ap)

        return float(np.mean(aps)) if aps else 0.0
