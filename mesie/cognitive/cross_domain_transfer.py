"""
Cross-Domain Spectral Transfer System.

Implements multi-domain spectral corpora management and cross-domain transfer
learning using CORAL, MMD, and domain-invariant normalization. Enables
foundation-model-style generalization across wildly different spectral domains
(earthquake → bridge vibration, EEG → audio resonance, etc.).
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class SpectralDomain(Enum):
    """Known spectral domains for cross-domain transfer."""
    SEISMIC = "seismic"
    STRUCTURAL_VIBRATION = "structural_vibration"
    EEG_NEURAL = "eeg_neural"
    AUDIO_ACOUSTIC = "audio_acoustic"
    ELECTROMAGNETIC = "electromagnetic"
    OPTICAL_SPECTROSCOPY = "optical_spectroscopy"
    MEDICAL_IMAGING = "medical_imaging"
    FINANCIAL_TIMESERIES = "financial_timeseries"
    CLIMATE_ATMOSPHERIC = "climate_atmospheric"
    CHEMICAL_SPECTRAL = "chemical_spectral"
    RADIO_FREQUENCY = "radio_frequency"
    GRAVITATIONAL_WAVE = "gravitational_wave"
    GENERIC = "generic"


@dataclass
class DomainDescriptor:
    """Describes a spectral domain's characteristics."""
    domain: SpectralDomain
    name: str
    frequency_range: tuple = (0.0, 1.0)  # Hz or normalized
    typical_snr: float = 20.0  # dB
    n_channels: int = 1
    sampling_rate: float = 1.0
    key_features: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def bandwidth(self) -> float:
        return self.frequency_range[1] - self.frequency_range[0]

    def compute_similarity(self, other: "DomainDescriptor") -> float:
        """Compute similarity between two domain descriptors."""
        # Frequency overlap
        overlap_low = max(self.frequency_range[0], other.frequency_range[0])
        overlap_high = min(self.frequency_range[1], other.frequency_range[1])
        freq_overlap = max(0, overlap_high - overlap_low) / max(self.bandwidth, other.bandwidth, 1e-10)

        # SNR similarity
        snr_sim = 1.0 / (1.0 + abs(self.typical_snr - other.typical_snr) / 10.0)

        # Feature overlap
        if self.key_features and other.key_features:
            common = set(self.key_features) & set(other.key_features)
            total = set(self.key_features) | set(other.key_features)
            feat_sim = len(common) / max(len(total), 1)
        else:
            feat_sim = 0.5

        return 0.4 * freq_overlap + 0.3 * snr_sim + 0.3 * feat_sim


@dataclass
class SpectralCorpus:
    """A collection of spectral data from a single domain."""
    domain: SpectralDomain
    descriptor: DomainDescriptor
    data: np.ndarray  # (n_samples, n_features)
    labels: Optional[np.ndarray] = None
    sample_ids: Optional[list] = None
    statistics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.statistics:
            self._compute_statistics()

    def _compute_statistics(self) -> None:
        """Compute corpus statistics for normalization."""
        self.statistics = {
            "mean": np.mean(self.data, axis=0),
            "std": np.std(self.data, axis=0) + 1e-8,
            "min": np.min(self.data, axis=0),
            "max": np.max(self.data, axis=0),
            "n_samples": self.data.shape[0],
            "n_features": self.data.shape[1],
            "spectral_centroid": self._spectral_centroid(),
            "spectral_bandwidth": self._spectral_bandwidth(),
            "energy_distribution": self._energy_distribution(),
        }

    def _spectral_centroid(self) -> float:
        """Compute mean spectral centroid of the corpus."""
        freqs = np.arange(self.data.shape[1])
        magnitudes = np.abs(self.data)
        centroids = np.sum(freqs * magnitudes, axis=1) / (np.sum(magnitudes, axis=1) + 1e-10)
        return float(np.mean(centroids))

    def _spectral_bandwidth(self) -> float:
        """Compute mean spectral bandwidth."""
        freqs = np.arange(self.data.shape[1])
        magnitudes = np.abs(self.data)
        centroids = np.sum(freqs * magnitudes, axis=1) / (np.sum(magnitudes, axis=1) + 1e-10)
        bandwidths = np.sqrt(
            np.sum(magnitudes * (freqs - centroids[:, None]) ** 2, axis=1)
            / (np.sum(magnitudes, axis=1) + 1e-10)
        )
        return float(np.mean(bandwidths))

    def _energy_distribution(self) -> np.ndarray:
        """Compute energy distribution across frequency bands."""
        n_bands = min(16, self.data.shape[1])
        band_size = self.data.shape[1] // n_bands
        energy = np.zeros(n_bands)
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            energy[i] = np.mean(self.data[:, start:end] ** 2)
        return energy / (np.sum(energy) + 1e-10)

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]


class DomainInvariantNormalizer:
    """
    Domain-invariant normalization for cross-domain transfer.

    Normalizes spectra from different domains into a shared representation
    space where domain-specific characteristics are factored out while
    preserving discriminative spectral structure.
    """

    def __init__(self, target_dim: int = 128, method: str = "whitening"):
        self.target_dim = target_dim
        self.method = method
        self._domain_transforms: dict[str, dict] = {}
        self._global_statistics: Optional[dict] = None
        self._is_fitted = False

    def fit_domain(self, corpus: SpectralCorpus) -> None:
        """Fit normalizer for a specific domain."""
        domain_key = corpus.domain.value
        data = corpus.data

        # Compute domain-specific whitening transform
        mean = np.mean(data, axis=0)
        centered = data - mean
        cov = np.dot(centered.T, centered) / (len(data) - 1)

        # Eigendecomposition for whitening
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Keep top components
        n_components = min(self.target_dim, len(eigenvalues))
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Whitening matrix
        whitening = eigenvectors * (1.0 / np.sqrt(eigenvalues + 1e-8))

        self._domain_transforms[domain_key] = {
            "mean": mean,
            "whitening": whitening,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "n_components": n_components,
            "corpus_stats": corpus.statistics,
        }
        self._is_fitted = True

    def normalize(self, data: np.ndarray, domain: SpectralDomain) -> np.ndarray:
        """Normalize data from a specific domain into shared space."""
        domain_key = domain.value
        if domain_key not in self._domain_transforms:
            # Fall back to simple z-normalization
            return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

        transform = self._domain_transforms[domain_key]
        centered = data - transform["mean"]

        if self.method == "whitening":
            # Apply whitening
            W = transform["whitening"]
            n_feat = min(centered.shape[1], W.shape[0])
            normalized = np.dot(centered[:, :n_feat], W[:n_feat, :])
        elif self.method == "standardize":
            std = np.sqrt(transform["eigenvalues"][:centered.shape[1]] + 1e-8)
            V = transform["eigenvectors"][:centered.shape[1], :]
            normalized = np.dot(centered, V) / std
        else:
            normalized = centered / (np.std(centered, axis=0) + 1e-8)

        # Pad or truncate to target_dim
        if normalized.shape[1] < self.target_dim:
            pad = np.zeros((normalized.shape[0], self.target_dim - normalized.shape[1]))
            normalized = np.concatenate([normalized, pad], axis=1)
        elif normalized.shape[1] > self.target_dim:
            normalized = normalized[:, :self.target_dim]

        return normalized

    def compute_domain_distance(self, domain_a: SpectralDomain, domain_b: SpectralDomain) -> float:
        """Compute distance between two domains in normalized space."""
        key_a = domain_a.value
        key_b = domain_b.value
        if key_a not in self._domain_transforms or key_b not in self._domain_transforms:
            return float("inf")

        # Compare eigenvalue spectra as domain fingerprints
        ev_a = self._domain_transforms[key_a]["eigenvalues"]
        ev_b = self._domain_transforms[key_b]["eigenvalues"]
        n = min(len(ev_a), len(ev_b))
        return float(np.sqrt(np.sum((ev_a[:n] - ev_b[:n]) ** 2)))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_domains(self) -> int:
        return len(self._domain_transforms)


class CORALTransfer:
    """
    CORrelation ALignment for cross-domain spectral transfer.

    Aligns the second-order statistics (covariance) of the source domain
    to match the target domain, enabling knowledge transfer across
    different spectral measurement contexts.
    """

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization
        self._source_transform: Optional[np.ndarray] = None
        self._target_cov: Optional[np.ndarray] = None
        self._source_cov: Optional[np.ndarray] = None
        self._alignment_loss: float = float("inf")
        self._is_fitted = False

    def fit(self, source_data: np.ndarray, target_data: np.ndarray) -> float:
        """
        Fit CORAL alignment from source to target domain.

        Returns alignment loss (lower = better alignment).
        """
        # Compute covariances
        source_mean = np.mean(source_data, axis=0)
        target_mean = np.mean(target_data, axis=0)

        Cs = np.cov(source_data.T) + self.regularization * np.eye(source_data.shape[1])
        Ct = np.cov(target_data.T) + self.regularization * np.eye(target_data.shape[1])

        self._source_cov = Cs
        self._target_cov = Ct

        # Compute whitening of source
        Ds, Vs = np.linalg.eigh(Cs)
        Ds = np.maximum(Ds, 1e-8)
        Cs_inv_sqrt = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T

        # Compute coloring of target
        Dt, Vt = np.linalg.eigh(Ct)
        Dt = np.maximum(Dt, 1e-8)
        Ct_sqrt = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T

        # Combined transform: whiten source, then color with target
        self._source_transform = Cs_inv_sqrt @ Ct_sqrt
        self._source_mean = source_mean
        self._target_mean = target_mean

        # Compute alignment loss (Frobenius norm of covariance difference)
        aligned_cov = self._source_transform.T @ Cs @ self._source_transform
        self._alignment_loss = float(np.linalg.norm(aligned_cov - Ct, "fro"))
        self._is_fitted = True
        return self._alignment_loss

    def transform(self, source_data: np.ndarray) -> np.ndarray:
        """Transform source data to align with target domain."""
        if not self._is_fitted:
            raise RuntimeError("CORAL not fitted. Call fit() first.")
        centered = source_data - self._source_mean
        aligned = centered @ self._source_transform
        return aligned + self._target_mean

    @property
    def alignment_loss(self) -> float:
        return self._alignment_loss

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class MMDTransfer:
    """
    Maximum Mean Discrepancy for cross-domain spectral transfer.

    Minimizes the MMD between source and target domain representations,
    enabling domain-agnostic feature learning for spectral data.
    """

    def __init__(self, kernel: str = "rbf", bandwidth: float = 1.0, n_features: int = 64):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_features = n_features
        self._transform_matrix: Optional[np.ndarray] = None
        self._mmd_history: list[float] = []
        self._is_fitted = False

    def compute_mmd(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute MMD between source and target distributions."""
        if self.kernel == "rbf":
            return self._rbf_mmd(source, target)
        elif self.kernel == "linear":
            return self._linear_mmd(source, target)
        elif self.kernel == "polynomial":
            return self._poly_mmd(source, target)
        return self._rbf_mmd(source, target)

    def _rbf_mmd(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute MMD with RBF kernel."""
        gamma = 1.0 / (2.0 * self.bandwidth**2)
        # K(source, source)
        ss = self._rbf_kernel_matrix(source, source, gamma)
        # K(target, target)
        tt = self._rbf_kernel_matrix(target, target, gamma)
        # K(source, target)
        st = self._rbf_kernel_matrix(source, target, gamma)

        n_s, n_t = len(source), len(target)
        mmd = np.sum(ss) / (n_s * n_s) + np.sum(tt) / (n_t * n_t) - 2 * np.sum(st) / (n_s * n_t)
        return float(max(0, mmd))

    def _linear_mmd(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute MMD with linear kernel."""
        mean_s = np.mean(source, axis=0)
        mean_t = np.mean(target, axis=0)
        return float(np.sum((mean_s - mean_t) ** 2))

    def _poly_mmd(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute MMD with polynomial kernel."""
        # Degree-2 polynomial kernel
        mean_s = np.mean(source, axis=0)
        mean_t = np.mean(target, axis=0)
        diff = mean_s - mean_t
        return float(np.sum(diff**2) + np.sum(diff**2) ** 2)

    def _rbf_kernel_matrix(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix efficiently."""
        # Use subset for efficiency
        max_samples = 200
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
        if len(Y) > max_samples:
            idx = np.random.choice(len(Y), max_samples, replace=False)
            Y = Y[idx]

        sq_X = np.sum(X**2, axis=1)[:, None]
        sq_Y = np.sum(Y**2, axis=1)[None, :]
        dists = sq_X + sq_Y - 2 * X @ Y.T
        return np.exp(-gamma * dists)

    def fit(self, source: np.ndarray, target: np.ndarray, n_iterations: int = 100, lr: float = 0.01) -> float:
        """Learn a transform that minimizes MMD between domains."""
        input_dim = source.shape[1]
        # Initialize transform
        rng = np.random.default_rng(42)
        W = rng.normal(0, 0.1, (input_dim, self.n_features))

        best_mmd = float("inf")
        best_W = W.copy()

        for iteration in range(n_iterations):
            # Transform source
            source_t = source @ W
            target_t = target @ W[:target.shape[1], :]

            # Compute MMD
            mmd = self.compute_mmd(source_t, target_t)
            self._mmd_history.append(mmd)

            if mmd < best_mmd:
                best_mmd = mmd
                best_W = W.copy()

            # Gradient approximation (random perturbation)
            perturbation = rng.normal(0, 0.01, W.shape)
            W_perturbed = W + perturbation
            source_p = source @ W_perturbed
            target_p = target @ W_perturbed[:target.shape[1], :]
            mmd_p = self.compute_mmd(source_p, target_p)

            # Update
            if mmd_p < mmd:
                W = W_perturbed
            else:
                W -= lr * perturbation

        self._transform_matrix = best_W
        self._is_fitted = True
        return best_mmd

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the learned MMD-minimizing projection."""
        if not self._is_fitted:
            raise RuntimeError("MMD transfer not fitted")
        n_feat = min(data.shape[1], self._transform_matrix.shape[0])
        return data[:, :n_feat] @ self._transform_matrix[:n_feat, :]

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def mmd_history(self) -> list[float]:
        return self._mmd_history.copy()


class CrossDomainTransferEngine:
    """
    Cross-domain spectral transfer engine.

    Enables models trained on one spectral domain (e.g., earthquake harmonics)
    to transfer knowledge to another domain (e.g., bridge vibration anomalies).
    This is the hallmark of a foundation model — generalization across domains.

    Supported transfer paths:
    - Seismic → Structural Vibration (earthquake harmonics → bridge anomalies)
    - EEG Neural → Audio Acoustic (brain oscillations → resonance detection)
    - Electromagnetic → Optical (RF patterns → spectroscopy)
    - Climate → Financial (atmospheric cycles → market patterns)
    """

    def __init__(
        self,
        normalizer: Optional[DomainInvariantNormalizer] = None,
        coral_reg: float = 1.0,
        mmd_bandwidth: float = 1.0,
        shared_dim: int = 64,
    ):
        self.normalizer = normalizer or DomainInvariantNormalizer(target_dim=shared_dim)
        self.coral = CORALTransfer(regularization=coral_reg)
        self.mmd = MMDTransfer(bandwidth=mmd_bandwidth, n_features=shared_dim)
        self.shared_dim = shared_dim
        self._corpora: dict[str, SpectralCorpus] = {}
        self._transfer_results: list[dict] = {}
        self._domain_graph: dict[str, dict[str, float]] = {}
        self._shared_encoder: Optional[np.ndarray] = None

    def register_corpus(self, corpus: SpectralCorpus) -> None:
        """Register a spectral corpus from a domain."""
        key = corpus.domain.value
        self._corpora[key] = corpus
        self.normalizer.fit_domain(corpus)
        # Update domain similarity graph
        for other_key, other_corpus in self._corpora.items():
            if other_key != key:
                sim = corpus.descriptor.compute_similarity(other_corpus.descriptor)
                if key not in self._domain_graph:
                    self._domain_graph[key] = {}
                if other_key not in self._domain_graph:
                    self._domain_graph[other_key] = {}
                self._domain_graph[key][other_key] = sim
                self._domain_graph[other_key][key] = sim

    def transfer(
        self,
        source_domain: SpectralDomain,
        target_domain: SpectralDomain,
        source_data: Optional[np.ndarray] = None,
        target_data: Optional[np.ndarray] = None,
        method: str = "coral",
    ) -> dict:
        """
        Transfer knowledge from source to target domain.

        Args:
            source_domain: The domain to transfer FROM
            target_domain: The domain to transfer TO
            source_data: Optional override data (uses corpus if None)
            target_data: Optional override data (uses corpus if None)
            method: Transfer method ('coral', 'mmd', 'combined')

        Returns:
            Transfer results including aligned data and metrics
        """
        # Get data
        src_key = source_domain.value
        tgt_key = target_domain.value

        if source_data is None and src_key in self._corpora:
            source_data = self._corpora[src_key].data
        if target_data is None and tgt_key in self._corpora:
            target_data = self._corpora[tgt_key].data

        if source_data is None or target_data is None:
            raise ValueError("No data available for transfer")

        # Normalize both to shared space
        src_norm = self.normalizer.normalize(source_data, source_domain)
        tgt_norm = self.normalizer.normalize(target_data, target_domain)

        # Ensure same dimensionality
        min_dim = min(src_norm.shape[1], tgt_norm.shape[1])
        src_norm = src_norm[:, :min_dim]
        tgt_norm = tgt_norm[:, :min_dim]

        # Apply transfer method
        if method == "coral":
            loss = self.coral.fit(src_norm, tgt_norm)
            aligned = self.coral.transform(src_norm)
            mmd_after = self.mmd.compute_mmd(aligned, tgt_norm)
        elif method == "mmd":
            mmd_before = self.mmd.compute_mmd(src_norm, tgt_norm)
            mmd_after_fit = self.mmd.fit(src_norm, tgt_norm, n_iterations=50)
            aligned = self.mmd.transform(src_norm)
            loss = mmd_after_fit
            mmd_after = loss
        elif method == "combined":
            # First CORAL, then MMD refinement
            coral_loss = self.coral.fit(src_norm, tgt_norm)
            coral_aligned = self.coral.transform(src_norm)
            mmd_loss = self.mmd.fit(coral_aligned, tgt_norm, n_iterations=30)
            aligned = self.mmd.transform(coral_aligned)
            loss = coral_loss + mmd_loss
            mmd_after = self.mmd.compute_mmd(aligned, tgt_norm)
        else:
            aligned = src_norm
            loss = 0.0
            mmd_after = self.mmd.compute_mmd(src_norm, tgt_norm)

        # Compute transfer metrics
        mmd_before = self.mmd.compute_mmd(src_norm, tgt_norm)
        transfer_efficiency = 1.0 - (mmd_after / (mmd_before + 1e-10))

        result = {
            "source_domain": src_key,
            "target_domain": tgt_key,
            "method": method,
            "alignment_loss": float(loss),
            "mmd_before": float(mmd_before),
            "mmd_after": float(mmd_after),
            "transfer_efficiency": float(transfer_efficiency),
            "aligned_data": aligned,
            "source_samples": src_norm.shape[0],
            "target_samples": tgt_norm.shape[0],
            "shared_dim": min_dim,
        }

        transfer_key = f"{src_key}->{tgt_key}"
        self._transfer_results[transfer_key] = result
        return result

    def find_best_source(self, target_domain: SpectralDomain) -> Optional[str]:
        """Find the most compatible source domain for transfer to target."""
        tgt_key = target_domain.value
        if tgt_key not in self._domain_graph:
            return None
        similarities = self._domain_graph[tgt_key]
        if not similarities:
            return None
        return max(similarities, key=similarities.get)

    def get_transfer_path(self, source: SpectralDomain, target: SpectralDomain) -> list[str]:
        """Find optimal multi-hop transfer path between distant domains."""
        src_key = source.value
        tgt_key = target.value

        if src_key not in self._domain_graph or tgt_key not in self._domain_graph:
            return [src_key, tgt_key]

        # Direct transfer
        if tgt_key in self._domain_graph.get(src_key, {}):
            return [src_key, tgt_key]

        # BFS for shortest path
        visited = {src_key}
        queue = [(src_key, [src_key])]
        while queue:
            current, path = queue.pop(0)
            for neighbor in self._domain_graph.get(current, {}):
                if neighbor == tgt_key:
                    return path + [tgt_key]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return [src_key, tgt_key]

    @property
    def n_registered_domains(self) -> int:
        return len(self._corpora)

    @property
    def transfer_history(self) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if kk != "aligned_data"}
                for k, v in self._transfer_results.items()}


class SpectralDomainGenerator:
    """
    Generate synthetic spectral data for different domains.

    Creates realistic spectral signals with domain-specific characteristics
    for testing and demonstrating cross-domain transfer capabilities.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_seismic(self, n_samples: int = 100, n_features: int = 256) -> SpectralCorpus:
        """Generate synthetic seismic spectral data (earthquake harmonics)."""
        t = np.linspace(0, 10, n_features)
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Seismic: low frequency, decaying oscillations
            event_type = i % 4  # P-wave, S-wave, surface wave, noise
            labels[i] = event_type

            if event_type == 0:  # P-wave: high freq, low amplitude
                freq = self.rng.uniform(5, 15)
                data[i] = np.exp(-t * 0.5) * np.sin(2 * np.pi * freq * t)
            elif event_type == 1:  # S-wave: lower freq, higher amplitude
                freq = self.rng.uniform(1, 5)
                data[i] = 2 * np.exp(-t * 0.3) * np.sin(2 * np.pi * freq * t)
            elif event_type == 2:  # Surface wave: very low freq, long duration
                freq = self.rng.uniform(0.1, 1)
                data[i] = 3 * np.exp(-t * 0.1) * np.sin(2 * np.pi * freq * t)
            else:  # Background noise
                data[i] = self.rng.normal(0, 0.3, n_features)

            data[i] += self.rng.normal(0, 0.1, n_features)

        descriptor = DomainDescriptor(
            domain=SpectralDomain.SEISMIC,
            name="Earthquake Harmonics",
            frequency_range=(0.01, 20.0),
            typical_snr=15.0,
            key_features=["decay_rate", "dominant_frequency", "amplitude_ratio", "duration"],
        )
        return SpectralCorpus(domain=SpectralDomain.SEISMIC, descriptor=descriptor, data=data, labels=labels)

    def generate_structural_vibration(self, n_samples: int = 100, n_features: int = 256) -> SpectralCorpus:
        """Generate synthetic structural vibration data (bridge anomalies)."""
        t = np.linspace(0, 10, n_features)
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            condition = i % 4  # healthy, crack, corrosion, overload
            labels[i] = condition

            # Base vibration (structural modes)
            modes = self.rng.integers(2, 6)
            for m in range(modes):
                freq = (m + 1) * self.rng.uniform(2, 8)
                amp = 1.0 / (m + 1)
                data[i] += amp * np.sin(2 * np.pi * freq * t)

            if condition == 1:  # Crack: adds high-frequency harmonics
                data[i] += 0.5 * np.sin(2 * np.pi * 50 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
            elif condition == 2:  # Corrosion: reduces resonance, adds damping
                data[i] *= np.exp(-t * 0.3)
                data[i] += self.rng.normal(0, 0.2, n_features)
            elif condition == 3:  # Overload: shifts modes, nonlinear
                data[i] = np.sign(data[i]) * np.abs(data[i]) ** 0.7

            data[i] += self.rng.normal(0, 0.05, n_features)

        descriptor = DomainDescriptor(
            domain=SpectralDomain.STRUCTURAL_VIBRATION,
            name="Bridge Vibration Monitoring",
            frequency_range=(0.1, 100.0),
            typical_snr=25.0,
            key_features=["modal_frequencies", "damping_ratio", "harmonic_distortion", "mode_coupling"],
        )
        return SpectralCorpus(domain=SpectralDomain.STRUCTURAL_VIBRATION, descriptor=descriptor, data=data, labels=labels)

    def generate_eeg(self, n_samples: int = 100, n_features: int = 256) -> SpectralCorpus:
        """Generate synthetic EEG spectral data (brain oscillations)."""
        t = np.linspace(0, 4, n_features)  # 4 seconds
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            brain_state = i % 5  # delta, theta, alpha, beta, gamma
            labels[i] = brain_state

            # EEG bands
            if brain_state == 0:  # Delta (0.5-4 Hz) - deep sleep
                freq = self.rng.uniform(0.5, 4)
                data[i] = 5 * np.sin(2 * np.pi * freq * t)
            elif brain_state == 1:  # Theta (4-8 Hz) - drowsy
                freq = self.rng.uniform(4, 8)
                data[i] = 3 * np.sin(2 * np.pi * freq * t)
            elif brain_state == 2:  # Alpha (8-13 Hz) - relaxed
                freq = self.rng.uniform(8, 13)
                data[i] = 2 * np.sin(2 * np.pi * freq * t)
            elif brain_state == 3:  # Beta (13-30 Hz) - active
                freq = self.rng.uniform(13, 30)
                data[i] = 1.5 * np.sin(2 * np.pi * freq * t)
            else:  # Gamma (30-100 Hz) - cognitive
                freq = self.rng.uniform(30, 80)
                data[i] = 0.5 * np.sin(2 * np.pi * freq * t)

            # Add 1/f noise (characteristic of EEG)
            freqs = np.fft.rfftfreq(n_features)
            pink_noise = self.rng.normal(0, 1, len(freqs))
            pink_noise /= (freqs + 0.01)
            pink_noise_td = np.fft.irfft(pink_noise, n_features)
            data[i] += 0.3 * pink_noise_td[:n_features]
            data[i] += self.rng.normal(0, 0.1, n_features)

        descriptor = DomainDescriptor(
            domain=SpectralDomain.EEG_NEURAL,
            name="EEG Brain Oscillations",
            frequency_range=(0.5, 100.0),
            typical_snr=10.0,
            n_channels=1,
            sampling_rate=256.0,
            key_features=["band_power", "peak_frequency", "spectral_slope", "connectivity"],
        )
        return SpectralCorpus(domain=SpectralDomain.EEG_NEURAL, descriptor=descriptor, data=data, labels=labels)

    def generate_audio(self, n_samples: int = 100, n_features: int = 256) -> SpectralCorpus:
        """Generate synthetic audio spectral data (acoustic resonance)."""
        t = np.linspace(0, 1, n_features)
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            sound_type = i % 4  # tone, chord, noise, resonance
            labels[i] = sound_type

            if sound_type == 0:  # Pure tone
                freq = self.rng.uniform(100, 4000)
                data[i] = np.sin(2 * np.pi * freq * t)
            elif sound_type == 1:  # Chord (harmonic series)
                fundamental = self.rng.uniform(100, 500)
                for h in range(1, 6):
                    amp = 1.0 / h
                    data[i] += amp * np.sin(2 * np.pi * fundamental * h * t)
            elif sound_type == 2:  # Filtered noise
                noise = self.rng.normal(0, 1, n_features)
                # Simple lowpass via moving average
                kernel_size = self.rng.integers(5, 20)
                kernel = np.ones(kernel_size) / kernel_size
                data[i] = np.convolve(noise, kernel, mode="same")
            else:  # Resonance (decaying oscillation)
                freq = self.rng.uniform(200, 2000)
                decay = self.rng.uniform(2, 10)
                data[i] = np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)

            data[i] += self.rng.normal(0, 0.02, n_features)

        descriptor = DomainDescriptor(
            domain=SpectralDomain.AUDIO_ACOUSTIC,
            name="Audio Resonance Detection",
            frequency_range=(20.0, 20000.0),
            typical_snr=30.0,
            sampling_rate=44100.0,
            key_features=["fundamental_freq", "harmonics", "decay_rate", "bandwidth"],
        )
        return SpectralCorpus(domain=SpectralDomain.AUDIO_ACOUSTIC, descriptor=descriptor, data=data, labels=labels)

    def generate_electromagnetic(self, n_samples: int = 100, n_features: int = 256) -> SpectralCorpus:
        """Generate synthetic electromagnetic spectral data."""
        wavelengths = np.linspace(300, 700, n_features)  # nm (visible range)
        data = np.zeros((n_samples, n_features))
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            element = i % 5
            labels[i] = element

            # Emission lines at characteristic wavelengths
            if element == 0:  # Hydrogen-like
                lines = [410, 434, 486, 656]
            elif element == 1:  # Helium-like
                lines = [388, 447, 471, 501, 587]
            elif element == 2:  # Sodium-like
                lines = [330, 498, 568, 589, 616]
            elif element == 3:  # Neon-like
                lines = [540, 585, 614, 640, 659]
            else:  # Broadband source
                lines = list(self.rng.uniform(350, 650, 8))

            for line in lines:
                width = self.rng.uniform(2, 10)
                amp = self.rng.uniform(0.5, 2.0)
                data[i] += amp * np.exp(-0.5 * ((wavelengths - line) / width) ** 2)

            # Add continuum and noise
            data[i] += 0.1 * (1 + 0.001 * (wavelengths - 500))
            data[i] += self.rng.normal(0, 0.02, n_features)

        descriptor = DomainDescriptor(
            domain=SpectralDomain.ELECTROMAGNETIC,
            name="Electromagnetic Spectroscopy",
            frequency_range=(4.3e14, 1e15),  # Hz for visible light
            typical_snr=35.0,
            key_features=["emission_lines", "line_width", "continuum_level", "line_ratios"],
        )
        return SpectralCorpus(domain=SpectralDomain.ELECTROMAGNETIC, descriptor=descriptor, data=data, labels=labels)

    def generate_all_domains(self, n_samples: int = 100, n_features: int = 256) -> dict[str, SpectralCorpus]:
        """Generate corpora for all available domains."""
        return {
            "seismic": self.generate_seismic(n_samples, n_features),
            "structural_vibration": self.generate_structural_vibration(n_samples, n_features),
            "eeg": self.generate_eeg(n_samples, n_features),
            "audio": self.generate_audio(n_samples, n_features),
            "electromagnetic": self.generate_electromagnetic(n_samples, n_features),
        }


class TransferLearningPipeline:
    """
    End-to-end cross-domain transfer learning pipeline.

    Orchestrates the full transfer process:
    1. Domain characterization and corpus registration
    2. Domain-invariant normalization
    3. Source-target alignment (CORAL/MMD/combined)
    4. Knowledge transfer and evaluation
    5. Multi-hop transfer for distant domains
    """

    def __init__(self, shared_dim: int = 64):
        self.shared_dim = shared_dim
        self.engine = CrossDomainTransferEngine(shared_dim=shared_dim)
        self.generator = SpectralDomainGenerator()
        self._evaluation_results: list[dict] = []
        self._is_initialized = False

    def initialize_with_synthetic(self, n_samples: int = 100, n_features: int = 256) -> dict:
        """Initialize pipeline with synthetic multi-domain corpora."""
        corpora = self.generator.generate_all_domains(n_samples, n_features)
        for name, corpus in corpora.items():
            self.engine.register_corpus(corpus)
        self._is_initialized = True
        return {"n_domains": len(corpora), "domains": list(corpora.keys())}

    def evaluate_transfer(
        self,
        source_domain: SpectralDomain,
        target_domain: SpectralDomain,
        method: str = "coral",
    ) -> dict:
        """Evaluate transfer between two domains."""
        result = self.engine.transfer(source_domain, target_domain, method=method)

        # Evaluate alignment quality
        aligned = result["aligned_data"]
        target_key = target_domain.value
        if target_key in self.engine._corpora:
            target_data = self.engine.normalizer.normalize(
                self.engine._corpora[target_key].data, target_domain
            )
            min_dim = min(aligned.shape[1], target_data.shape[1])
            # Nearest-neighbor transfer accuracy
            if self.engine._corpora[target_key].labels is not None:
                src_key = source_domain.value
                if src_key in self.engine._corpora and self.engine._corpora[src_key].labels is not None:
                    src_labels = self.engine._corpora[src_key].labels
                    tgt_labels = self.engine._corpora[target_key].labels

                    # For each target sample, find nearest aligned source
                    n_correct = 0
                    n_total = min(50, len(target_data))
                    for j in range(n_total):
                        dists = np.linalg.norm(aligned[:, :min_dim] - target_data[j, :min_dim], axis=1)
                        nearest_src = np.argmin(dists)
                        if src_labels[nearest_src] == tgt_labels[j]:
                            n_correct += 1
                    transfer_accuracy = n_correct / n_total
                    result["transfer_accuracy"] = transfer_accuracy

        self._evaluation_results.append(result)
        return result

    def evaluate_all_transfers(self, method: str = "coral") -> dict:
        """Evaluate transfer between all registered domain pairs."""
        domains = list(self.engine._corpora.keys())
        results = {}
        for src in domains:
            for tgt in domains:
                if src != tgt:
                    src_domain = SpectralDomain(src)
                    tgt_domain = SpectralDomain(tgt)
                    key = f"{src}->{tgt}"
                    try:
                        result = self.evaluate_transfer(src_domain, tgt_domain, method)
                        results[key] = {
                            "efficiency": result["transfer_efficiency"],
                            "mmd_reduction": result["mmd_before"] - result["mmd_after"],
                        }
                    except Exception as e:
                        results[key] = {"error": str(e)}
        return results

    def find_optimal_transfer_strategy(
        self, source: SpectralDomain, target: SpectralDomain
    ) -> dict:
        """Find the best transfer method for a source-target pair."""
        methods = ["coral", "mmd", "combined"]
        best_method = None
        best_efficiency = float("-inf")
        results = {}

        for method in methods:
            try:
                result = self.engine.transfer(source, target, method=method)
                efficiency = result["transfer_efficiency"]
                results[method] = efficiency
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_method = method
            except Exception:
                results[method] = None

        return {
            "best_method": best_method,
            "best_efficiency": best_efficiency,
            "all_results": results,
        }

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def n_evaluations(self) -> int:
        return len(self._evaluation_results)
