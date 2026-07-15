"""Domain alignment methods for cross-domain spectral transfer.

Implements CORAL (CORrelation ALignment), MMD (Maximum Mean Discrepancy),
and domain-invariant normalization for aligning spectral representations
across heterogeneous domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class CORAL:
    """CORrelation ALignment for spectral domain adaptation.

    Aligns the second-order statistics (covariance) of source domain features
    to match those of the target domain, enabling transfer of spectral patterns
    across domains (e.g., seismic → vibration).

    Reference:
        Sun, Feng, Saenko (2016). "Return of Frustratingly Easy Domain Adaptation."

    Example:
        >>> coral = CORAL()
        >>> coral.fit(source_embeddings, target_embeddings)
        >>> aligned = coral.transform(source_embeddings)
    """

    def __init__(self, regularization: float = 1e-6) -> None:
        """Initialize CORAL alignment.

        Args:
            regularization: Regularization added to covariance diagonal
                for numerical stability.
        """
        self.regularization = regularization
        self._source_cov_sqrt_inv: Optional[np.ndarray] = None
        self._target_cov_sqrt: Optional[np.ndarray] = None
        self._source_mean: Optional[np.ndarray] = None
        self._target_mean: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, source: np.ndarray, target: np.ndarray) -> "CORAL":
        """Compute alignment transform from source to target domain.

        Args:
            source: Source domain embeddings, shape (n_source, d).
            target: Target domain embeddings, shape (n_target, d).

        Returns:
            Self for chaining.
        """
        self._source_mean = np.mean(source, axis=0)
        self._target_mean = np.mean(target, axis=0)

        # Center the data
        source_centered = source - self._source_mean
        target_centered = target - self._target_mean

        # Covariance matrices with regularization
        d = source.shape[1]
        reg = self.regularization * np.eye(d)

        cov_source = np.cov(source_centered, rowvar=False) + reg
        cov_target = np.cov(target_centered, rowvar=False) + reg

        # Whitening transform: C_s^{-1/2}
        self._source_cov_sqrt_inv = self._matrix_power(cov_source, -0.5)
        # Coloring transform: C_t^{1/2}
        self._target_cov_sqrt = self._matrix_power(cov_target, 0.5)

        self._is_fitted = True
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source domain data to align with target domain.

        Args:
            source: Source domain embeddings, shape (n, d).

        Returns:
            Aligned embeddings, shape (n, d).

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("CORAL.fit() must be called before transform().")

        # Center, whiten, color, re-center
        centered = source - self._source_mean
        aligned = centered @ self._source_cov_sqrt_inv @ self._target_cov_sqrt
        return aligned + self._target_mean

    def fit_transform(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fit on source/target and return aligned source.

        Args:
            source: Source domain embeddings.
            target: Target domain embeddings.

        Returns:
            Aligned source embeddings.
        """
        self.fit(source, target)
        return self.transform(source)

    def coral_distance(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute CORAL distance (Frobenius norm of covariance difference).

        Args:
            source: Source domain embeddings, shape (n_source, d).
            target: Target domain embeddings, shape (n_target, d).

        Returns:
            CORAL distance (scalar).
        """
        d = source.shape[1]
        reg = self.regularization * np.eye(d)
        cov_s = np.cov(source, rowvar=False) + reg
        cov_t = np.cov(target, rowvar=False) + reg
        diff = cov_s - cov_t
        return float(np.sum(diff ** 2)) / (4.0 * d * d)

    @staticmethod
    def _matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
        """Compute matrix raised to a fractional power via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        return (eigenvectors * (eigenvalues ** power)) @ eigenvectors.T


class MMD:
    """Maximum Mean Discrepancy for measuring domain divergence.

    Computes the MMD statistic between two distributions using kernel
    embeddings. Useful for quantifying how different two spectral domains
    are in the shared latent space.

    Reference:
        Gretton et al. (2012). "A Kernel Two-Sample Test."

    Example:
        >>> mmd = MMD(kernel='rbf', bandwidth=1.0)
        >>> distance = mmd.compute(source_embeddings, target_embeddings)
    """

    def __init__(self, kernel: str = "rbf", bandwidth: Optional[float] = None) -> None:
        """Initialize MMD computation.

        Args:
            kernel: Kernel type ('rbf' or 'linear').
            bandwidth: RBF kernel bandwidth. If None, uses median heuristic.
        """
        if kernel not in ("rbf", "linear"):
            raise ValueError(f"Unsupported kernel '{kernel}'. Use 'rbf' or 'linear'.")
        self.kernel = kernel
        self.bandwidth = bandwidth

    def compute(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute MMD^2 between source and target distributions.

        Args:
            source: Source domain samples, shape (n, d).
            target: Target domain samples, shape (m, d).

        Returns:
            MMD squared distance (non-negative scalar).
        """
        # For RBF kernel, compute bandwidth once from combined data
        bandwidth = self.bandwidth
        if self.kernel == "rbf" and bandwidth is None:
            combined = np.vstack([source, target])
            dists_all = self._pairwise_distances(combined, combined)
            pos = dists_all[dists_all > 0]
            bandwidth = float(np.sqrt(np.median(pos))) if len(pos) > 0 else 1.0

        k_ss = self._kernel_matrix(source, source, bandwidth)
        k_tt = self._kernel_matrix(target, target, bandwidth)
        k_st = self._kernel_matrix(source, target, bandwidth)

        n = source.shape[0]
        m = target.shape[0]

        # Unbiased estimator
        # Remove diagonal for unbiased estimate
        np.fill_diagonal(k_ss, 0.0)
        np.fill_diagonal(k_tt, 0.0)

        mmd2 = (
            np.sum(k_ss) / max(n * (n - 1), 1)
            + np.sum(k_tt) / max(m * (m - 1), 1)
            - 2.0 * np.sum(k_st) / max(n * m, 1)
        )
        return max(float(mmd2), 0.0)

    def domain_divergence(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute domain divergence as sqrt(MMD^2).

        Args:
            source: Source domain samples.
            target: Target domain samples.

        Returns:
            MMD distance (non-negative scalar).
        """
        return float(np.sqrt(self.compute(source, target)))

    def _kernel_matrix(
        self, x: np.ndarray, y: np.ndarray, bandwidth: Optional[float] = None
    ) -> np.ndarray:
        """Compute kernel matrix between x and y."""
        if self.kernel == "linear":
            return x @ y.T
        else:
            # RBF kernel
            dists = self._pairwise_distances(x, y)
            bw = bandwidth or self.bandwidth
            if bw is None:
                all_dists = dists.ravel()
                pos = all_dists[all_dists > 0]
                bw = float(np.sqrt(np.median(pos))) if len(pos) > 0 else 1.0
            return np.exp(-dists / (2.0 * bw ** 2))

    @staticmethod
    def _pairwise_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distances."""
        xx = np.sum(x ** 2, axis=1, keepdims=True)
        yy = np.sum(y ** 2, axis=1, keepdims=True)
        return xx + yy.T - 2.0 * (x @ y.T)


class DomainInvariantNormalizer:
    """Domain-invariant normalization for spectral embeddings.

    Applies standardization that removes domain-specific distributional
    shifts while preserving spectral structure. Combines z-score
    normalization with optional whitening to create representations
    that are transferable across domains.

    Example:
        >>> normalizer = DomainInvariantNormalizer()
        >>> normalizer.fit(multi_domain_embeddings, domain_labels)
        >>> invariant = normalizer.transform(new_embeddings, domain='seismic')
    """

    def __init__(self, whiten: bool = False) -> None:
        """Initialize domain-invariant normalizer.

        Args:
            whiten: If True, apply full whitening (decorrelation + unit variance).
        """
        self.whiten = whiten
        self._global_mean: Optional[np.ndarray] = None
        self._global_std: Optional[np.ndarray] = None
        self._whitening_matrix: Optional[np.ndarray] = None
        self._domain_stats: dict = {}
        self._is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        domain_labels: np.ndarray,
    ) -> "DomainInvariantNormalizer":
        """Fit normalizer on multi-domain data.

        Computes per-domain statistics and global normalization parameters.

        Args:
            embeddings: Multi-domain embeddings, shape (n, d).
            domain_labels: Domain label for each sample, shape (n,).

        Returns:
            Self for chaining.
        """
        # Global statistics
        self._global_mean = np.mean(embeddings, axis=0)
        self._global_std = np.std(embeddings, axis=0)
        self._global_std[self._global_std < 1e-12] = 1.0

        # Per-domain statistics for domain-specific removal
        unique_domains = np.unique(domain_labels)
        for domain in unique_domains:
            mask = domain_labels == domain
            domain_data = embeddings[mask]
            self._domain_stats[domain] = {
                "mean": np.mean(domain_data, axis=0),
                "std": np.std(domain_data, axis=0),
            }
            self._domain_stats[domain]["std"][
                self._domain_stats[domain]["std"] < 1e-12
            ] = 1.0

        # Optional whitening
        if self.whiten:
            centered = embeddings - self._global_mean
            cov = np.cov(centered, rowvar=False) + 1e-6 * np.eye(embeddings.shape[1])
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            self._whitening_matrix = (
                eigenvectors * (1.0 / np.sqrt(eigenvalues))
            ) @ eigenvectors.T

        self._is_fitted = True
        return self

    def transform(
        self,
        embeddings: np.ndarray,
        domain: Optional[str] = None,
    ) -> np.ndarray:
        """Transform embeddings to domain-invariant space.

        Args:
            embeddings: Input embeddings, shape (n, d).
            domain: If provided, first removes domain-specific bias,
                then applies global normalization.

        Returns:
            Domain-invariant embeddings, shape (n, d).

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "DomainInvariantNormalizer.fit() must be called before transform()."
            )

        result = embeddings.copy()

        # Remove domain-specific bias if domain is known
        if domain is not None and domain in self._domain_stats:
            stats = self._domain_stats[domain]
            result = (result - stats["mean"]) / stats["std"]
            # Re-scale to global distribution
            result = result * self._global_std + self._global_mean

        # Global standardization
        result = (result - self._global_mean) / self._global_std

        # Optional whitening
        if self.whiten and self._whitening_matrix is not None:
            result = result @ self._whitening_matrix

        return result

    def fit_transform(
        self,
        embeddings: np.ndarray,
        domain_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            embeddings: Multi-domain embeddings.
            domain_labels: Domain labels.

        Returns:
            Domain-invariant embeddings.
        """
        self.fit(embeddings, domain_labels)
        return self.transform(embeddings)
