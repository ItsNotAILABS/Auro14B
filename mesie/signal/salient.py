"""Salient point / vector extraction from time-frequency maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from mesie.signal.time_frequency import TimeFrequencyMap


@dataclass
class SalientPoint:
    """A local maximum in the time-frequency plane."""

    freq_hz: float
    time_s: float
    magnitude: float
    freq_index: int
    time_index: int


@dataclass
class SalientFeatureSet:
    """Salient landmarks plus compact feature vector for embedding."""

    points: List[SalientPoint]
    feature_vector: np.ndarray
    n_points: int
    threshold: float


class SalientFeatureExtractor:
    """Extract salient TF peaks and a fixed-length descriptor vector."""

    def __init__(
        self,
        max_points: int = 32,
        percentile_threshold: float = 85.0,
        neighborhood: int = 2,
    ) -> None:
        self.max_points = max_points
        self.percentile_threshold = percentile_threshold
        self.neighborhood = neighborhood

    def extract(self, tf_map: TimeFrequencyMap) -> SalientFeatureSet:
        mag = np.asarray(tf_map.matrix, dtype=np.float64)
        if mag.size == 0:
            return SalientFeatureSet([], np.zeros(self.max_points * 4), 0, 0.0)

        thresh = float(np.percentile(mag, self.percentile_threshold))
        candidates: List[SalientPoint] = []
        nf, nt = mag.shape
        r = self.neighborhood

        for i in range(r, nf - r):
            for j in range(r, nt - r):
                v = mag[i, j]
                if v < thresh:
                    continue
                patch = mag[i - r : i + r + 1, j - r : j + r + 1]
                if v >= np.max(patch) - 1e-12:
                    candidates.append(
                        SalientPoint(
                            freq_hz=float(tf_map.frequencies_hz[i]),
                            time_s=float(tf_map.times_s[j]),
                            magnitude=float(v),
                            freq_index=i,
                            time_index=j,
                        )
                    )

        candidates.sort(key=lambda p: p.magnitude, reverse=True)
        points = candidates[: self.max_points]
        vec = self._vectorize(points, tf_map)
        return SalientFeatureSet(
            points=points,
            feature_vector=vec,
            n_points=len(points),
            threshold=thresh,
        )

    def _vectorize(self, points: List[SalientPoint], tf_map: TimeFrequencyMap) -> np.ndarray:
        """Fixed-length vector: normalized (f, t, mag, rank) per salient point."""
        dim = self.max_points * 4
        out = np.zeros(dim, dtype=np.float64)
        f_scale = max(float(np.max(tf_map.frequencies_hz)), 1e-9)
        t_scale = max(float(np.max(tf_map.times_s)), 1e-9)
        m_scale = max(max(p.magnitude for p in points), 1e-12) if points else 1.0

        for k, p in enumerate(points):
            base = k * 4
            out[base] = p.freq_hz / f_scale
            out[base + 1] = p.time_s / t_scale
            out[base + 2] = p.magnitude / m_scale
            out[base + 3] = 1.0 - k / max(self.max_points, 1)
        return out

    def match_salient_sets(
        self,
        a: SalientFeatureSet,
        b: SalientFeatureSet,
    ) -> float:
        """Similarity in [0,1] from salient vector cosine."""
        va, vb = a.feature_vector, b.feature_vector
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.clip(np.dot(va, vb) / (na * nb), 0, 1))