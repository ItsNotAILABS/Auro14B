"""Sparse Johnson-Lindenstrauss projections for Auro model and memory lanes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Dict, Literal

import numpy as np

from auro_native_llm.model.walsh_hadamard import fwht, next_power_of_two

ProjectionKind = Literal["achlioptas", "srht"]


def achlioptas_matrix(
    input_dim: int,
    output_dim: int,
    *,
    seed: int = 873539,
    density: float = 1.0 / 3.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return a deterministic sparse JL matrix with {-scale, 0, +scale} entries."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    rng = np.random.default_rng(seed)
    active = rng.random((output_dim, input_dim)) < density
    signs = rng.choice(np.array([-1.0, 1.0]), size=(output_dim, input_dim))
    scale = math.sqrt(1.0 / (output_dim * density))
    return (active * signs * scale).astype(dtype, copy=False)


@dataclass(frozen=True)
class AchlioptasProjector:
    matrix: np.ndarray
    seed: int
    density: float

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        *,
        seed: int = 873539,
        density: float = 1.0 / 3.0,
    ) -> "AchlioptasProjector":
        return cls(achlioptas_matrix(input_dim, output_dim, seed=seed, density=density), seed, density)

    @property
    def input_dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def nonzero_count(self) -> int:
        return int(np.count_nonzero(self.matrix))

    def transform(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values)
        if source.shape[-1] != self.input_dim:
            raise ValueError(f"expected final dimension {self.input_dim}")
        return source @ self.matrix.T


@dataclass(frozen=True)
class SRHTProjector:
    input_dim: int
    output_dim: int
    padded_dim: int
    seed: int
    ordering: str
    signs: np.ndarray
    sample_indices: np.ndarray

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        *,
        seed: int = 873539,
        ordering: str = "sequency",
    ) -> "SRHTProjector":
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        padded = next_power_of_two(input_dim)
        if output_dim <= 0 or output_dim > padded:
            raise ValueError("output_dim must be within the padded transform width")
        if ordering not in {"natural", "sequency"}:
            raise ValueError(f"unknown ordering: {ordering}")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=padded)
        indices = rng.permutation(padded)[:output_dim]
        if ordering == "sequency":
            identity = np.eye(padded)
            basis = fwht(identity, normalize=False, axis=1)
            changes = np.count_nonzero(basis[:, 1:] != basis[:, :-1], axis=1)
            indices = np.argsort(changes, kind="stable")[indices]
        return cls(input_dim, output_dim, padded, seed, ordering, signs, indices)

    def transform(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values)
        if source.shape[-1] != self.input_dim:
            raise ValueError(f"expected final dimension {self.input_dim}")
        if self.padded_dim != self.input_dim:
            pad = [(0, 0)] * source.ndim
            pad[-1] = (0, self.padded_dim - self.input_dim)
            source = np.pad(source, pad)
        mixed = fwht(source * self.signs, normalize=True, axis=-1)
        return mixed[..., self.sample_indices] * math.sqrt(self.padded_dim / self.output_dim)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    method: str
    sample_count: int
    input_dim: int
    output_dim: int
    mean_relative_distance_error: float
    p95_relative_distance_error: float
    maximum_relative_distance_error: float
    state_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def projection_diagnostics(
    source: np.ndarray,
    projected: np.ndarray,
    *,
    method: str,
    state: np.ndarray,
) -> ProjectionDiagnostics:
    """Measure pairwise Euclidean distortion and hash projection state."""
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(projected, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("source and projected must be 2D with matching rows")
    upper = np.triu_indices(x.shape[0], 1)
    dx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)[upper]
    dy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)[upper]
    mask = dx > np.finfo(np.float64).eps
    relative = np.abs(dy[mask] - dx[mask]) / dx[mask]
    digest = hashlib.sha256(np.ascontiguousarray(state).view(np.uint8)).hexdigest()
    return ProjectionDiagnostics(
        method=method,
        sample_count=int(x.shape[0]),
        input_dim=int(x.shape[1]),
        output_dim=int(y.shape[1]),
        mean_relative_distance_error=float(relative.mean()) if relative.size else 0.0,
        p95_relative_distance_error=float(np.quantile(relative, 0.95)) if relative.size else 0.0,
        maximum_relative_distance_error=float(relative.max()) if relative.size else 0.0,
        state_sha256=digest,
    )
