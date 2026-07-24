"""Walsh-Hadamard transforms for deterministic Auro model pre-wiring.

The implementation uses normalized Sylvester bases, supports natural and
sequency ordering, pads arbitrary feature widths to the next power of two, and
emits diagnostics suitable for checkpoint birth receipts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Dict, Literal

import numpy as np

WalshOrder = Literal["natural", "sequency"]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def fwht(values: np.ndarray, *, normalize: bool = True, axis: int = -1) -> np.ndarray:
    """Apply the fast Walsh-Hadamard transform along one power-of-two axis."""
    source = np.asarray(values)
    if source.ndim == 0:
        raise ValueError("FWHT requires at least one dimension")
    axis = axis % source.ndim
    n = int(source.shape[axis])
    if not is_power_of_two(n):
        raise ValueError(f"transform length must be a power of two, got {n}")

    dtype = np.result_type(source.dtype, np.float32)
    result = np.moveaxis(source.astype(dtype, copy=True), axis, -1)
    h = 1
    while h < n:
        shaped = result.reshape(*result.shape[:-1], -1, 2 * h)
        left = shaped[..., :h].copy()
        right = shaped[..., h : 2 * h].copy()
        shaped[..., :h] = left + right
        shaped[..., h : 2 * h] = left - right
        h *= 2
    if normalize:
        result /= math.sqrt(n)
    return np.moveaxis(result, -1, axis)


def hadamard_matrix(
    order: int,
    *,
    normalize: bool = True,
    ordering: WalshOrder = "natural",
) -> np.ndarray:
    """Construct a Sylvester Hadamard matrix with optional sequency ordering."""
    if not is_power_of_two(order):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = fwht(np.eye(order, dtype=np.float64), normalize=normalize, axis=1)
    if ordering == "natural":
        return matrix
    if ordering != "sequency":
        raise ValueError(f"unknown Walsh ordering: {ordering}")
    changes = np.count_nonzero(matrix[:, 1:] != matrix[:, :-1], axis=1)
    return matrix[np.argsort(changes, kind="stable")]


def walsh_tensor(
    rows: int,
    cols: int,
    *,
    seed: int,
    ordering: WalshOrder = "sequency",
) -> np.ndarray:
    """Create a deterministic dense, row-normalized tensor from a Walsh basis."""
    if rows <= 0 or cols <= 0:
        raise ValueError("tensor dimensions must be positive")
    width = next_power_of_two(cols)
    basis = hadamard_matrix(width, normalize=True, ordering=ordering)
    rng = np.random.default_rng(seed)
    basis = basis[rng.permutation(width)]
    basis = basis * rng.choice(np.array([-1.0, 1.0]), size=width)[:, None]
    tensor = np.tile(basis, (math.ceil(rows / width), 1))[:rows, :cols]
    norms = np.linalg.norm(tensor, axis=1, keepdims=True)
    return tensor / np.maximum(norms, 1e-12)


@dataclass(frozen=True)
class WalshDiagnostics:
    order: int
    ordering: WalshOrder
    orthogonality_max_error: float
    involution_max_error: float
    energy_error: float
    basis_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def diagnose(order: int, *, ordering: WalshOrder = "sequency") -> WalshDiagnostics:
    """Measure the mathematical invariants used by the model birth gate."""
    matrix = hadamard_matrix(order, normalize=True, ordering=ordering)
    gram = matrix @ matrix.T
    probe = np.arange(order, dtype=np.float64)
    transformed = fwht(probe, normalize=True)
    recovered = fwht(transformed, normalize=True)
    return WalshDiagnostics(
        order=order,
        ordering=ordering,
        orthogonality_max_error=float(np.max(np.abs(gram - np.eye(order)))),
        involution_max_error=float(np.max(np.abs(recovered - probe))),
        energy_error=float(abs(np.linalg.norm(transformed) - np.linalg.norm(probe))),
        basis_sha256=hashlib.sha256(np.ascontiguousarray(matrix).view(np.uint8)).hexdigest(),
    )
