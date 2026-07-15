"""SovereignTensor — Pure native tensor engine with MESIE spectral integration.

SIMD-style vectorized operations using stdlib array module for cache-friendly
float32 buffers. No NumPy dependency in the core path. Integrates directly
with MESIE SpectralComponent records, helix encoding, and resonance weighting.
"""

from __future__ import annotations

import array
import math
import struct
from typing import Any, Dict, List, Optional, Tuple


class SovereignTensor:
    """SIMD-ready tensor with MESIE spectral integration.

    Attributes:
        shape: Tuple of dimension sizes.
        data: Cache-friendly float32 buffer (flattened).
        spectral_meta: MESIE spectral metadata (resonance, helix params, lineage).
        resonance: Resonance weighting score from spectral metadata.
    """

    def __init__(
        self,
        data: List[float],
        shape: Tuple[int, ...],
        spectral_meta: Optional[Dict[str, Any]] = None,
    ):
        self.shape = shape
        self.data = array.array("f", data)  # float32 buffer
        self.spectral_meta = spectral_meta or {}
        expected = self._compute_size(shape)
        if len(self.data) != expected:
            raise ValueError(
                f"Shape {shape} expects {expected} elements, got {len(self.data)}"
            )

        # MESIE-native metadata
        self.resonance = self.spectral_meta.get("resonance", 1.0)
        self.helix_params = self.spectral_meta.get("helix", {})
        self.lineage = self.spectral_meta.get("lineage", [])

    @staticmethod
    def _compute_size(shape: Tuple[int, ...]) -> int:
        """Compute total element count from shape."""
        p = 1
        for d in shape:
            p *= d
        return p

    def to_bytes(self) -> bytes:
        """Deterministic binary serialization for QSHA + Vault."""
        return self.data.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes, shape: Tuple[int, ...]) -> "SovereignTensor":
        """Reconstruct tensor from deterministic binary."""
        n = cls._compute_size(shape)
        data = list(struct.unpack(f"{n}f", raw))
        return cls(data, shape)

    @classmethod
    def from_mesie_component(cls, component: Dict) -> "SovereignTensor":
        """Direct ingestion from MESIE SpectralComponent (frequency + amplitude).

        Args:
            component: Dictionary with 'amplitude', 'frequency', and optional
                       'element_weight' and 'node_id' fields.
        """
        amp = component.get("amplitude", [0.0])
        data = list(amp)
        shape = (len(data),)
        meta = {
            "resonance": component.get("element_weight", 1.0),
            "helix": {"turns": 8, "dimensions": len(data)},
            "lineage": component.get("node_id", []),
        }
        return cls(data, shape, meta)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> "SovereignTensor":
        """Create a zero-filled tensor."""
        n = cls._compute_size(shape)
        return cls([0.0] * n, shape)

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> "SovereignTensor":
        """Create a tensor filled with ones."""
        n = cls._compute_size(shape)
        return cls([1.0] * n, shape)

    # ------------------------------------------------------------------
    # SIMD-style vectorized operations (manual unrolling for hot paths)
    # ------------------------------------------------------------------

    def vector_add(self, other: "SovereignTensor") -> "SovereignTensor":
        """SIMD-friendly 8-wide unrolled element-wise addition."""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        n = len(self.data)
        result = array.array("f", [0.0] * n)

        # 8-wide unroll (SIMD-friendly)
        i = 0
        while i + 8 <= n:
            result[i] = self.data[i] + other.data[i]
            result[i + 1] = self.data[i + 1] + other.data[i + 1]
            result[i + 2] = self.data[i + 2] + other.data[i + 2]
            result[i + 3] = self.data[i + 3] + other.data[i + 3]
            result[i + 4] = self.data[i + 4] + other.data[i + 4]
            result[i + 5] = self.data[i + 5] + other.data[i + 5]
            result[i + 6] = self.data[i + 6] + other.data[i + 6]
            result[i + 7] = self.data[i + 7] + other.data[i + 7]
            i += 8
        # Scalar tail
        while i < n:
            result[i] = self.data[i] + other.data[i]
            i += 1

        return SovereignTensor(list(result), self.shape, self.spectral_meta)

    def vector_mul(self, other: "SovereignTensor") -> "SovereignTensor":
        """SIMD-friendly element-wise multiplication."""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        n = len(self.data)
        result = array.array("f", [0.0] * n)

        i = 0
        while i + 4 <= n:
            result[i] = self.data[i] * other.data[i]
            result[i + 1] = self.data[i + 1] * other.data[i + 1]
            result[i + 2] = self.data[i + 2] * other.data[i + 2]
            result[i + 3] = self.data[i + 3] * other.data[i + 3]
            i += 4
        while i < n:
            result[i] = self.data[i] * other.data[i]
            i += 1

        return SovereignTensor(list(result), self.shape, self.spectral_meta)

    def scale(self, factor: float) -> "SovereignTensor":
        """Scalar multiplication."""
        result = [x * factor for x in self.data]
        return SovereignTensor(result, self.shape, self.spectral_meta)

    def resonance_matmul(self, other: "SovereignTensor") -> "SovereignTensor":
        """Resonance-weighted matrix multiplication.

        Applies resonance product weighting to the output, optimized
        for fixed small spectral shapes with 4-wide inner unrolling.
        """
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul requires 2D tensors")
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Inner dimensions mismatch: {self.shape[1]} vs {other.shape[0]}"
            )

        m, k = self.shape
        _, n = other.shape
        result = array.array("f", [0.0] * (m * n))
        resonance = self.resonance * other.resonance

        for i in range(m):
            for j in range(n):
                acc = 0.0
                p = 0
                # 4-wide inner unroll
                while p + 4 <= k:
                    acc += (
                        self.data[i * k + p] * other.data[p * n + j]
                        + self.data[i * k + p + 1] * other.data[(p + 1) * n + j]
                        + self.data[i * k + p + 2] * other.data[(p + 2) * n + j]
                        + self.data[i * k + p + 3] * other.data[(p + 3) * n + j]
                    )
                    p += 4
                # Scalar tail
                while p < k:
                    acc += self.data[i * k + p] * other.data[p * n + j]
                    p += 1
                result[i * n + j] = acc * resonance
        return SovereignTensor(list(result), (m, n), self.spectral_meta)

    def dot(self, other: "SovereignTensor") -> float:
        """Dot product of two 1D tensors."""
        if self.shape != other.shape or len(self.shape) != 1:
            raise ValueError("dot requires matching 1D tensors")
        acc = 0.0
        for i in range(len(self.data)):
            acc += self.data[i] * other.data[i]
        return acc

    # ------------------------------------------------------------------
    # Quantization for edge deployment
    # ------------------------------------------------------------------

    def quantize_int8(self) -> "SovereignTensor":
        """Native INT8 quantization for edge deployment.

        Stores quantized values as floats with quant_scale in metadata.
        """
        max_val = max(abs(x) for x in self.data) if self.data else 1.0
        scale = max_val if max_val > 0 else 1.0
        qdata = [float(int((x / scale) * 127)) for x in self.data]
        meta = dict(self.spectral_meta)
        meta["quant_scale"] = scale
        meta["quantized"] = True
        return SovereignTensor(qdata, self.shape, meta)

    def dequantize(self) -> "SovereignTensor":
        """Reverse INT8 quantization."""
        scale = self.spectral_meta.get("quant_scale", 1.0)
        data = [x * scale / 127.0 for x in self.data]
        meta = dict(self.spectral_meta)
        meta.pop("quant_scale", None)
        meta.pop("quantized", None)
        return SovereignTensor(data, self.shape, meta)

    # ------------------------------------------------------------------
    # Helix encoding utilities
    # ------------------------------------------------------------------

    def helix_encode(self, turns: int = 8) -> "SovereignTensor":
        """Apply helix rotation encoding to the tensor data.

        Maps data onto a helical manifold for efficient spectral retrieval.
        """
        n = len(self.data)
        encoded = array.array("f", [0.0] * n)
        for i in range(n):
            phase = 2.0 * math.pi * turns * (i / max(n - 1, 1))
            encoded[i] = self.data[i] * math.cos(phase) + math.sin(phase) * 0.1
        meta = dict(self.spectral_meta)
        meta["helix"] = {"turns": turns, "dimensions": n, "encoded": True}
        return SovereignTensor(list(encoded), self.shape, meta)

    def norm(self) -> float:
        """L2 norm of the tensor."""
        return math.sqrt(sum(x * x for x in self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"SovereignTensor(shape={self.shape}, resonance={self.resonance:.3f}, "
            f"size={len(self.data)})"
        )
