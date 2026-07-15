"""Serialization protocol for spectral records.

Provides multi-format serialization and deserialization for spectral
data including JSON, binary (NumPy), MessagePack-style, and CSV formats.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Optional

import numpy as np


class SerializationFormat(Enum):
    """Supported serialization formats."""

    JSON = "json"
    BINARY = "binary"
    COMPACT = "compact"
    CSV = "csv"


@dataclass
class SerializedRecord:
    """Container for a serialized spectral record.

    Args:
        data: Serialized bytes or string.
        format: Format used for serialization.
        size_bytes: Size of serialized data in bytes.
        checksum: Integrity checksum of the data.
    """

    data: bytes | str
    format: SerializationFormat
    size_bytes: int
    checksum: str


class SpectralSerializer:
    """Multi-format serializer for spectral data.

    Handles serialization and deserialization of spectral records,
    frequencies, amplitudes, and metadata across formats.

    Args:
        default_format: Default serialization format.
        compress: Whether to apply compression.
        precision: Floating-point precision (16, 32, or 64 bits).
    """

    def __init__(
        self,
        default_format: SerializationFormat = SerializationFormat.JSON,
        compress: bool = False,
        precision: int = 64,
    ) -> None:
        self.default_format = default_format
        self.compress = compress
        self.precision = precision
        self._dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]

    def serialize(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        format: Optional[SerializationFormat] = None,
    ) -> SerializedRecord:
        """Serialize spectral data.

        Args:
            frequencies: Frequency array.
            amplitudes: Amplitude array.
            metadata: Optional metadata dictionary.
            format: Serialization format (uses default if None).

        Returns:
            SerializedRecord containing the serialized data.
        """
        fmt = format or self.default_format
        metadata = metadata or {}

        if fmt == SerializationFormat.JSON:
            return self._serialize_json(frequencies, amplitudes, metadata)
        elif fmt == SerializationFormat.BINARY:
            return self._serialize_binary(frequencies, amplitudes, metadata)
        elif fmt == SerializationFormat.COMPACT:
            return self._serialize_compact(frequencies, amplitudes, metadata)
        elif fmt == SerializationFormat.CSV:
            return self._serialize_csv(frequencies, amplitudes, metadata)

        raise ValueError(f"Unsupported format: {fmt}")

    def deserialize(
        self,
        record: SerializedRecord,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Deserialize a spectral record.

        Args:
            record: SerializedRecord to deserialize.

        Returns:
            Tuple of (frequencies, amplitudes, metadata).
        """
        if record.format == SerializationFormat.JSON:
            return self._deserialize_json(record.data)
        elif record.format == SerializationFormat.BINARY:
            return self._deserialize_binary(record.data)
        elif record.format == SerializationFormat.COMPACT:
            return self._deserialize_compact(record.data)
        elif record.format == SerializationFormat.CSV:
            return self._deserialize_csv(record.data)

        raise ValueError(f"Unsupported format: {record.format}")

    def _serialize_json(
        self, freqs: np.ndarray, amps: np.ndarray, meta: dict
    ) -> SerializedRecord:
        """Serialize to JSON format."""
        data = {
            "frequencies": freqs.astype(self._dtype).tolist(),
            "amplitudes": amps.astype(self._dtype).tolist(),
            "n_points": len(freqs),
            "metadata": meta,
            "format_version": "2.0",
        }
        json_str = json.dumps(data, indent=2)
        checksum = self._compute_checksum(json_str.encode())
        return SerializedRecord(
            data=json_str,
            format=SerializationFormat.JSON,
            size_bytes=len(json_str.encode()),
            checksum=checksum,
        )

    def _serialize_binary(
        self, freqs: np.ndarray, amps: np.ndarray, meta: dict
    ) -> SerializedRecord:
        """Serialize to binary NumPy format."""
        buf = BytesIO()

        # Header: magic bytes + version + n_points + precision
        n_points = len(freqs)
        header = struct.pack("<4sIII", b"MESI", 2, n_points, self.precision)
        buf.write(header)

        # Data arrays
        freqs_typed = freqs.astype(self._dtype)
        amps_typed = amps.astype(self._dtype)
        buf.write(freqs_typed.tobytes())
        buf.write(amps_typed.tobytes())

        # Metadata as JSON bytes
        meta_bytes = json.dumps(meta).encode()
        buf.write(struct.pack("<I", len(meta_bytes)))
        buf.write(meta_bytes)

        data = buf.getvalue()
        checksum = self._compute_checksum(data)
        return SerializedRecord(
            data=data,
            format=SerializationFormat.BINARY,
            size_bytes=len(data),
            checksum=checksum,
        )

    def _serialize_compact(
        self, freqs: np.ndarray, amps: np.ndarray, meta: dict
    ) -> SerializedRecord:
        """Serialize to compact format (delta-encoded frequencies)."""
        # Delta encode frequencies for compression
        freq_deltas = np.diff(freqs, prepend=freqs[0])
        data = {
            "freq_start": float(freqs[0]),
            "freq_deltas": freq_deltas.astype(np.float32).tolist(),
            "amplitudes": amps.astype(np.float32).tolist(),
            "metadata": meta,
        }
        json_str = json.dumps(data)
        checksum = self._compute_checksum(json_str.encode())
        return SerializedRecord(
            data=json_str,
            format=SerializationFormat.COMPACT,
            size_bytes=len(json_str.encode()),
            checksum=checksum,
        )

    def _serialize_csv(
        self, freqs: np.ndarray, amps: np.ndarray, meta: dict
    ) -> SerializedRecord:
        """Serialize to CSV format."""
        lines = ["frequency,amplitude"]
        for f, a in zip(freqs, amps):
            lines.append(f"{f},{a}")
        csv_str = "\n".join(lines)
        checksum = self._compute_checksum(csv_str.encode())
        return SerializedRecord(
            data=csv_str,
            format=SerializationFormat.CSV,
            size_bytes=len(csv_str.encode()),
            checksum=checksum,
        )

    def _deserialize_json(self, data: bytes | str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Deserialize from JSON."""
        if isinstance(data, bytes):
            data = data.decode()
        parsed = json.loads(data)
        freqs = np.array(parsed["frequencies"], dtype=np.float64)
        amps = np.array(parsed["amplitudes"], dtype=np.float64)
        meta = parsed.get("metadata", {})
        return freqs, amps, meta

    def _deserialize_binary(self, data: bytes | str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Deserialize from binary format."""
        if isinstance(data, str):
            data = data.encode()
        buf = BytesIO(data)

        # Read header
        header = buf.read(16)
        magic, version, n_points, precision = struct.unpack("<4sIII", header)
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
        bytes_per_elem = precision // 8

        # Read arrays
        freqs = np.frombuffer(buf.read(n_points * bytes_per_elem), dtype=dtype).copy()
        amps = np.frombuffer(buf.read(n_points * bytes_per_elem), dtype=dtype).copy()

        # Read metadata
        meta_len = struct.unpack("<I", buf.read(4))[0]
        meta = json.loads(buf.read(meta_len).decode())

        return freqs.astype(np.float64), amps.astype(np.float64), meta

    def _deserialize_compact(self, data: bytes | str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Deserialize from compact format."""
        if isinstance(data, bytes):
            data = data.decode()
        parsed = json.loads(data)
        deltas = np.array(parsed["freq_deltas"], dtype=np.float64)
        freqs = np.cumsum(deltas)
        freqs[0] = parsed["freq_start"]
        freqs = np.cumsum(np.concatenate([[parsed["freq_start"]], np.diff(freqs)]))
        amps = np.array(parsed["amplitudes"], dtype=np.float64)
        meta = parsed.get("metadata", {})
        return freqs, amps, meta

    def _deserialize_csv(self, data: bytes | str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Deserialize from CSV format."""
        if isinstance(data, bytes):
            data = data.decode()
        lines = data.strip().split("\n")
        freqs = []
        amps = []
        for line in lines[1:]:  # Skip header
            parts = line.split(",")
            freqs.append(float(parts[0]))
            amps.append(float(parts[1]))
        return np.array(freqs), np.array(amps), {}

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        import hashlib
        return hashlib.sha256(data).hexdigest()[:16]

    def estimate_size(self, n_points: int, format: Optional[SerializationFormat] = None) -> int:
        """Estimate serialized size in bytes.

        Args:
            n_points: Number of frequency points.
            format: Target format.

        Returns:
            Estimated size in bytes.
        """
        fmt = format or self.default_format
        bytes_per_point = self.precision // 8

        if fmt == SerializationFormat.BINARY:
            return 16 + 2 * n_points * bytes_per_point + 100  # header + data + meta
        elif fmt == SerializationFormat.JSON:
            return n_points * 20 * 2 + 200  # ~20 chars per number
        elif fmt == SerializationFormat.COMPACT:
            return n_points * 12 * 2 + 200  # smaller numbers
        elif fmt == SerializationFormat.CSV:
            return n_points * 25 + 50  # ~25 chars per line
        return 0
