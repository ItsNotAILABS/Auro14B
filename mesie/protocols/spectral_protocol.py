"""Spectral data exchange protocol.

Defines the standard message format, versioning, and validation
for exchanging spectral records between systems and services.
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class ProtocolVersion(Enum):
    """Supported protocol versions."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class MessageType(Enum):
    """Types of protocol messages."""

    RECORD = "record"
    QUERY = "query"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    BATCH = "batch"
    METADATA = "metadata"


@dataclass
class ProtocolMessage:
    """Standard protocol message for spectral data exchange.

    Args:
        message_type: Type of the message.
        payload: Message data content.
        version: Protocol version.
        timestamp: Unix timestamp of message creation.
        message_id: Unique identifier for the message.
        source: Origin system identifier.
        destination: Target system identifier.
        correlation_id: ID linking related messages.
        headers: Additional message headers.
    """

    message_type: MessageType
    payload: dict[str, Any]
    version: ProtocolVersion = ProtocolVersion.V2_0
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""
    source: str = ""
    destination: str = ""
    correlation_id: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message_id:
            content = f"{self.message_type.value}:{self.timestamp}:{id(self)}"
            self.message_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "protocol": "mesie-spectral",
            "version": self.version.value,
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "destination": self.destination,
            "correlation_id": self.correlation_id,
            "headers": self.headers,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        d = self.to_dict()
        # Convert numpy arrays in payload
        d["payload"] = _serialize_payload(d["payload"])
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolMessage":
        """Deserialize message from dictionary."""
        return cls(
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            version=ProtocolVersion(data.get("version", "2.0")),
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id", ""),
            source=data.get("source", ""),
            destination=data.get("destination", ""),
            correlation_id=data.get("correlation_id", ""),
            headers=data.get("headers", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ProtocolMessage":
        """Deserialize message from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def validate(self) -> list[str]:
        """Validate message structure. Returns list of errors."""
        errors = []
        if not self.message_id:
            errors.append("Missing message_id")
        if not self.payload:
            errors.append("Empty payload")
        if self.message_type == MessageType.RECORD:
            if "frequencies" not in self.payload and "record" not in self.payload:
                errors.append("RECORD message must contain 'frequencies' or 'record' in payload")
        return errors


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    result = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _serialize_payload(value)
        elif isinstance(value, (np.float32, np.float64)):
            result[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            result[key] = int(value)
        else:
            result[key] = value
    return result


class SpectralDataProtocol:
    """Protocol handler for spectral data exchange.

    Manages message creation, validation, routing, and acknowledgment
    for spectral record communication between systems.

    Args:
        source_id: Identifier for this protocol endpoint.
        version: Protocol version to use.
    """

    def __init__(
        self,
        source_id: str = "mesie-engine",
        version: ProtocolVersion = ProtocolVersion.V2_0,
    ) -> None:
        self.source_id = source_id
        self.version = version
        self._message_log: list[ProtocolMessage] = []
        self._handlers: dict[MessageType, list[Any]] = {mt: [] for mt in MessageType}

    def create_record_message(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        destination: str = "",
    ) -> ProtocolMessage:
        """Create a spectral record message.

        Args:
            frequencies: Frequency array in Hz.
            amplitudes: Amplitude array.
            metadata: Optional metadata dictionary.
            destination: Target system identifier.

        Returns:
            Formatted ProtocolMessage.
        """
        payload = {
            "frequencies": frequencies,
            "amplitudes": amplitudes,
            "n_points": len(frequencies),
            "freq_range": [float(frequencies.min()), float(frequencies.max())],
        }
        if metadata:
            payload["metadata"] = metadata

        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload=payload,
            version=self.version,
            source=self.source_id,
            destination=destination,
        )
        self._message_log.append(msg)
        return msg

    def create_query_message(
        self,
        query_type: str,
        parameters: dict[str, Any],
        destination: str = "",
    ) -> ProtocolMessage:
        """Create a query message.

        Args:
            query_type: Type of query (e.g., 'match', 'search', 'validate').
            parameters: Query parameters.
            destination: Target system.

        Returns:
            Formatted ProtocolMessage.
        """
        payload = {
            "query_type": query_type,
            "parameters": parameters,
        }
        msg = ProtocolMessage(
            message_type=MessageType.QUERY,
            payload=payload,
            version=self.version,
            source=self.source_id,
            destination=destination,
        )
        self._message_log.append(msg)
        return msg

    def create_batch_message(
        self,
        records: list[dict[str, Any]],
        destination: str = "",
    ) -> ProtocolMessage:
        """Create a batch message containing multiple records.

        Args:
            records: List of record dictionaries.
            destination: Target system.

        Returns:
            Formatted ProtocolMessage.
        """
        payload = {
            "batch_size": len(records),
            "records": records,
        }
        msg = ProtocolMessage(
            message_type=MessageType.BATCH,
            payload=payload,
            version=self.version,
            source=self.source_id,
            destination=destination,
        )
        self._message_log.append(msg)
        return msg

    def create_error_message(
        self,
        error_code: str,
        error_message: str,
        correlation_id: str = "",
    ) -> ProtocolMessage:
        """Create an error message.

        Args:
            error_code: Machine-readable error code.
            error_message: Human-readable error description.
            correlation_id: ID of the message that caused the error.

        Returns:
            Formatted ProtocolMessage.
        """
        payload = {
            "error_code": error_code,
            "error_message": error_message,
        }
        return ProtocolMessage(
            message_type=MessageType.ERROR,
            payload=payload,
            version=self.version,
            source=self.source_id,
            correlation_id=correlation_id,
        )

    def register_handler(self, message_type: MessageType, handler: Any) -> None:
        """Register a handler for a specific message type."""
        self._handlers[message_type].append(handler)

    def process_message(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Process an incoming message.

        Args:
            message: Incoming protocol message.

        Returns:
            Response message if applicable.
        """
        errors = message.validate()
        if errors:
            return self.create_error_message(
                error_code="VALIDATION_ERROR",
                error_message="; ".join(errors),
                correlation_id=message.message_id,
            )

        handlers = self._handlers.get(message.message_type, [])
        for handler in handlers:
            result = handler(message)
            if result is not None:
                return result

        return None

    @property
    def message_count(self) -> int:
        """Total messages created by this protocol instance."""
        return len(self._message_log)
