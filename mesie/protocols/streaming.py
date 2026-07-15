"""Streaming protocol for real-time spectral data.

Provides buffered streaming, windowed processing, and event-driven
spectral data ingestion for live signal sources.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np


class StreamState(Enum):
    """Streaming connection states."""

    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    ERROR = "error"


class EventType(Enum):
    """Types of streaming events."""

    DATA = "data"
    WINDOW_COMPLETE = "window_complete"
    BUFFER_FULL = "buffer_full"
    ANOMALY_DETECTED = "anomaly_detected"
    CONNECTION_CHANGE = "connection_change"
    METADATA_UPDATE = "metadata_update"


@dataclass
class StreamEvent:
    """Event emitted by the streaming protocol.

    Args:
        event_type: Type of the event.
        data: Event data payload.
        timestamp: When the event occurred.
        source_id: Source of the event.
    """

    event_type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source_id: str = ""


@dataclass
class StreamConfig:
    """Configuration for streaming protocol.

    Args:
        buffer_size: Maximum number of samples in buffer.
        window_size: Samples per processing window.
        overlap: Number of overlapping samples between windows.
        sample_rate: Expected samples per second.
        auto_flush: Automatically flush when buffer is full.
    """

    buffer_size: int = 4096
    window_size: int = 256
    overlap: int = 64
    sample_rate: float = 100.0
    auto_flush: bool = True


class StreamBuffer:
    """Circular buffer for streaming spectral data.

    Manages incoming data samples with windowed access and
    overflow handling.

    Args:
        capacity: Maximum buffer capacity in samples.
        n_channels: Number of data channels.
    """

    def __init__(self, capacity: int = 4096, n_channels: int = 1) -> None:
        self.capacity = capacity
        self.n_channels = n_channels
        self._buffer = np.zeros((capacity, n_channels))
        self._write_pos = 0
        self._read_pos = 0
        self._total_written = 0
        self._overflow_count = 0

    def write(self, data: np.ndarray) -> int:
        """Write data to the buffer.

        Args:
            data: Array of shape (n_samples,) or (n_samples, n_channels).

        Returns:
            Number of samples actually written.
        """
        data = np.atleast_2d(data)
        if data.shape[1] != self.n_channels:
            if data.shape[0] == self.n_channels:
                data = data.T
            else:
                data = data[:, :self.n_channels]

        n_samples = data.shape[0]
        available = self.capacity - self.available

        if n_samples > available:
            self._overflow_count += n_samples - available
            n_samples = available

        if n_samples == 0:
            return 0

        # Write with wrap-around
        end_pos = self._write_pos + n_samples
        if end_pos <= self.capacity:
            self._buffer[self._write_pos:end_pos] = data[:n_samples]
        else:
            first_chunk = self.capacity - self._write_pos
            self._buffer[self._write_pos:] = data[:first_chunk]
            remaining = n_samples - first_chunk
            self._buffer[:remaining] = data[first_chunk:n_samples]

        self._write_pos = (self._write_pos + n_samples) % self.capacity
        self._total_written += n_samples
        return n_samples

    def read(self, n_samples: int) -> np.ndarray:
        """Read data from the buffer.

        Args:
            n_samples: Number of samples to read.

        Returns:
            Array of shape (n_samples, n_channels).
        """
        n_samples = min(n_samples, self.available)
        if n_samples == 0:
            return np.zeros((0, self.n_channels))

        end_pos = self._read_pos + n_samples
        if end_pos <= self.capacity:
            data = self._buffer[self._read_pos:end_pos].copy()
        else:
            first_chunk = self.capacity - self._read_pos
            data = np.vstack([
                self._buffer[self._read_pos:],
                self._buffer[:n_samples - first_chunk],
            ])

        self._read_pos = (self._read_pos + n_samples) % self.capacity
        return data

    def peek(self, n_samples: int) -> np.ndarray:
        """Peek at data without consuming it."""
        n_samples = min(n_samples, self.available)
        if n_samples == 0:
            return np.zeros((0, self.n_channels))

        end_pos = self._read_pos + n_samples
        if end_pos <= self.capacity:
            return self._buffer[self._read_pos:end_pos].copy()
        else:
            first_chunk = self.capacity - self._read_pos
            return np.vstack([
                self._buffer[self._read_pos:],
                self._buffer[:n_samples - first_chunk],
            ])

    @property
    def available(self) -> int:
        """Number of unread samples in buffer."""
        if self._write_pos >= self._read_pos:
            return self._write_pos - self._read_pos
        return self.capacity - self._read_pos + self._write_pos

    @property
    def is_full(self) -> bool:
        """Whether buffer is at capacity."""
        return self.available >= self.capacity - 1

    @property
    def overflow_count(self) -> int:
        """Number of samples lost to overflow."""
        return self._overflow_count

    def clear(self) -> None:
        """Clear all buffer contents."""
        self._buffer[:] = 0
        self._write_pos = 0
        self._read_pos = 0


class StreamingProtocol:
    """Real-time streaming protocol for spectral data.

    Manages data ingestion, windowed processing, event emission,
    and consumer notification for live spectral streams.

    Args:
        config: Streaming configuration.
        source_id: Identifier for this stream source.
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        source_id: str = "mesie-stream",
    ) -> None:
        self.config = config or StreamConfig()
        self.source_id = source_id
        self._state = StreamState.IDLE
        self._buffer = StreamBuffer(
            capacity=self.config.buffer_size,
            n_channels=1,
        )
        self._event_queue: deque[StreamEvent] = deque(maxlen=1000)
        self._listeners: dict[EventType, list[Callable]] = {et: [] for et in EventType}
        self._windows_processed = 0
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the streaming protocol."""
        self._state = StreamState.ACTIVE
        self._start_time = time.time()
        self._emit_event(StreamEvent(
            event_type=EventType.CONNECTION_CHANGE,
            data={"state": "active"},
            source_id=self.source_id,
        ))

    def stop(self) -> None:
        """Stop the streaming protocol."""
        self._state = StreamState.CLOSED
        self._emit_event(StreamEvent(
            event_type=EventType.CONNECTION_CHANGE,
            data={"state": "closed"},
            source_id=self.source_id,
        ))

    def pause(self) -> None:
        """Pause data processing."""
        self._state = StreamState.PAUSED

    def resume(self) -> None:
        """Resume data processing."""
        self._state = StreamState.ACTIVE

    def ingest(self, samples: np.ndarray) -> list[StreamEvent]:
        """Ingest new spectral samples.

        Args:
            samples: New data samples to process.

        Returns:
            List of events triggered by this ingestion.
        """
        if self._state != StreamState.ACTIVE:
            return []

        events: list[StreamEvent] = []
        samples = np.atleast_1d(samples)
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        written = self._buffer.write(samples)

        # Emit data event
        data_event = StreamEvent(
            event_type=EventType.DATA,
            data={"samples_ingested": written},
            source_id=self.source_id,
        )
        events.append(data_event)
        self._emit_event(data_event)

        # Check for complete windows
        while self._buffer.available >= self.config.window_size:
            window = self._buffer.read(self.config.window_size)
            self._windows_processed += 1
            window_event = StreamEvent(
                event_type=EventType.WINDOW_COMPLETE,
                data={
                    "window_data": window,
                    "window_index": self._windows_processed,
                },
                source_id=self.source_id,
            )
            events.append(window_event)
            self._emit_event(window_event)

            # Simple anomaly detection (amplitude spike)
            if np.max(np.abs(window)) > 3.0 * np.std(window) + np.mean(np.abs(window)):
                anomaly_event = StreamEvent(
                    event_type=EventType.ANOMALY_DETECTED,
                    data={
                        "window_index": self._windows_processed,
                        "max_amplitude": float(np.max(np.abs(window))),
                    },
                    source_id=self.source_id,
                )
                events.append(anomaly_event)
                self._emit_event(anomaly_event)

        # Check buffer full
        if self._buffer.is_full:
            full_event = StreamEvent(
                event_type=EventType.BUFFER_FULL,
                data={"overflow_count": self._buffer.overflow_count},
                source_id=self.source_id,
            )
            events.append(full_event)
            self._emit_event(full_event)

        return events

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to a specific event type.

        Args:
            event_type: Type of event to listen for.
            callback: Function to call when event occurs.
        """
        self._listeners[event_type].append(callback)

    def _emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all registered listeners."""
        self._event_queue.append(event)
        for callback in self._listeners.get(event.event_type, []):
            callback(event)

    @property
    def state(self) -> StreamState:
        """Current stream state."""
        return self._state

    @property
    def windows_processed(self) -> int:
        """Total windows processed."""
        return self._windows_processed

    @property
    def uptime(self) -> float:
        """Seconds since stream started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def event_count(self) -> int:
        """Total events emitted."""
        return len(self._event_queue)
