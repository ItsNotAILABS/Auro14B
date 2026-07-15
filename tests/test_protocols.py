"""Tests for mesie.protocols module."""

import numpy as np
import pytest

from mesie.protocols.spectral_protocol import (
    SpectralDataProtocol,
    ProtocolMessage,
    ProtocolVersion,
    MessageType,
)
from mesie.protocols.streaming import (
    StreamingProtocol,
    StreamBuffer,
    StreamConfig,
    StreamState,
    EventType,
)
from mesie.protocols.serialization import (
    SpectralSerializer,
    SerializationFormat,
)


class TestProtocolMessage:
    """Tests for ProtocolMessage."""

    def test_creation(self):
        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload={"frequencies": [1, 2, 3], "amplitudes": [0.1, 0.2, 0.3]},
        )
        assert msg.message_id != ""
        assert msg.version == ProtocolVersion.V2_0

    def test_to_dict(self):
        msg = ProtocolMessage(
            message_type=MessageType.QUERY,
            payload={"query_type": "match"},
            source="test-system",
        )
        d = msg.to_dict()
        assert d["protocol"] == "mesie-spectral"
        assert d["message_type"] == "query"
        assert d["source"] == "test-system"

    def test_to_json_and_back(self):
        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload={"frequencies": [1.0, 2.0], "amplitudes": [0.5, 0.8]},
        )
        json_str = msg.to_json()
        restored = ProtocolMessage.from_json(json_str)
        assert restored.message_type == MessageType.RECORD
        assert restored.payload["frequencies"] == [1.0, 2.0]

    def test_validate_valid(self):
        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload={"frequencies": [1, 2], "amplitudes": [0.1, 0.2]},
        )
        errors = msg.validate()
        assert len(errors) == 0

    def test_validate_invalid(self):
        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload={"data": "invalid"},
        )
        errors = msg.validate()
        assert len(errors) > 0


class TestSpectralDataProtocol:
    """Tests for SpectralDataProtocol."""

    def test_create_record_message(self):
        protocol = SpectralDataProtocol(source_id="test")
        freqs = np.array([1.0, 2.0, 3.0])
        amps = np.array([0.5, 0.8, 0.3])
        msg = protocol.create_record_message(freqs, amps)
        assert msg.message_type == MessageType.RECORD
        assert msg.source == "test"
        assert protocol.message_count == 1

    def test_create_query_message(self):
        protocol = SpectralDataProtocol()
        msg = protocol.create_query_message("match", {"threshold": 0.8})
        assert msg.message_type == MessageType.QUERY
        assert msg.payload["query_type"] == "match"

    def test_create_batch_message(self):
        protocol = SpectralDataProtocol()
        records = [{"id": "1"}, {"id": "2"}]
        msg = protocol.create_batch_message(records)
        assert msg.payload["batch_size"] == 2

    def test_process_valid_message(self):
        protocol = SpectralDataProtocol()
        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload={"frequencies": [1], "amplitudes": [0.5]},
        )
        result = protocol.process_message(msg)
        assert result is None  # No handler registered

    def test_process_invalid_message(self):
        protocol = SpectralDataProtocol()
        msg = ProtocolMessage(
            message_type=MessageType.RECORD,
            payload={},
        )
        result = protocol.process_message(msg)
        assert result is not None
        assert result.message_type == MessageType.ERROR


class TestStreamBuffer:
    """Tests for StreamBuffer."""

    def test_write_and_read(self):
        buf = StreamBuffer(capacity=100, n_channels=1)
        data = np.ones((10, 1))
        written = buf.write(data)
        assert written == 10
        assert buf.available == 10
        read_data = buf.read(10)
        assert read_data.shape == (10, 1)
        assert np.allclose(read_data, 1.0)

    def test_overflow(self):
        buf = StreamBuffer(capacity=10, n_channels=1)
        data = np.ones((20, 1))
        written = buf.write(data)
        assert written < 20
        assert buf.overflow_count > 0

    def test_clear(self):
        buf = StreamBuffer(capacity=50, n_channels=1)
        buf.write(np.ones((10, 1)))
        buf.clear()
        assert buf.available == 0


class TestStreamingProtocol:
    """Tests for StreamingProtocol."""

    def test_lifecycle(self):
        proto = StreamingProtocol()
        assert proto.state == StreamState.IDLE
        proto.start()
        assert proto.state == StreamState.ACTIVE
        proto.pause()
        assert proto.state == StreamState.PAUSED
        proto.resume()
        assert proto.state == StreamState.ACTIVE
        proto.stop()
        assert proto.state == StreamState.CLOSED

    def test_ingest(self):
        config = StreamConfig(buffer_size=1024, window_size=32, overlap=8)
        proto = StreamingProtocol(config)
        proto.start()
        events = proto.ingest(np.random.randn(64))
        assert len(events) > 0
        assert proto.windows_processed >= 1

    def test_ingest_when_not_active(self):
        proto = StreamingProtocol()
        events = proto.ingest(np.random.randn(10))
        assert len(events) == 0

    def test_subscribe(self):
        proto = StreamingProtocol(StreamConfig(window_size=16))
        proto.start()
        received = []
        proto.subscribe(EventType.WINDOW_COMPLETE, lambda e: received.append(e))
        proto.ingest(np.random.randn(32))
        assert len(received) >= 1


class TestSpectralSerializer:
    """Tests for SpectralSerializer."""

    def test_json_roundtrip(self):
        serializer = SpectralSerializer(default_format=SerializationFormat.JSON)
        freqs = np.linspace(0.1, 10, 50)
        amps = np.random.randn(50)
        record = serializer.serialize(freqs, amps, {"source": "test"})
        assert record.size_bytes > 0
        assert record.checksum != ""
        f, a, m = serializer.deserialize(record)
        np.testing.assert_allclose(f, freqs, rtol=1e-5)
        np.testing.assert_allclose(a, amps, rtol=1e-5)
        assert m["source"] == "test"

    def test_binary_roundtrip(self):
        serializer = SpectralSerializer(default_format=SerializationFormat.BINARY)
        freqs = np.linspace(1, 100, 64)
        amps = np.random.randn(64)
        record = serializer.serialize(freqs, amps, {"key": "value"})
        f, a, m = serializer.deserialize(record)
        np.testing.assert_allclose(f, freqs, rtol=1e-10)
        np.testing.assert_allclose(a, amps, rtol=1e-10)

    def test_csv_roundtrip(self):
        serializer = SpectralSerializer(default_format=SerializationFormat.CSV)
        freqs = np.array([1.0, 2.0, 3.0])
        amps = np.array([0.5, 0.8, 0.3])
        record = serializer.serialize(freqs, amps)
        f, a, m = serializer.deserialize(record)
        np.testing.assert_allclose(f, freqs, rtol=1e-5)
        np.testing.assert_allclose(a, amps, rtol=1e-5)

    def test_compact_roundtrip(self):
        serializer = SpectralSerializer(default_format=SerializationFormat.COMPACT)
        freqs = np.linspace(0.1, 50, 100)
        amps = np.random.randn(100)
        record = serializer.serialize(freqs, amps)
        f, a, m = serializer.deserialize(record)
        assert len(f) == 100
        assert len(a) == 100

    def test_estimate_size(self):
        serializer = SpectralSerializer()
        size = serializer.estimate_size(100, SerializationFormat.BINARY)
        assert size > 0
        size_json = serializer.estimate_size(100, SerializationFormat.JSON)
        assert size_json > size  # JSON is larger than binary

    def test_precision(self):
        serializer_32 = SpectralSerializer(precision=32)
        serializer_64 = SpectralSerializer(precision=64)
        freqs = np.linspace(1, 10, 20)
        amps = np.random.randn(20)
        rec_32 = serializer_32.serialize(freqs, amps, format=SerializationFormat.BINARY)
        rec_64 = serializer_64.serialize(freqs, amps, format=SerializationFormat.BINARY)
        assert rec_32.size_bytes < rec_64.size_bytes
