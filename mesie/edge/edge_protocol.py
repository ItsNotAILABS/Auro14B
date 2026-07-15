"""Edge spectral communication protocol.

Defines the message format and routing logic for spectral data exchange
between satellite edge nodes using the Hz-ladder backbone.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from mesie.edge.hz_ladder import HzLadder, FrequencyTier, compute_free_space_loss
from mesie.edge.satellite_nodes import SatelliteEdgeNode, VirtualNodeNetwork


class EdgeMessageType(Enum):
    """Types of edge communication messages."""

    DATA_UPLINK = "data_uplink"
    DATA_DOWNLINK = "data_downlink"
    CROSSLINK = "crosslink"
    HANDSHAKE = "handshake"
    BEACON = "beacon"
    SPECTRAL_RECORD = "spectral_record"
    CONTROL = "control"
    ACK = "ack"


@dataclass
class EdgeMessage:
    """Message exchanged between satellite edge nodes.

    Attributes:
        message_type: Type of edge message.
        source_node_id: Originating node ID.
        dest_node_id: Destination node ID.
        frequency_Hz: Carrier frequency for this message.
        payload: Data payload.
        timestamp: Unix timestamp.
        message_id: Unique identifier.
        ttl: Time-to-live (max hops).
        priority: Message priority (0=lowest, 9=highest).
        spectral_signature: Frequency fingerprint for routing.
    """

    message_type: EdgeMessageType
    source_node_id: str
    dest_node_id: str
    frequency_Hz: float
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""
    ttl: int = 10
    priority: int = 5
    spectral_signature: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.message_id:
            content = f"{self.source_node_id}:{self.dest_node_id}:{self.timestamp}"
            self.message_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        result = {
            "message_type": self.message_type.value,
            "source_node_id": self.source_node_id,
            "dest_node_id": self.dest_node_id,
            "frequency_Hz": self.frequency_Hz,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "ttl": self.ttl,
            "priority": self.priority,
            "payload": self.payload,
        }
        if self.spectral_signature is not None:
            result["spectral_signature"] = self.spectral_signature.tolist()
        return result


@dataclass
class EdgeRoute:
    """A computed route through the satellite edge network.

    Attributes:
        source_id: Origin node.
        dest_id: Destination node.
        hops: List of intermediate node IDs.
        frequencies_Hz: Carrier frequency at each hop.
        total_latency_ms: End-to-end latency.
        total_loss_dB: Total path loss.
        ladder_tiers_used: Hz-ladder tiers traversed.
    """

    source_id: str
    dest_id: str
    hops: List[str] = field(default_factory=list)
    frequencies_Hz: List[float] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_loss_dB: float = 0.0
    ladder_tiers_used: List[int] = field(default_factory=list)


@dataclass
class SpectralHandshake:
    """Spectral handshake for establishing edge communication links.

    Uses frequency fingerprints to negotiate communication parameters
    between two nodes. The handshake includes:
    - Supported frequency bands
    - Doppler compensation parameters
    - Eco-Hz timing references
    - Agreed ladder tier for data exchange

    Attributes:
        initiator_id: Node initiating the handshake.
        responder_id: Node responding.
        offered_frequencies_Hz: Frequencies offered by initiator.
        agreed_frequency_Hz: Mutually agreed frequency.
        doppler_compensation_Hz: Pre-computed Doppler offset.
        eco_hz_sync: Eco-Hz reference used for timing.
        ladder_tier: Agreed Hz-ladder tier.
        established: Whether handshake completed.
    """

    initiator_id: str
    responder_id: str
    offered_frequencies_Hz: List[float] = field(default_factory=list)
    agreed_frequency_Hz: float = 0.0
    doppler_compensation_Hz: float = 0.0
    eco_hz_sync: float = 7.83  # Schumann fundamental default
    ladder_tier: int = 4  # SHF/Satellite default
    established: bool = False


class EdgeSpectralProtocol:
    """Protocol for spectral data exchange across satellite edge network.

    Manages message creation, routing via the Hz-ladder, spectral
    handshakes, and data delivery between virtual edge nodes.

    Args:
        network: The virtual node network.
        hz_ladder: Optional custom Hz-ladder (uses network's if not provided).
    """

    def __init__(
        self,
        network: VirtualNodeNetwork,
        hz_ladder: Optional[HzLadder] = None,
    ) -> None:
        self.network = network
        self.hz_ladder = hz_ladder or network.hz_ladder
        self._message_queue: List[EdgeMessage] = []
        self._delivered: List[EdgeMessage] = []
        self._handshakes: Dict[str, SpectralHandshake] = {}

    def initiate_handshake(
        self,
        initiator_id: str,
        responder_id: str,
    ) -> SpectralHandshake:
        """Initiate a spectral handshake between two nodes.

        Args:
            initiator_id: Node starting the connection.
            responder_id: Target node.

        Returns:
            SpectralHandshake object (established=True if successful).
        """
        initiator = self.network.nodes.get(initiator_id)
        responder = self.network.nodes.get(responder_id)

        if initiator is None or responder is None:
            return SpectralHandshake(
                initiator_id=initiator_id,
                responder_id=responder_id,
                established=False,
            )

        # Offer the initiator's frequencies
        offered = [
            initiator.uplink_frequency_Hz,
            initiator.downlink_frequency_Hz,
            initiator.crosslink_frequency_Hz,
        ]

        # Find common frequency (prefer crosslink for inter-satellite)
        responder_freqs = {
            responder.uplink_frequency_Hz,
            responder.downlink_frequency_Hz,
            responder.crosslink_frequency_Hz,
        }

        agreed = 0.0
        for f in offered:
            if f in responder_freqs:
                agreed = f
                break

        # If no exact match, use crosslink
        if agreed == 0.0:
            agreed = initiator.crosslink_frequency_Hz

        # Compute Doppler compensation
        relative_v = abs(
            initiator.orbital_tier.velocity_m_s - responder.orbital_tier.velocity_m_s
        )
        doppler = agreed * relative_v / 299_792_458.0

        # Determine ladder tier
        tier = self.hz_ladder.frequency_to_tier(agreed)
        ladder_tier = tier.tier_id if tier else 5

        # Use first eco-Hz reference from initiator
        eco_sync = 7.83
        if initiator.eco_hz_refs:
            eco_sync = initiator.eco_hz_refs[0].frequency_Hz

        handshake = SpectralHandshake(
            initiator_id=initiator_id,
            responder_id=responder_id,
            offered_frequencies_Hz=offered,
            agreed_frequency_Hz=agreed,
            doppler_compensation_Hz=doppler,
            eco_hz_sync=eco_sync,
            ladder_tier=ladder_tier,
            established=True,
        )

        key = f"{initiator_id}:{responder_id}"
        self._handshakes[key] = handshake
        return handshake

    def send_spectral_data(
        self,
        source_id: str,
        dest_id: str,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EdgeMessage:
        """Send spectral record data between edge nodes.

        Args:
            source_id: Source node.
            dest_id: Destination node.
            frequencies: Frequency array in Hz.
            amplitudes: Amplitude array.
            metadata: Optional metadata.

        Returns:
            The created EdgeMessage.
        """
        # Check for established handshake
        key = f"{source_id}:{dest_id}"
        handshake = self._handshakes.get(key)

        if handshake is None or not handshake.established:
            handshake = self.initiate_handshake(source_id, dest_id)

        carrier = handshake.agreed_frequency_Hz if handshake.established else 14e9

        payload = {
            "frequencies": frequencies.tolist(),
            "amplitudes": amplitudes.tolist(),
            "n_points": len(frequencies),
            "freq_range_Hz": [float(frequencies.min()), float(frequencies.max())],
        }
        if metadata:
            payload["metadata"] = metadata

        msg = EdgeMessage(
            message_type=EdgeMessageType.SPECTRAL_RECORD,
            source_node_id=source_id,
            dest_node_id=dest_id,
            frequency_Hz=carrier,
            payload=payload,
            spectral_signature=frequencies[:8] if len(frequencies) >= 8 else frequencies,
        )

        self._message_queue.append(msg)
        return msg

    def send_beacon(self, node_id: str) -> EdgeMessage:
        """Send an eco-Hz beacon from a node.

        Beacons are broadcast messages using Schumann fundamental
        frequency as a spectral reference.

        Args:
            node_id: Broadcasting node.

        Returns:
            The beacon message.
        """
        node = self.network.nodes.get(node_id)
        eco_freq = 7.83
        if node and node.eco_hz_refs:
            eco_freq = node.eco_hz_refs[0].frequency_Hz

        msg = EdgeMessage(
            message_type=EdgeMessageType.BEACON,
            source_node_id=node_id,
            dest_node_id="broadcast",
            frequency_Hz=eco_freq,
            payload={
                "eco_hz": eco_freq,
                "orbital_hz": node.orbital_tier.orbital_frequency_Hz if node else 0.0,
                "position_tier": node.orbital_tier.name if node else "unknown",
            },
        )
        self._message_queue.append(msg)
        return msg

    def process_queue(self) -> List[EdgeMessage]:
        """Process all queued messages and deliver them.

        Returns:
            List of successfully delivered messages.
        """
        delivered = []
        remaining = []

        for msg in self._message_queue:
            if msg.dest_node_id == "broadcast" or msg.dest_node_id in self.network.nodes:
                delivered.append(msg)
            elif msg.ttl > 0:
                msg.ttl -= 1
                remaining.append(msg)

        self._delivered.extend(delivered)
        self._message_queue = remaining
        return delivered

    def compute_route(self, source_id: str, dest_id: str) -> EdgeRoute:
        """Compute optimal route between two nodes using Hz-ladder.

        Args:
            source_id: Source node ID.
            dest_id: Destination node ID.

        Returns:
            EdgeRoute with path details.
        """
        route_info = self.network.compute_route(source_id, dest_id)

        if "error" in route_info:
            return EdgeRoute(source_id=source_id, dest_id=dest_id)

        return EdgeRoute(
            source_id=source_id,
            dest_id=dest_id,
            hops=[source_id, dest_id],
            frequencies_Hz=[
                self.network.nodes[source_id].comm_frequency_Hz,
                self.network.nodes[dest_id].comm_frequency_Hz,
            ] if source_id in self.network.nodes and dest_id in self.network.nodes else [],
            total_latency_ms=route_info.get("total_latency_ms", 0.0),
            total_loss_dB=route_info.get("total_path_loss_dB", 0.0),
            ladder_tiers_used=route_info.get("ladder_tiers", []),
        )

    @property
    def pending_messages(self) -> int:
        """Number of messages in queue."""
        return len(self._message_queue)

    @property
    def delivered_count(self) -> int:
        """Total delivered messages."""
        return len(self._delivered)
