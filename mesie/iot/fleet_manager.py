"""Fleet-scale IoT device management for spectral intelligence.

Manages fleets of sensor devices with local-first processing,
reducing cloud dependency, costs, latency, and privacy risks.
Coordinates distributed anomaly detection across device groups.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from mesie.iot.anomaly_detector import AnomalyConfig, AnomalyDetector, AnomalyResult


class DeviceStatus(Enum):
    """Operational status of a fleet device."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ALERTING = "alerting"


class ProcessingMode(Enum):
    """Where spectral processing occurs."""

    EDGE_ONLY = "edge_only"
    EDGE_FIRST = "edge_first"
    CLOUD_FALLBACK = "cloud_fallback"
    HYBRID = "hybrid"


@dataclass
class DeviceNode:
    """Represents a sensor device in the fleet.

    Attributes:
        device_id: Unique device identifier.
        name: Human-readable device name.
        group: Logical grouping (e.g., facility, zone).
        sample_rate: Device sampling rate in Hz.
        status: Current operational status.
        anomaly_config: Detection configuration for this device.
        last_seen: Unix timestamp of last communication.
        metadata: Device-specific metadata.
    """

    device_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = ""
    group: str = "default"
    sample_rate: float = 1000.0
    status: DeviceStatus = DeviceStatus.ONLINE
    anomaly_config: AnomalyConfig = field(default_factory=AnomalyConfig)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FleetConfig:
    """Configuration for fleet management.

    Args:
        processing_mode: Where processing happens (edge/cloud/hybrid).
        heartbeat_interval_s: Expected interval between device heartbeats.
        offline_threshold_s: Seconds of silence before marking offline.
        aggregate_anomalies: Whether to correlate anomalies across fleet.
        max_devices: Maximum devices in the fleet.
        sync_interval_s: Interval for fleet-wide state sync.
    """

    processing_mode: ProcessingMode = ProcessingMode.EDGE_FIRST
    heartbeat_interval_s: float = 10.0
    offline_threshold_s: float = 60.0
    aggregate_anomalies: bool = True
    max_devices: int = 10000
    sync_interval_s: float = 300.0


@dataclass
class FleetStatus:
    """Snapshot of fleet-wide status.

    Attributes:
        total_devices: Total registered devices.
        online_devices: Currently online count.
        alerting_devices: Devices in alerting state.
        total_anomalies: Fleet-wide anomaly count.
        fleet_health_score: Overall fleet health (0-1).
        timestamp: When this snapshot was taken.
    """

    total_devices: int = 0
    online_devices: int = 0
    alerting_devices: int = 0
    total_anomalies: int = 0
    fleet_health_score: float = 1.0
    timestamp: float = field(default_factory=time.time)


class FleetManager:
    """Manages a fleet of IoT sensor devices with edge-first spectral processing.

    Coordinates device registration, health monitoring, and distributed
    anomaly detection with minimal cloud dependency.

    Args:
        config: Fleet management configuration.
    """

    def __init__(self, config: Optional[FleetConfig] = None) -> None:
        self.config = config or FleetConfig()
        self._devices: Dict[str, DeviceNode] = {}
        self._detectors: Dict[str, AnomalyDetector] = {}
        self._fleet_anomalies: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[str, AnomalyResult], None]] = []

    def register_device(self, device: DeviceNode) -> str:
        """Register a new device in the fleet.

        Args:
            device: Device node to register.

        Returns:
            Device ID of the registered device.

        Raises:
            ValueError: If fleet is at maximum capacity.
        """
        if len(self._devices) >= self.config.max_devices:
            raise ValueError(
                f"Fleet at maximum capacity ({self.config.max_devices})"
            )

        self._devices[device.device_id] = device
        self._detectors[device.device_id] = AnomalyDetector(
            config=device.anomaly_config,
            sample_rate=device.sample_rate,
        )
        return device.device_id

    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the fleet.

        Args:
            device_id: ID of the device to remove.

        Returns:
            True if device was removed, False if not found.
        """
        if device_id in self._devices:
            del self._devices[device_id]
            del self._detectors[device_id]
            return True
        return False

    def ingest_device_data(
        self, device_id: str, samples: np.ndarray
    ) -> List[AnomalyResult]:
        """Ingest sensor data from a specific device.

        Processes data locally using the device's anomaly detector.

        Args:
            device_id: Source device ID.
            samples: Raw sensor samples.

        Returns:
            List of anomaly results.

        Raises:
            KeyError: If device is not registered.
        """
        if device_id not in self._devices:
            raise KeyError(f"Device '{device_id}' not registered in fleet")

        device = self._devices[device_id]
        device.last_seen = time.time()
        device.status = DeviceStatus.ONLINE

        detector = self._detectors[device_id]
        results = detector.ingest(samples)

        # Track fleet-wide anomalies
        for result in results:
            if result.is_anomaly:
                device.status = DeviceStatus.ALERTING
                fleet_event = {
                    "device_id": device_id,
                    "group": device.group,
                    "result": result,
                    "timestamp": time.time(),
                }
                self._fleet_anomalies.append(fleet_event)
                for cb in self._callbacks:
                    cb(device_id, result)

        return results

    def on_fleet_anomaly(
        self, callback: Callable[[str, AnomalyResult], None]
    ) -> None:
        """Register a callback for fleet-wide anomaly events.

        Args:
            callback: Function called with (device_id, result) on anomaly.
        """
        self._callbacks.append(callback)

    def get_fleet_status(self) -> FleetStatus:
        """Get current fleet-wide status snapshot.

        Returns:
            FleetStatus with aggregated metrics.
        """
        now = time.time()
        online = 0
        alerting = 0

        for device in self._devices.values():
            elapsed = now - device.last_seen
            if elapsed > self.config.offline_threshold_s:
                device.status = DeviceStatus.OFFLINE
            if device.status == DeviceStatus.ONLINE:
                online += 1
            elif device.status == DeviceStatus.ALERTING:
                alerting += 1
                online += 1  # Alerting devices are still online

        total = len(self._devices)
        health = online / total if total > 0 else 1.0

        return FleetStatus(
            total_devices=total,
            online_devices=online,
            alerting_devices=alerting,
            total_anomalies=len(self._fleet_anomalies),
            fleet_health_score=health,
        )

    def get_device(self, device_id: str) -> Optional[DeviceNode]:
        """Get a specific device by ID.

        Args:
            device_id: Device identifier.

        Returns:
            DeviceNode or None if not found.
        """
        return self._devices.get(device_id)

    def get_group_devices(self, group: str) -> List[DeviceNode]:
        """Get all devices in a logical group.

        Args:
            group: Group name to filter by.

        Returns:
            List of devices in the group.
        """
        return [d for d in self._devices.values() if d.group == group]

    def get_correlated_anomalies(
        self, time_window_s: float = 60.0
    ) -> List[Dict[str, Any]]:
        """Find anomalies that correlate across multiple devices.

        Groups anomalies occurring within the time window across
        devices in the same group, indicating fleet-wide events.

        Args:
            time_window_s: Window to consider for correlation.

        Returns:
            List of correlated anomaly clusters.
        """
        if not self._fleet_anomalies:
            return []

        now = time.time()
        recent = [
            e for e in self._fleet_anomalies
            if now - e["timestamp"] <= time_window_s
        ]

        # Group by device group
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for event in recent:
            grp = event["group"]
            groups.setdefault(grp, []).append(event)

        # Only return groups with multiple affected devices
        correlated = []
        for grp, events in groups.items():
            device_ids = set(e["device_id"] for e in events)
            if len(device_ids) > 1:
                correlated.append({
                    "group": grp,
                    "device_count": len(device_ids),
                    "device_ids": list(device_ids),
                    "event_count": len(events),
                    "time_span_s": (
                        max(e["timestamp"] for e in events)
                        - min(e["timestamp"] for e in events)
                    ),
                })

        return correlated

    @property
    def device_count(self) -> int:
        """Total number of registered devices."""
        return len(self._devices)
