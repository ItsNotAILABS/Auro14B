"""Neuromorphic runtime for edge spectral intelligence.

Optimized runtime that mimics neuromorphic computing principles —
spike-based encoding, event-driven processing, and ultra-low-power
operation patterns. Enables millions of ops/sec on tiny devices
for always-on spectral intelligence.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class ProcessingMode(Enum):
    """Runtime processing modes for power/performance trade-off."""

    ULTRA_LOW_POWER = "ultra_low_power"
    LOW_POWER = "low_power"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    BURST = "burst"


class EncodingScheme(Enum):
    """Spike encoding schemes for spectral data."""

    RATE_CODING = "rate_coding"
    TEMPORAL_CODING = "temporal_coding"
    PHASE_CODING = "phase_coding"
    POPULATION_CODING = "population_coding"


@dataclass
class RuntimeConfig:
    """Configuration for the neuromorphic runtime.

    Args:
        processing_mode: Power/performance trade-off.
        encoding: Spike encoding scheme.
        n_neurons: Number of virtual neurons in the spike layer.
        time_step_ms: Simulation time step in milliseconds.
        threshold: Spike firing threshold.
        decay_rate: Membrane potential decay rate (0-1).
        refractory_period_ms: Post-spike refractory period.
        max_spike_rate_hz: Maximum spike rate limit.
        batch_size: Samples processed per batch.
        enable_stdp: Enable spike-timing-dependent plasticity.
    """

    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    encoding: EncodingScheme = EncodingScheme.RATE_CODING
    n_neurons: int = 256
    time_step_ms: float = 1.0
    threshold: float = 1.0
    decay_rate: float = 0.95
    refractory_period_ms: float = 2.0
    max_spike_rate_hz: float = 1000.0
    batch_size: int = 32
    enable_stdp: bool = True


@dataclass
class SpikeEvent:
    """A spike event from the neuromorphic layer.

    Attributes:
        neuron_id: Which neuron fired.
        timestamp_ms: When the spike occurred.
        potential: Membrane potential at firing.
        layer: Which processing layer generated the spike.
    """

    neuron_id: int
    timestamp_ms: float
    potential: float = 0.0
    layer: int = 0


@dataclass
class RuntimeMetrics:
    """Performance metrics for the neuromorphic runtime.

    Attributes:
        ops_per_second: Operations per second achieved.
        spike_rate_hz: Current spike rate.
        energy_score: Estimated energy efficiency (0-1, 1=best).
        latency_ms: Average processing latency.
        active_neurons: Number of currently active neurons.
        total_spikes: Total spikes generated.
        uptime_s: Runtime uptime in seconds.
    """

    ops_per_second: float = 0.0
    spike_rate_hz: float = 0.0
    energy_score: float = 1.0
    latency_ms: float = 0.0
    active_neurons: int = 0
    total_spikes: int = 0
    uptime_s: float = 0.0


class NeuromorphicRuntime:
    """Optimized runtime for edge neuromorphic spectral processing.

    Implements spike-based encoding and event-driven processing of
    spectral signals, enabling ultra-efficient always-on intelligence
    on resource-constrained devices.

    The runtime maintains a virtual neuron layer that converts spectral
    features into spike trains, enabling efficient pattern matching
    and anomaly detection with minimal compute.

    Args:
        config: Runtime configuration.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self._membrane_potential = np.zeros(self.config.n_neurons)
        self._refractory_state = np.zeros(self.config.n_neurons)
        self._weights = self._init_weights()
        self._spike_history: deque = deque(maxlen=10000)
        self._total_spikes = 0
        self._total_ops = 0
        self._start_time = time.time()
        self._last_process_time = 0.0
        self._callbacks: List[Callable[[List[SpikeEvent]], None]] = []

    def _init_weights(self) -> np.ndarray:
        """Initialize synaptic weight matrix."""
        n = self.config.n_neurons
        # Sparse random connectivity
        weights = np.random.randn(n, n) * 0.1
        # Zero diagonal (no self-connections)
        np.fill_diagonal(weights, 0)
        # Make sparse (~10% connectivity)
        mask = np.random.random((n, n)) > 0.9
        weights *= mask
        return weights

    def encode(self, spectral_signal: np.ndarray) -> np.ndarray:
        """Encode a spectral signal into spike-compatible input currents.

        Args:
            spectral_signal: Raw spectral signal (any length).

        Returns:
            Input current array of shape (n_neurons,).
        """
        signal = np.asarray(spectral_signal, dtype=np.float64).ravel()

        # Resize to n_neurons
        if len(signal) != self.config.n_neurons:
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, self.config.n_neurons)
            signal = np.interp(x_new, x_old, signal)

        if self.config.encoding == EncodingScheme.RATE_CODING:
            # Normalize to [0, 1] range as firing probability
            min_val = signal.min()
            max_val = signal.max()
            if max_val - min_val > 0:
                signal = (signal - min_val) / (max_val - min_val)
            return signal

        elif self.config.encoding == EncodingScheme.TEMPORAL_CODING:
            # Strong signals produce earlier spikes (inverted)
            max_val = np.abs(signal).max()
            if max_val > 0:
                signal = signal / max_val
            return signal

        elif self.config.encoding == EncodingScheme.PHASE_CODING:
            # Encode as phase offset
            return np.sin(2.0 * np.pi * signal / (signal.max() + 1e-10))

        elif self.config.encoding == EncodingScheme.POPULATION_CODING:
            # Spread across neuron populations with Gaussian tuning
            centers = np.linspace(signal.min(), signal.max(), self.config.n_neurons)
            mean_signal = np.mean(signal)
            return np.exp(-0.5 * ((centers - mean_signal) / 0.5) ** 2)

        return signal

    def process(self, spectral_signal: np.ndarray) -> List[SpikeEvent]:
        """Process a spectral signal through the neuromorphic layer.

        Encodes the signal, runs leaky-integrate-and-fire dynamics,
        and returns generated spike events.

        Args:
            spectral_signal: Input spectral signal.

        Returns:
            List of spike events generated during processing.
        """
        start = time.time()

        # Encode input
        input_current = self.encode(spectral_signal)

        # Leaky integrate-and-fire step
        spikes = self._lif_step(input_current)

        # STDP learning
        if self.config.enable_stdp and spikes:
            self._apply_stdp(spikes)

        # Record metrics
        elapsed = time.time() - start
        self._last_process_time = elapsed
        self._total_ops += self.config.n_neurons

        # Notify callbacks
        if spikes and self._callbacks:
            for cb in self._callbacks:
                cb(spikes)

        return spikes

    def process_stream(
        self, signal_stream: np.ndarray, window_size: int = 256
    ) -> List[List[SpikeEvent]]:
        """Process a continuous signal stream in windows.

        Args:
            signal_stream: Long signal array.
            window_size: Samples per processing window.

        Returns:
            List of spike event lists, one per window.
        """
        signal = np.asarray(signal_stream, dtype=np.float64).ravel()
        n_windows = max(1, len(signal) // window_size)
        all_spikes: List[List[SpikeEvent]] = []

        for i in range(n_windows):
            start = i * window_size
            window = signal[start:start + window_size]
            spikes = self.process(window)
            all_spikes.append(spikes)

        return all_spikes

    def _lif_step(self, input_current: np.ndarray) -> List[SpikeEvent]:
        """Run one leaky-integrate-and-fire step.

        Args:
            input_current: Input currents per neuron.

        Returns:
            List of spike events for neurons that fired.
        """
        # Decay membrane potential
        self._membrane_potential *= self.config.decay_rate

        # Apply refractory period (zero out refractory neurons)
        active_mask = self._refractory_state <= 0
        self._membrane_potential += input_current * active_mask

        # Add recurrent input from weights
        if self._total_spikes > 0:
            recurrent = self._weights @ (self._membrane_potential > 0).astype(float)
            self._membrane_potential += recurrent * 0.1 * active_mask

        # Check threshold crossings
        fired = self._membrane_potential >= self.config.threshold
        fired_indices = np.where(fired)[0]

        spikes: List[SpikeEvent] = []
        current_time = (time.time() - self._start_time) * 1000  # ms

        for idx in fired_indices:
            spike = SpikeEvent(
                neuron_id=int(idx),
                timestamp_ms=current_time,
                potential=float(self._membrane_potential[idx]),
            )
            spikes.append(spike)
            self._spike_history.append(spike)

        # Reset fired neurons
        self._membrane_potential[fired] = 0.0
        self._refractory_state[fired] = (
            self.config.refractory_period_ms / self.config.time_step_ms
        )

        # Decrement refractory counters
        self._refractory_state = np.maximum(0, self._refractory_state - 1)

        self._total_spikes += len(spikes)
        return spikes

    def _apply_stdp(self, spikes: List[SpikeEvent]) -> None:
        """Apply spike-timing-dependent plasticity to weights.

        Strengthens connections from pre to post-synaptic neurons
        that fire close together in time.

        Args:
            spikes: Recent spike events.
        """
        if len(self._spike_history) < 2:
            return

        # Simple STDP: strengthen connections between recently co-active neurons
        recent_spikes = list(self._spike_history)[-100:]
        fired_ids = [s.neuron_id for s in spikes]

        for spike in recent_spikes[-20:]:
            pre_id = spike.neuron_id
            for post_id in fired_ids:
                if pre_id != post_id:
                    dt = spikes[0].timestamp_ms - spike.timestamp_ms
                    if 0 < dt < 20:  # Pre before post
                        self._weights[pre_id, post_id] += 0.001
                    elif -20 < dt < 0:  # Post before pre
                        self._weights[pre_id, post_id] -= 0.0005

        # Clip weights
        np.clip(self._weights, -1.0, 1.0, out=self._weights)

    def on_spikes(self, callback: Callable[[List[SpikeEvent]], None]) -> None:
        """Register a callback for spike events.

        Args:
            callback: Function called with list of spikes.
        """
        self._callbacks.append(callback)

    def get_metrics(self) -> RuntimeMetrics:
        """Get current runtime performance metrics.

        Returns:
            RuntimeMetrics with current statistics.
        """
        uptime = time.time() - self._start_time
        ops_per_sec = self._total_ops / uptime if uptime > 0 else 0.0
        spike_rate = self._total_spikes / uptime if uptime > 0 else 0.0

        # Energy score based on processing mode
        energy_map = {
            ProcessingMode.ULTRA_LOW_POWER: 0.95,
            ProcessingMode.LOW_POWER: 0.85,
            ProcessingMode.BALANCED: 0.7,
            ProcessingMode.HIGH_PERFORMANCE: 0.4,
            ProcessingMode.BURST: 0.2,
        }

        active = int(np.sum(self._membrane_potential > 0))

        return RuntimeMetrics(
            ops_per_second=ops_per_sec,
            spike_rate_hz=spike_rate,
            energy_score=energy_map.get(self.config.processing_mode, 0.7),
            latency_ms=self._last_process_time * 1000,
            active_neurons=active,
            total_spikes=self._total_spikes,
            uptime_s=uptime,
        )

    def reset(self) -> None:
        """Reset runtime state (membrane potentials, history)."""
        self._membrane_potential = np.zeros(self.config.n_neurons)
        self._refractory_state = np.zeros(self.config.n_neurons)
        self._spike_history.clear()
        self._total_spikes = 0
        self._total_ops = 0
        self._start_time = time.time()

    @property
    def neuron_count(self) -> int:
        """Number of neurons in the runtime."""
        return self.config.n_neurons
