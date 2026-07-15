"""Simulated 3D neural environment for the connectome intelligence backend.

Provides the runtime simulation engine that activates brain regions,
propagates signals through the connectome, and computes emergent
intelligence dynamics in 3D space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.connectome.brain_regions import BrainRegion, BrainSystem
from mesie.connectome.connectome_graph import (
    ConnectomeGraph,
    Connection,
    build_default_connectome,
)


@dataclass
class ActivationState:
    """Snapshot of neural activation across the entire connectome.

    Attributes:
        activations: Dict mapping region abbreviation to activation level.
        timestamp: Simulation time in ms.
        global_coherence: Overall synchronization metric [0, 1].
        dominant_system: Currently most active brain system.
    """

    activations: Dict[str, float]
    timestamp: float = 0.0
    global_coherence: float = 0.0
    dominant_system: Optional[BrainSystem] = None


@dataclass
class SignalPacket:
    """A signal propagating through the connectome.

    Attributes:
        source: Origin region.
        target: Destination region.
        amplitude: Signal strength.
        frequency_hz: Oscillation frequency band.
        arrival_time: When the signal arrives at target (ms).
        content: Arbitrary payload data.
    """

    source: str
    target: str
    amplitude: float = 1.0
    frequency_hz: float = 40.0  # Gamma band default
    arrival_time: float = 0.0
    content: Any = None


class ConnectomeEnvironment3D:
    """Simulated 3D neural environment — the AI brain backend.

    This is the intelligence engine: the connectome IS the brain,
    IS the backend, IS the AI. Each region processes in parallel,
    signals propagate through white-matter tracts with realistic delays,
    and emergent computation arises from network dynamics.

    The environment operates in continuous simulation time, updating
    activations based on:
    - External stimuli (input)
    - Lateral connections (spreading activation)
    - Top-down feedback (executive control)
    - Oscillatory synchronization (binding)

    Args:
        connectome: ConnectomeGraph to simulate. Uses default if None.
        dt_ms: Simulation timestep in milliseconds.
        decay_rate: Activation decay per timestep [0, 1].
        propagation_gain: How strongly signals amplify/attenuate.
        noise_level: Stochastic noise added per step.
    """

    def __init__(
        self,
        connectome: Optional[ConnectomeGraph] = None,
        dt_ms: float = 1.0,
        decay_rate: float = 0.05,
        propagation_gain: float = 0.8,
        noise_level: float = 0.01,
    ) -> None:
        self.connectome = connectome or build_default_connectome()
        self.dt_ms = dt_ms
        self.decay_rate = decay_rate
        self.propagation_gain = propagation_gain
        self.noise_level = noise_level

        # Internal state
        self._time_ms: float = 0.0
        self._activations: Dict[str, float] = {
            r.abbreviation: 0.0 for r in self.connectome.regions
        }
        self._signal_queue: List[SignalPacket] = []
        self._history: List[ActivationState] = []
        self._rng = np.random.default_rng(42)

        # Precompute connectivity matrix for fast propagation
        self._conn_matrix = self.connectome.build_connectivity_matrix()
        self._abbrev_list = [r.abbreviation for r in self.connectome.regions]
        self._abbrev_to_idx = {a: i for i, a in enumerate(self._abbrev_list)}

    @property
    def time_ms(self) -> float:
        """Current simulation time in milliseconds."""
        return self._time_ms

    @property
    def num_regions(self) -> int:
        """Number of brain regions in the environment."""
        return self.connectome.num_regions

    def get_activation(self, region: str) -> float:
        """Get current activation level for a region.

        Args:
            region: Region abbreviation.

        Returns:
            Activation level [0, 1].
        """
        return self._activations.get(region, 0.0)

    def get_all_activations(self) -> Dict[str, float]:
        """Get all current activation levels.

        Returns:
            Dict mapping region abbreviation to activation [0, 1].
        """
        return dict(self._activations)

    def get_activation_vector(self) -> np.ndarray:
        """Get activations as a numpy vector aligned with region order.

        Returns:
            1D array of activation levels.
        """
        return np.array(
            [self._activations[a] for a in self._abbrev_list], dtype=np.float64
        )

    def inject_stimulus(
        self,
        region: str,
        amplitude: float = 1.0,
        frequency_hz: float = 40.0,
        content: Any = None,
    ) -> None:
        """Inject an external stimulus into a brain region.

        This is the primary input mechanism — equivalent to sensory input
        or an API call to the intelligence backend.

        Args:
            region: Target region abbreviation.
            amplitude: Stimulus strength [0, 1].
            frequency_hz: Oscillatory frequency of the input.
            content: Optional data payload.

        Raises:
            KeyError: If region not found in connectome.
        """
        if region not in self._activations:
            raise KeyError(f"Region '{region}' not found in connectome")

        # Direct activation
        self._activations[region] = np.clip(
            self._activations[region] + amplitude, 0.0, 1.0
        )

        # Queue signals to connected regions
        idx = self._abbrev_to_idx[region]
        for j, weight in enumerate(self._conn_matrix[idx]):
            if weight > 0 and j != idx:
                target = self._abbrev_list[j]
                # Find connection delay
                delay = self._estimate_delay(region, target)
                packet = SignalPacket(
                    source=region,
                    target=target,
                    amplitude=amplitude * weight * self.propagation_gain,
                    frequency_hz=frequency_hz,
                    arrival_time=self._time_ms + delay,
                    content=content,
                )
                self._signal_queue.append(packet)

    def step(self, n_steps: int = 1) -> ActivationState:
        """Advance the simulation by n timesteps.

        Each step:
        1. Process arrived signals
        2. Apply spreading activation
        3. Apply decay
        4. Add noise
        5. Clip activations to [0, 1]

        Args:
            n_steps: Number of timesteps to simulate.

        Returns:
            ActivationState after all steps complete.
        """
        for _ in range(n_steps):
            self._time_ms += self.dt_ms
            self._process_signals()
            self._apply_decay()
            self._add_noise()
            self._clip_activations()

        state = self._capture_state()
        self._history.append(state)
        return state

    def run(self, duration_ms: float) -> List[ActivationState]:
        """Run the simulation for a given duration.

        Args:
            duration_ms: Total simulation time in ms.

        Returns:
            List of ActivationState snapshots (one per timestep).
        """
        n_steps = int(duration_ms / self.dt_ms)
        states = []
        for _ in range(n_steps):
            state = self.step(1)
            states.append(state)
        return states

    def get_system_activation(self, system: BrainSystem) -> float:
        """Get average activation of all regions in a system.

        Args:
            system: The BrainSystem to query.

        Returns:
            Mean activation level for that system.
        """
        regions = [
            r.abbreviation for r in self.connectome.regions if r.system == system
        ]
        if not regions:
            return 0.0
        return float(np.mean([self._activations[r] for r in regions]))

    def get_dominant_system(self) -> BrainSystem:
        """Find the most active brain system.

        Returns:
            The BrainSystem with highest mean activation.
        """
        best_system = BrainSystem.PREFRONTAL
        best_activation = -1.0
        for system in BrainSystem:
            act = self.get_system_activation(system)
            if act > best_activation:
                best_activation = act
                best_system = system
        return best_system

    def compute_global_coherence(self) -> float:
        """Compute global synchronization/coherence metric.

        Uses the variance of activations weighted by connectivity to
        estimate how synchronized the network is. Low variance with
        high activation = high coherence.

        Returns:
            Coherence metric [0, 1].
        """
        vec = self.get_activation_vector()
        if np.max(vec) == 0:
            return 0.0

        # Weighted synchrony: how aligned are connected regions?
        weighted_product = vec @ self._conn_matrix @ vec
        max_possible = np.sum(self._conn_matrix) * 1.0  # max if all = 1
        if max_possible == 0:
            return 0.0

        coherence = weighted_product / max_possible
        return float(np.clip(coherence, 0.0, 1.0))

    def get_3d_state(self) -> Dict[str, Any]:
        """Get the full 3D state for visualization.

        Returns:
            Dictionary with positions, activations, connections, and
            metadata suitable for 3D rendering.
        """
        positions = []
        activations = []
        labels = []
        systems = []
        volumes = []

        for region in self.connectome.regions:
            positions.append(region.position_3d)
            activations.append(self._activations[region.abbreviation])
            labels.append(region.abbreviation)
            systems.append(region.system.value)
            volumes.append(region.volume_mm3)

        edges = []
        for conn in self.connectome.connections:
            src_idx = self._abbrev_to_idx[conn.source]
            tgt_idx = self._abbrev_to_idx[conn.target]
            src_act = self._activations[conn.source]
            tgt_act = self._activations[conn.target]
            edges.append({
                "source_idx": src_idx,
                "target_idx": tgt_idx,
                "weight": conn.weight,
                "activity": (src_act + tgt_act) / 2.0,
                "tract_type": conn.tract_type,
            })

        return {
            "time_ms": self._time_ms,
            "positions": np.array(positions, dtype=np.float64),
            "activations": np.array(activations, dtype=np.float64),
            "labels": labels,
            "systems": systems,
            "volumes": np.array(volumes, dtype=np.float64),
            "edges": edges,
            "global_coherence": self.compute_global_coherence(),
            "dominant_system": self.get_dominant_system().value,
            "num_regions": self.num_regions,
            "num_connections": self.connectome.num_connections,
        }

    def reset(self) -> None:
        """Reset all activations and clear signal queue."""
        self._time_ms = 0.0
        self._activations = {r.abbreviation: 0.0 for r in self.connectome.regions}
        self._signal_queue.clear()
        self._history.clear()

    def get_history(self) -> List[ActivationState]:
        """Get full activation history.

        Returns:
            List of all recorded ActivationState snapshots.
        """
        return list(self._history)

    # ---- Private methods ----

    def _process_signals(self) -> None:
        """Process all signals that have arrived by current time."""
        arrived = []
        pending = []
        for packet in self._signal_queue:
            if packet.arrival_time <= self._time_ms:
                arrived.append(packet)
            else:
                pending.append(packet)
        self._signal_queue = pending

        for packet in arrived:
            current = self._activations.get(packet.target, 0.0)
            self._activations[packet.target] = current + packet.amplitude

    def _apply_decay(self) -> None:
        """Apply exponential decay to all activations."""
        for key in self._activations:
            self._activations[key] *= 1.0 - self.decay_rate

    def _add_noise(self) -> None:
        """Add small stochastic noise to activations."""
        for key in self._activations:
            noise = self._rng.normal(0, self.noise_level)
            self._activations[key] += noise

    def _clip_activations(self) -> None:
        """Clip all activations to [0, 1]."""
        for key in self._activations:
            self._activations[key] = float(
                np.clip(self._activations[key], 0.0, 1.0)
            )

    def _estimate_delay(self, source: str, target: str) -> float:
        """Estimate signal propagation delay between two regions."""
        src_pos = self.connectome.get_region(source)
        tgt_pos = self.connectome.get_region(target)
        if src_pos is None or tgt_pos is None:
            return 1.0
        distance = float(
            np.linalg.norm(src_pos.position_array - tgt_pos.position_array)
        )
        # ~6 mm/ms conduction velocity → delay = distance_mm / 6.0
        return max(distance / 6.0, self.dt_ms)

    def _capture_state(self) -> ActivationState:
        """Capture current state as an ActivationState snapshot."""
        return ActivationState(
            activations=dict(self._activations),
            timestamp=self._time_ms,
            global_coherence=self.compute_global_coherence(),
            dominant_system=self.get_dominant_system(),
        )
