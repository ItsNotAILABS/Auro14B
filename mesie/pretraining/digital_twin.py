"""Digital twin simulation environments for agent-level spectral pretraining.

Simulates environments (factories, bridges, robots, power systems) where
entities emit spectral streams over time. Each entity gets its own spectral
encoder + memory, and agents are trained with RL/IL reward functions tied
to spectral reasoning objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class EntityType(Enum):
    """Types of simulated entities in digital twin environments."""

    ROTATING_MACHINERY = "rotating_machinery"
    STRUCTURAL_ELEMENT = "structural_element"
    POWER_SYSTEM = "power_system"
    ROBOTIC_JOINT = "robotic_joint"
    FLUID_SYSTEM = "fluid_system"


@dataclass
class SpectralStream:
    """A time-indexed stream of spectral observations from an entity.

    Attributes
    ----------
    entity_id : str
        Identifier of the emitting entity.
    timestamps : ndarray, shape (n_timesteps,)
        Time indices for each observation.
    frequencies : ndarray, shape (n_freq,)
        Frequency axis (shared across all timesteps).
    amplitudes : ndarray, shape (n_timesteps, n_freq)
        Amplitude spectra over time.
    phases : ndarray or None, shape (n_timesteps, n_freq)
        Phase spectra over time (if available).
    events : list of tuple
        List of (timestamp, event_type, metadata) for notable events.
    """

    entity_id: str
    timestamps: np.ndarray
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: Optional[np.ndarray] = None
    events: List[Tuple[float, str, Dict]] = field(default_factory=list)

    @property
    def n_timesteps(self) -> int:
        return self.amplitudes.shape[0]

    @property
    def n_freq(self) -> int:
        return self.frequencies.shape[0]

    def get_window(self, start: int, end: int) -> "SpectralStream":
        """Extract a time window from the stream."""
        return SpectralStream(
            entity_id=self.entity_id,
            timestamps=self.timestamps[start:end],
            frequencies=self.frequencies,
            amplitudes=self.amplitudes[start:end],
            phases=self.phases[start:end] if self.phases is not None else None,
            events=[
                (t, etype, meta)
                for t, etype, meta in self.events
                if start <= t < end
            ],
        )


@dataclass
class SpectralEntity:
    """A simulated entity that emits spectral data over time.

    Attributes
    ----------
    entity_id : str
        Unique identifier.
    entity_type : EntityType
        Type of physical system being simulated.
    natural_frequencies : ndarray
        Natural frequencies (resonances) of the entity.
    damping_ratios : ndarray
        Damping ratios corresponding to each natural frequency.
    base_amplitude : float
        Baseline amplitude level.
    noise_level : float
        Ambient noise floor level.
    drift_rate : float
        Rate of spectral drift (distributional shift per timestep).
    """

    entity_id: str
    entity_type: EntityType
    natural_frequencies: np.ndarray
    damping_ratios: np.ndarray
    base_amplitude: float = 1.0
    noise_level: float = 0.01
    drift_rate: float = 0.001

    def generate_spectrum(
        self,
        frequencies: np.ndarray,
        time_step: int = 0,
        excitation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a single spectral observation at a given time step.

        Parameters
        ----------
        frequencies : ndarray, shape (n_freq,)
            Frequency axis.
        time_step : int
            Current time step (affects drift).
        excitation : ndarray or None, shape (n_freq,)
            External excitation spectrum (optional).

        Returns
        -------
        ndarray, shape (n_freq,)
            Generated amplitude spectrum.
        """
        spectrum = np.ones_like(frequencies) * self.noise_level

        # Add resonance peaks with time-dependent drift
        for i, (fn, zeta) in enumerate(
            zip(self.natural_frequencies, self.damping_ratios)
        ):
            # Apply slow drift to natural frequency
            fn_drifted = fn * (1.0 + self.drift_rate * time_step)
            # Frequency response function magnitude
            r = frequencies / fn_drifted
            denominator = np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
            denominator = np.maximum(denominator, 1e-10)
            response = self.base_amplitude / denominator
            spectrum += response

        # Add excitation if provided
        if excitation is not None:
            spectrum *= (1.0 + excitation)

        # Add stochastic noise
        spectrum += np.random.randn(len(frequencies)) * self.noise_level

        return np.maximum(spectrum, 0.0)


@dataclass
class RewardSignals:
    """Reward signals for RL/IL training of spectral agents.

    Attributes
    ----------
    resonance_avoidance : float
        Reward for staying away from resonance conditions.
    drift_minimization : float
        Reward for maintaining stable spectral characteristics.
    coherence_maintenance : float
        Reward for maintaining inter-component coherence.
    anomaly_detection : float
        Reward for early detection of anomalous conditions.
    total : float
        Weighted sum of all reward components.
    """

    resonance_avoidance: float = 0.0
    drift_minimization: float = 0.0
    coherence_maintenance: float = 0.0
    anomaly_detection: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.resonance_avoidance
            + self.drift_minimization
            + self.coherence_maintenance
            + self.anomaly_detection
        )


class DigitalTwinEnvironment:
    """Simulated environment containing multiple spectral entities.

    Generates spectral streams from simulated physical systems for
    agent-level pretraining. Provides step-based interaction with
    reward signals tied to spectral reasoning objectives.

    Parameters
    ----------
    frequencies : ndarray, shape (n_freq,)
        Shared frequency axis for all entities.
    entities : list of SpectralEntity
        Entities in the environment.
    episode_length : int
        Maximum number of time steps per episode.
    reward_weights : dict or None
        Weights for different reward components.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        entities: Optional[List[SpectralEntity]] = None,
        episode_length: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.frequencies = frequencies
        self.entities = entities or []
        self.episode_length = episode_length
        self.reward_weights = reward_weights or {
            "resonance_avoidance": 1.0,
            "drift_minimization": 1.0,
            "coherence_maintenance": 0.5,
            "anomaly_detection": 2.0,
        }

        # Internal state
        self._time_step: int = 0
        self._history: Dict[str, List[np.ndarray]] = {}
        self._baseline_spectra: Dict[str, np.ndarray] = {}
        self._done: bool = False

    def add_entity(self, entity: SpectralEntity) -> None:
        """Add an entity to the environment."""
        self.entities.append(entity)

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment to initial state.

        Returns
        -------
        dict mapping entity_id -> initial spectrum observation
        """
        self._time_step = 0
        self._history = {e.entity_id: [] for e in self.entities}
        self._done = False

        observations = {}
        for entity in self.entities:
            spectrum = entity.generate_spectrum(self.frequencies, time_step=0)
            observations[entity.entity_id] = spectrum
            self._history[entity.entity_id].append(spectrum)
            self._baseline_spectra[entity.entity_id] = spectrum.copy()

        return observations

    def step(
        self, actions: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[Dict[str, np.ndarray], RewardSignals, bool, Dict]:
        """Advance the environment by one time step.

        Parameters
        ----------
        actions : dict or None
            Optional actions (excitations) to apply to entities.
            Maps entity_id -> excitation spectrum.

        Returns
        -------
        observations : dict mapping entity_id -> spectrum
        rewards : RewardSignals
        done : bool
        info : dict with additional environment information
        """
        self._time_step += 1

        if self._time_step >= self.episode_length:
            self._done = True

        observations = {}
        for entity in self.entities:
            excitation = (
                actions.get(entity.entity_id) if actions else None
            )
            spectrum = entity.generate_spectrum(
                self.frequencies, time_step=self._time_step, excitation=excitation
            )
            observations[entity.entity_id] = spectrum
            self._history[entity.entity_id].append(spectrum)

        # Compute rewards
        rewards = self._compute_rewards(observations)

        info = {
            "time_step": self._time_step,
            "n_entities": len(self.entities),
        }

        return observations, rewards, self._done, info

    def _compute_rewards(
        self, observations: Dict[str, np.ndarray]
    ) -> RewardSignals:
        """Compute reward signals from current observations."""
        resonance_reward = 0.0
        drift_reward = 0.0
        coherence_reward = 0.0
        anomaly_reward = 0.0

        for entity in self.entities:
            eid = entity.entity_id
            spectrum = observations[eid]
            baseline = self._baseline_spectra.get(eid)

            # Resonance avoidance: penalize high peak-to-mean ratios
            mean_amp = np.mean(spectrum)
            if mean_amp > 0:
                peak_ratio = np.max(spectrum) / mean_amp
                resonance_reward += max(0, 1.0 - peak_ratio / 10.0)

            # Drift minimization: penalize deviation from baseline
            if baseline is not None:
                drift = np.sqrt(np.mean((spectrum - baseline) ** 2))
                drift_reward += max(0, 1.0 - drift)

            # Anomaly detection: reward detecting sudden changes
            history = self._history[eid]
            if len(history) >= 2:
                prev = history[-2]
                change = np.sqrt(np.mean((spectrum - prev) ** 2))
                # Small reward for stability, larger reward for detecting anomalies
                if change > 0.5:  # Anomalous change threshold
                    anomaly_reward += 0.5
                else:
                    anomaly_reward += 0.1

        # Coherence maintenance across entities
        if len(self.entities) > 1:
            spectra = np.array([observations[e.entity_id] for e in self.entities])
            mean_correlation = 0.0
            count = 0
            for i in range(len(self.entities)):
                for j in range(i + 1, len(self.entities)):
                    corr = np.corrcoef(spectra[i], spectra[j])[0, 1]
                    mean_correlation += abs(corr) if not np.isnan(corr) else 0.0
                    count += 1
            if count > 0:
                coherence_reward = mean_correlation / count

        n = max(len(self.entities), 1)
        return RewardSignals(
            resonance_avoidance=resonance_reward / n * self.reward_weights["resonance_avoidance"],
            drift_minimization=drift_reward / n * self.reward_weights["drift_minimization"],
            coherence_maintenance=coherence_reward * self.reward_weights["coherence_maintenance"],
            anomaly_detection=anomaly_reward / n * self.reward_weights["anomaly_detection"],
        )

    def generate_stream(
        self, entity_id: str, n_timesteps: Optional[int] = None
    ) -> SpectralStream:
        """Generate a complete spectral stream for an entity.

        Parameters
        ----------
        entity_id : str
            Entity to generate stream for.
        n_timesteps : int or None
            Number of timesteps. Defaults to episode_length.

        Returns
        -------
        SpectralStream
            Complete time-indexed spectral stream.
        """
        n = n_timesteps or self.episode_length
        entity = next(
            (e for e in self.entities if e.entity_id == entity_id), None
        )
        if entity is None:
            raise ValueError(f"Entity '{entity_id}' not found in environment.")

        timestamps = np.arange(n, dtype=np.float64)
        amplitudes = np.zeros((n, len(self.frequencies)))
        events: List[Tuple[float, str, Dict]] = []

        for t in range(n):
            amplitudes[t] = entity.generate_spectrum(
                self.frequencies, time_step=t
            )
            # Check for resonance events
            mean_amp = np.mean(amplitudes[t])
            if mean_amp > 0 and np.max(amplitudes[t]) / mean_amp > 8.0:
                events.append((float(t), "resonance", {"peak_ratio": float(np.max(amplitudes[t]) / mean_amp)}))

        return SpectralStream(
            entity_id=entity_id,
            timestamps=timestamps,
            frequencies=self.frequencies,
            amplitudes=amplitudes,
            events=events,
        )

    @classmethod
    def create_factory_environment(
        cls, n_machines: int = 5, n_freq: int = 256
    ) -> "DigitalTwinEnvironment":
        """Create a factory simulation with rotating machinery.

        Parameters
        ----------
        n_machines : int
            Number of machines to simulate.
        n_freq : int
            Number of frequency bins.

        Returns
        -------
        DigitalTwinEnvironment
            Configured environment.
        """
        frequencies = np.linspace(0.1, 100.0, n_freq)
        entities = []

        for i in range(n_machines):
            # Each machine has different natural frequencies
            rng = np.random.default_rng(seed=42 + i)
            n_modes = rng.integers(2, 6)
            nat_freqs = np.sort(rng.uniform(5, 80, size=n_modes))
            damping = rng.uniform(0.01, 0.1, size=n_modes)

            entities.append(
                SpectralEntity(
                    entity_id=f"machine_{i}",
                    entity_type=EntityType.ROTATING_MACHINERY,
                    natural_frequencies=nat_freqs,
                    damping_ratios=damping,
                    base_amplitude=rng.uniform(0.5, 2.0),
                    noise_level=rng.uniform(0.005, 0.02),
                    drift_rate=rng.uniform(0.0001, 0.005),
                )
            )

        return cls(frequencies=frequencies, entities=entities)

    @classmethod
    def create_bridge_environment(
        cls, n_sensors: int = 8, n_freq: int = 512
    ) -> "DigitalTwinEnvironment":
        """Create a bridge structural health monitoring simulation.

        Parameters
        ----------
        n_sensors : int
            Number of vibration sensors.
        n_freq : int
            Number of frequency bins.

        Returns
        -------
        DigitalTwinEnvironment
            Configured environment.
        """
        frequencies = np.linspace(0.01, 50.0, n_freq)
        entities = []

        for i in range(n_sensors):
            rng = np.random.default_rng(seed=100 + i)
            # Bridge modes are typically low frequency
            n_modes = rng.integers(3, 8)
            nat_freqs = np.sort(rng.uniform(0.5, 20, size=n_modes))
            damping = rng.uniform(0.005, 0.05, size=n_modes)

            entities.append(
                SpectralEntity(
                    entity_id=f"sensor_{i}",
                    entity_type=EntityType.STRUCTURAL_ELEMENT,
                    natural_frequencies=nat_freqs,
                    damping_ratios=damping,
                    base_amplitude=rng.uniform(0.3, 1.0),
                    noise_level=rng.uniform(0.001, 0.01),
                    drift_rate=rng.uniform(0.00001, 0.001),
                )
            )

        return cls(frequencies=frequencies, entities=entities)
