"""Teotl Energy Flow — monistic energy transformation between cosmological layers.

In Nahua (Aztec) metaphysics, Teotl is the single vivifying force that
continuously generates and regenerates all reality. There is no static
equilibrium — only transformation through sacrifice and renewal.

This module implements inter-layer energy flow dynamics:
- Energy sacrificed (consumed) in one layer feeds adjacent layers
- The Five Suns cycle models catastrophic renewal (spectral regime shifts)
- Balance is maintained through ongoing cost, not passive harmony
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesie.cosmology.layers import CosmicLayer, CosmicSpectralDecomposer, LayerDomain
from mesie.core.records import SpectralComponent


class SunEra(enum.Enum):
    """The Five Suns — cosmic epochs each ending in transformation.

    Each era represents a regime of spectral energy distribution that
    eventually collapses and gives rise to the next.
    """

    NAHUI_OCELOTL = "4-Jaguar"   # Earth/low-freq dominated, ends in devour
    NAHUI_EHECATL = "4-Wind"     # Dispersed energy, ends in hurricane
    NAHUI_QUIAHUITL = "4-Rain"   # Mid-frequency burst, ends in fire rain
    NAHUI_ATL = "4-Water"        # Fluid/broadband, ends in flood
    NAHUI_OLLIN = "4-Movement"   # Current era — balanced but unstable


@dataclass
class EnergyFlowState:
    """State of energy flow between cosmological layers.

    Attributes:
        layer_energies: Energy at each of the 22 layers.
        flow_rates: Net energy flow between adjacent layers (21 values).
        total_energy: Conservation check — total system energy.
        era: Current Sun era classification.
        stability: System stability metric (0=chaotic, 1=stable).
        sacrifice_total: Total energy sacrificed (transformed) this cycle.
    """

    layer_energies: np.ndarray
    flow_rates: np.ndarray
    total_energy: float
    era: SunEra
    stability: float
    sacrifice_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize flow state."""
        return {
            "layer_energies": self.layer_energies.tolist(),
            "flow_rates": self.flow_rates.tolist(),
            "total_energy": self.total_energy,
            "era": self.era.value,
            "stability": self.stability,
            "sacrifice_total": self.sacrifice_total,
        }


class TeotlEnergyFlow:
    """Models energy transformation between cosmological spectral layers.

    Implements the Teotl principle: energy is never created or destroyed,
    only transformed through sacrifice (cost) from one layer to adjacent
    layers. The system tends toward dynamic instability unless energy
    flows are actively maintained.

    Args:
        decomposer: CosmicSpectralDecomposer for layer structure.
        diffusion_rate: Base rate of energy diffusion between layers.
        sacrifice_ratio: Fraction of energy lost during transformation.
    """

    def __init__(
        self,
        decomposer: Optional[CosmicSpectralDecomposer] = None,
        diffusion_rate: float = 0.1,
        sacrifice_ratio: float = 0.05,
    ) -> None:
        self.decomposer = decomposer or CosmicSpectralDecomposer()
        self.diffusion_rate = diffusion_rate
        self.sacrifice_ratio = sacrifice_ratio
        self._state: Optional[EnergyFlowState] = None

    @property
    def state(self) -> Optional[EnergyFlowState]:
        """Current energy flow state."""
        return self._state

    def initialize_from_component(self, component: SpectralComponent) -> EnergyFlowState:
        """Initialize energy flow state from a spectral component.

        Args:
            component: Source spectral component.

        Returns:
            Initial EnergyFlowState.
        """
        energies = self.decomposer.energy_distribution(component)
        flows = np.zeros(len(energies) - 1)
        total = float(np.sum(energies))
        era = self._classify_era(energies)
        stability = self._compute_stability(energies)

        self._state = EnergyFlowState(
            layer_energies=energies,
            flow_rates=flows,
            total_energy=total,
            era=era,
            stability=stability,
        )
        return self._state

    def step(self, n_steps: int = 1) -> EnergyFlowState:
        """Advance the energy flow by n diffusion steps.

        Energy flows from high-energy layers to low-energy neighbors,
        with a sacrifice cost (energy lost to transformation).

        Args:
            n_steps: Number of diffusion steps.

        Returns:
            Updated EnergyFlowState.

        Raises:
            RuntimeError: If not initialized.
        """
        if self._state is None:
            raise RuntimeError("Must call initialize_from_component() first.")

        energies = self._state.layer_energies.copy()
        total_sacrifice = self._state.sacrifice_total

        for _ in range(n_steps):
            flows = np.zeros(len(energies) - 1)
            for i in range(len(flows)):
                # Energy flows from higher to lower concentration
                gradient = energies[i] - energies[i + 1]
                flow = gradient * self.diffusion_rate
                sacrifice = abs(flow) * self.sacrifice_ratio
                flows[i] = flow

                energies[i] -= flow
                energies[i + 1] += flow - sacrifice
                total_sacrifice += sacrifice

            # Enforce non-negativity (energy cannot go below zero)
            energies = np.maximum(energies, 0.0)

        total = float(np.sum(energies))
        era = self._classify_era(energies)
        stability = self._compute_stability(energies)

        self._state = EnergyFlowState(
            layer_energies=energies,
            flow_rates=flows,
            total_energy=total,
            era=era,
            stability=stability,
            sacrifice_total=total_sacrifice,
        )
        return self._state

    def inject_energy(self, layer_index: int, amount: float) -> None:
        """Inject energy into a specific layer (external input / renewal).

        Args:
            layer_index: Target layer (0-21).
            amount: Energy to inject.

        Raises:
            RuntimeError: If not initialized.
            ValueError: If layer index out of range.
        """
        if self._state is None:
            raise RuntimeError("Must call initialize_from_component() first.")
        if layer_index < 0 or layer_index >= len(self._state.layer_energies):
            raise ValueError(f"Layer index must be 0-{len(self._state.layer_energies) - 1}")
        self._state.layer_energies[layer_index] += amount
        self._state.total_energy += amount

    def catastrophic_renewal(self) -> EnergyFlowState:
        """Trigger a Five Suns catastrophic renewal event.

        Redistributes all energy uniformly across layers — modeling the
        destruction of a world-era and birth of the next. Total energy
        is conserved minus sacrifice cost.

        Returns:
            New EnergyFlowState after renewal.

        Raises:
            RuntimeError: If not initialized.
        """
        if self._state is None:
            raise RuntimeError("Must call initialize_from_component() first.")

        total = self._state.total_energy
        sacrifice = total * self.sacrifice_ratio
        remaining = total - sacrifice
        n_layers = len(self._state.layer_energies)

        # Uniform redistribution — new era begins
        new_energies = np.full(n_layers, remaining / n_layers)
        flows = np.zeros(n_layers - 1)

        era = self._classify_era(new_energies)
        self._state = EnergyFlowState(
            layer_energies=new_energies,
            flow_rates=flows,
            total_energy=remaining,
            era=era,
            stability=1.0,  # Perfect stability after renewal
            sacrifice_total=self._state.sacrifice_total + sacrifice,
        )
        return self._state

    def _classify_era(self, energies: np.ndarray) -> SunEra:
        """Classify the current energy distribution into a Sun era."""
        n = len(energies)
        underworld = energies[:9]
        heaven = energies[9:]

        uw_ratio = float(np.sum(underworld)) / max(float(np.sum(energies)), 1e-12)
        hv_ratio = float(np.sum(heaven)) / max(float(np.sum(energies)), 1e-12)
        spread = float(np.std(energies)) / max(float(np.mean(energies)), 1e-12)

        if uw_ratio > 0.7:
            return SunEra.NAHUI_OCELOTL  # Low-freq (Jaguar) dominated
        elif spread > 2.0:
            return SunEra.NAHUI_EHECATL  # Highly dispersed (Wind)
        elif hv_ratio > 0.7:
            return SunEra.NAHUI_QUIAHUITL  # High-freq dominated (Rain of fire)
        elif spread < 0.3:
            return SunEra.NAHUI_ATL  # Uniform/fluid (Water)
        else:
            return SunEra.NAHUI_OLLIN  # Balanced/unstable (Movement)

    def _compute_stability(self, energies: np.ndarray) -> float:
        """Compute system stability (0=chaotic, 1=perfectly balanced)."""
        if len(energies) == 0:
            return 1.0
        mean_e = float(np.mean(energies))
        if mean_e <= 0:
            return 1.0
        cv = float(np.std(energies)) / mean_e  # coefficient of variation
        # Map CV to stability: CV=0 → stability=1, CV≥2 → stability≈0
        return float(np.exp(-cv))
