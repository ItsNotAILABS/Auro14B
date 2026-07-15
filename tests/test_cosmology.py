"""Tests for the cosmology module — layers, token governor, and Teotl energy flow."""

import numpy as np
import pytest

from mesie.core.records import SpectralComponent
from mesie.cosmology.layers import (
    CosmicLayer,
    CosmicSpectralDecomposer,
    LayerDomain,
)
from mesie.cosmology.token_governor import (
    CalendricalTokenGovernor,
    TokenBudget,
)
from mesie.cosmology.teotl_flow import (
    EnergyFlowState,
    SunEra,
    TeotlEnergyFlow,
)


# --- Fixtures ---

@pytest.fixture
def broadband_component():
    """Broadband signal spanning the full frequency range."""
    freq = np.logspace(-2, 2, 500)
    amp = np.random.default_rng(42).exponential(1.0, 500)
    return SpectralComponent(name="broadband", frequency=freq, amplitude=amp)


@pytest.fixture
def low_freq_component():
    """Low-frequency dominated signal."""
    freq = np.logspace(-2, 2, 500)
    amp = 10.0 / (1.0 + freq)  # 1/f-like, low-freq heavy
    return SpectralComponent(name="low_freq", frequency=freq, amplitude=amp)


@pytest.fixture
def high_freq_component():
    """High-frequency dominated signal."""
    freq = np.logspace(-2, 2, 500)
    amp = freq / 100.0  # Rising with frequency
    return SpectralComponent(name="high_freq", frequency=freq, amplitude=amp)


# --- CosmicSpectralDecomposer Tests ---

class TestCosmicSpectralDecomposer:
    def test_layer_count(self):
        decomposer = CosmicSpectralDecomposer()
        assert len(decomposer.layers) == 22

    def test_layer_domains(self):
        decomposer = CosmicSpectralDecomposer()
        underworlds = [l for l in decomposer.layers if l.domain == LayerDomain.UNDERWORLD]
        heavens = [l for l in decomposer.layers if l.domain == LayerDomain.HEAVEN]
        assert len(underworlds) == 9
        assert len(heavens) == 13

    def test_layer_edges_monotonic(self):
        decomposer = CosmicSpectralDecomposer()
        edges = decomposer.layer_edges
        assert np.all(np.diff(edges) > 0)

    def test_decompose_broadband(self, broadband_component):
        decomposer = CosmicSpectralDecomposer()
        layers = decomposer.decompose(broadband_component)
        assert len(layers) == 22
        # At least some layers should have energy
        energies = [l.energy for l in layers]
        assert sum(energies) > 0

    def test_energy_distribution_shape(self, broadband_component):
        decomposer = CosmicSpectralDecomposer()
        dist = decomposer.energy_distribution(broadband_component)
        assert dist.shape == (22,)
        assert np.all(dist >= 0)

    def test_cosmic_balance_low_freq(self, low_freq_component):
        decomposer = CosmicSpectralDecomposer()
        balance = decomposer.cosmic_balance(low_freq_component)
        # Low-freq dominated signal should have balance < 1
        assert balance < 1.0

    def test_cosmic_balance_high_freq(self, high_freq_component):
        decomposer = CosmicSpectralDecomposer()
        balance = decomposer.cosmic_balance(high_freq_component)
        # High-freq dominated signal should have balance > 1
        assert balance > 1.0

    def test_layer_similarity_identical(self, broadband_component):
        decomposer = CosmicSpectralDecomposer()
        sim = decomposer.layer_similarity(broadband_component, broadband_component)
        assert sim.shape == (22,)
        # Identical signals should have similarity 1 where there's energy
        populated = sim[sim > 0]
        assert np.all(populated > 0.99)

    def test_linear_scale(self, broadband_component):
        decomposer = CosmicSpectralDecomposer(log_scale=False)
        layers = decomposer.decompose(broadband_component)
        assert len(layers) == 22


# --- CalendricalTokenGovernor Tests ---

class TestCalendricalTokenGovernor:
    def test_begin_cycle(self):
        gov = CalendricalTokenGovernor(default_budget=100.0)
        budget = gov.begin_cycle()
        assert budget.total_tokens == 100.0
        assert budget.remaining_tokens == 100.0
        assert budget.is_active

    def test_spend_tokens(self):
        gov = CalendricalTokenGovernor(default_budget=100.0)
        gov.begin_cycle()
        exp = gov.spend("matching", 25.0, quality=0.8)
        assert exp.tokens_spent == 25.0
        assert gov.current_cycle.remaining_tokens == 75.0

    def test_budget_exhaustion(self):
        gov = CalendricalTokenGovernor(default_budget=50.0)
        gov.begin_cycle()
        gov.spend("op1", 50.0)
        assert gov.current_cycle.is_exhausted
        with pytest.raises(RuntimeError):
            gov.spend("op2", 1.0)

    def test_governed_operation(self):
        gov = CalendricalTokenGovernor(default_budget=100.0)
        gov.begin_cycle()
        result = gov.governed_operation("add", 10.0, lambda: 42)
        assert result == 42
        assert gov.current_cycle.spent_tokens == 10.0

    def test_governed_operation_budget_refused(self):
        gov = CalendricalTokenGovernor(default_budget=5.0)
        gov.begin_cycle()
        gov.spend("warmup", 5.0, quality=1.0)
        result = gov.governed_operation("refused", 10.0, lambda: 99)
        assert result is None

    def test_cycle_history(self):
        gov = CalendricalTokenGovernor(default_budget=100.0)
        gov.begin_cycle(name="Tun-A")
        gov.spend("op", 50.0, quality=0.9)
        gov.end_cycle()
        gov.begin_cycle(name="Tun-B")
        gov.spend("op", 30.0, quality=0.95)
        gov.end_cycle()
        assert gov.cycle_count == 2
        assert len(gov.history) == 2
        assert gov.history[0].cycle_name == "Tun-A"

    def test_convergence(self):
        gov = CalendricalTokenGovernor(default_budget=100.0, convergence_threshold=0.9)
        budget = gov.begin_cycle()
        gov.spend("op1", 10.0, quality=0.95)
        gov.spend("op2", 10.0, quality=0.92)
        assert budget.has_converged

    def test_summary(self):
        gov = CalendricalTokenGovernor(default_budget=100.0)
        gov.begin_cycle()
        gov.spend("op", 20.0, quality=0.7)
        gov.end_cycle()
        summary = gov.summary()
        assert summary["cycles_completed"] == 1
        assert summary["total_tokens_spent"] == 20.0

    def test_adaptive_budget_increase(self):
        gov = CalendricalTokenGovernor(default_budget=100.0, adaptive=True)
        # Simulate cycles that fail to converge
        for _ in range(5):
            gov.begin_cycle()
            gov.spend("op", 100.0, quality=0.2)  # Low quality
        gov.end_cycle()
        # Next cycle should get increased budget
        budget = gov.begin_cycle()
        assert budget.total_tokens > 100.0


# --- TeotlEnergyFlow Tests ---

class TestTeotlEnergyFlow:
    def test_initialize(self, broadband_component):
        flow = TeotlEnergyFlow()
        state = flow.initialize_from_component(broadband_component)
        assert state.layer_energies.shape == (22,)
        assert state.total_energy > 0
        assert 0.0 <= state.stability <= 1.0
        assert isinstance(state.era, SunEra)

    def test_step_conserves_energy_approx(self, broadband_component):
        flow = TeotlEnergyFlow(sacrifice_ratio=0.0)  # No sacrifice for conservation test
        state = flow.initialize_from_component(broadband_component)
        initial_total = state.total_energy
        state = flow.step(n_steps=10)
        # Without sacrifice, energy should be approximately conserved
        assert abs(state.total_energy - initial_total) < initial_total * 0.01

    def test_step_with_sacrifice_reduces_energy(self, broadband_component):
        flow = TeotlEnergyFlow(sacrifice_ratio=0.1)
        state = flow.initialize_from_component(broadband_component)
        initial_total = state.total_energy
        state = flow.step(n_steps=10)
        # With sacrifice, total energy should decrease
        assert state.total_energy < initial_total
        assert state.sacrifice_total > 0

    def test_inject_energy(self, broadband_component):
        flow = TeotlEnergyFlow()
        flow.initialize_from_component(broadband_component)
        initial = flow.state.layer_energies[5]
        flow.inject_energy(5, 100.0)
        assert flow.state.layer_energies[5] == initial + 100.0

    def test_catastrophic_renewal(self, broadband_component):
        flow = TeotlEnergyFlow()
        flow.initialize_from_component(broadband_component)
        state = flow.catastrophic_renewal()
        # After renewal, energy should be uniform
        assert state.stability == 1.0
        energies = state.layer_energies
        assert np.allclose(energies, energies[0])

    def test_era_classification_low_freq(self, low_freq_component):
        flow = TeotlEnergyFlow()
        state = flow.initialize_from_component(low_freq_component)
        assert state.era == SunEra.NAHUI_OCELOTL  # Jaguar era (low-freq dominated)

    def test_flow_state_serialization(self, broadband_component):
        flow = TeotlEnergyFlow()
        state = flow.initialize_from_component(broadband_component)
        d = state.to_dict()
        assert "layer_energies" in d
        assert "era" in d
        assert len(d["layer_energies"]) == 22
