"""Tests for the Spectral Reasoning Engine."""

import numpy as np
import pytest

from mesie.cognitive.reasoning_engine import (
    AbductiveReasoner,
    BayesianUpdater,
    CausalGraph,
    CausalLink,
    ConfidenceLevel,
    CounterfactualEngine,
    Evidence,
    EvidenceAccumulator,
    EvidenceType,
    Hypothesis,
    ReasoningMode,
    SpectralPatternRecognizer,
    SpectralReasoningEngine,
)


class TestEvidence:
    def test_supporting_evidence(self):
        e = Evidence(evidence_type=EvidenceType.SPECTRAL_PATTERN, description="peak", strength=0.5)
        assert e.is_supporting
        assert e.absolute_strength == 0.5

    def test_contradicting_evidence(self):
        e = Evidence(evidence_type=EvidenceType.STATISTICAL, description="low", strength=-0.3)
        assert not e.is_supporting
        assert abs(e.absolute_strength - 0.3) < 1e-6


class TestHypothesis:
    def test_posterior_no_evidence(self):
        h = Hypothesis(hypothesis_id="h1", description="test", prior_probability=0.5)
        assert abs(h.posterior_probability - 0.5) < 1e-6

    def test_posterior_with_supporting_evidence(self):
        h = Hypothesis(hypothesis_id="h1", description="test", prior_probability=0.5)
        h.evidence.append(Evidence(
            evidence_type=EvidenceType.SPECTRAL_PATTERN, description="peak", strength=0.8
        ))
        assert h.posterior_probability > 0.5

    def test_posterior_with_contradicting_evidence(self):
        h = Hypothesis(hypothesis_id="h1", description="test", prior_probability=0.5)
        h.evidence.append(Evidence(
            evidence_type=EvidenceType.SPECTRAL_PATTERN, description="no peak", strength=-0.8
        ))
        assert h.posterior_probability < 0.5

    def test_confidence_level(self):
        h = Hypothesis(hypothesis_id="h1", description="test", prior_probability=0.9)
        assert h.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH)


class TestCausalGraph:
    def test_add_nodes_and_edges(self):
        g = CausalGraph()
        g.add_node("A")
        g.add_node("B")
        link = CausalLink(cause="A", effect="B", strength=0.8)
        g.add_edge(link)
        assert g.n_nodes == 2
        assert g.n_edges == 1

    def test_cycle_prevention(self):
        g = CausalGraph()
        g.add_edge(CausalLink(cause="A", effect="B", strength=0.5))
        g.add_edge(CausalLink(cause="B", effect="C", strength=0.5))
        g.add_edge(CausalLink(cause="C", effect="A", strength=0.5))  # Would create cycle
        assert g.n_edges == 2  # Third edge rejected

    def test_causal_chain(self):
        g = CausalGraph()
        g.add_edge(CausalLink(cause="A", effect="B", strength=0.9))
        g.add_edge(CausalLink(cause="B", effect="C", strength=0.8))
        chain = g.get_causal_chain("A", "C")
        assert len(chain) == 2

    def test_causal_strength(self):
        g = CausalGraph()
        g.add_edge(CausalLink(cause="A", effect="B", strength=0.9))
        g.add_edge(CausalLink(cause="B", effect="C", strength=0.8))
        strength = g.compute_causal_strength("A", "C")
        assert abs(strength - 0.72) < 1e-6

    def test_intervene(self):
        g = CausalGraph()
        g.add_edge(CausalLink(cause="A", effect="B", strength=0.7))
        g.add_edge(CausalLink(cause="B", effect="C", strength=0.5))
        effects = g.intervene("A", 1.0)
        assert "B" in effects
        assert "C" in effects

    def test_root_causes_and_terminal_effects(self):
        g = CausalGraph()
        g.add_edge(CausalLink(cause="A", effect="B", strength=0.5))
        g.add_edge(CausalLink(cause="B", effect="C", strength=0.5))
        assert "A" in g.get_root_causes()
        assert "C" in g.get_terminal_effects()


class TestBayesianUpdater:
    def test_initialize_uniform(self):
        updater = BayesianUpdater(prior_type="uniform")
        updater.initialize_beliefs(["h1", "h2", "h3"])
        assert abs(updater.get_belief("h1") - 1.0 / 3) < 1e-6

    def test_update_increases_belief(self):
        updater = BayesianUpdater()
        updater.initialize_beliefs(["h1", "h2"])
        evidence = Evidence(evidence_type=EvidenceType.SPECTRAL_PATTERN, description="x", strength=0.8)
        posterior = updater.update("h1", evidence)
        assert posterior > 0.5

    def test_beliefs_sum_to_one(self):
        updater = BayesianUpdater()
        updater.initialize_beliefs(["h1", "h2", "h3"])
        evidence = Evidence(evidence_type=EvidenceType.STATISTICAL, description="x", strength=0.5)
        updater.update("h1", evidence)
        total = sum(updater.beliefs.values())
        assert abs(total - 1.0) < 1e-6

    def test_entropy(self):
        updater = BayesianUpdater()
        updater.initialize_beliefs(["h1", "h2"])
        entropy = updater.get_entropy()
        assert entropy > 0  # Uniform has maximum entropy


class TestAbductiveReasoner:
    def test_generate_hypotheses(self):
        reasoner = AbductiveReasoner()
        obs = {"pattern_type": "peak_shift", "magnitude": 2.0}
        hypotheses = reasoner.generate_hypotheses(obs)
        assert len(hypotheses) > 0

    def test_evaluate_hypothesis(self):
        reasoner = AbductiveReasoner()
        h = Hypothesis(hypothesis_id="h1", description="resonance_approach")
        spectrum = np.random.randn(100)
        score = reasoner.evaluate_hypothesis(h, spectrum)
        assert 0 <= score <= 1


class TestCounterfactualEngine:
    def test_remove_peak(self):
        engine = CounterfactualEngine()
        spectrum = np.zeros(100)
        spectrum[50] = 5.0  # Peak
        cf = engine.create_counterfactual(spectrum, "remove_peak", {"center": 50, "width": 5})
        assert cf[50] < spectrum[50]

    def test_add_damping(self):
        engine = CounterfactualEngine()
        spectrum = np.ones(100)
        cf = engine.create_counterfactual(spectrum, "add_damping", {"factor": 0.5})
        assert np.mean(cf) < np.mean(spectrum)

    def test_assess_causal_effect(self):
        engine = CounterfactualEngine()
        spectrum = np.random.randn(100) + 5.0
        result = engine.assess_causal_effect(spectrum, "add_damping", {"factor": 0.1})
        assert "effect_size" in result
        assert "interpretation" in result


class TestEvidenceAccumulator:
    def test_accumulation(self):
        acc = EvidenceAccumulator(threshold=2.0)
        for _ in range(10):
            e = Evidence(evidence_type=EvidenceType.SPECTRAL_PATTERN, description="x", strength=0.5)
            decision = acc.accumulate(e)
            if decision is not None:
                assert decision == "accept"
                break

    def test_reject_decision(self):
        acc = EvidenceAccumulator(threshold=2.0)
        for _ in range(20):
            e = Evidence(evidence_type=EvidenceType.SPECTRAL_PATTERN, description="x", strength=-0.5)
            decision = acc.accumulate(e)
            if decision is not None:
                assert decision == "reject"
                break

    def test_state(self):
        acc = EvidenceAccumulator(threshold=5.0)
        e = Evidence(evidence_type=EvidenceType.STATISTICAL, description="x", strength=0.3)
        acc.accumulate(e)
        state = acc.get_state()
        assert state["step_count"] == 1
        assert not state["decision_made"]


class TestSpectralPatternRecognizer:
    def test_detect_peaks(self):
        recognizer = SpectralPatternRecognizer(min_peak_prominence=0.5)
        spectrum = np.zeros(100)
        spectrum[30] = 5.0
        spectrum[60] = 3.0
        peaks = recognizer.detect_peaks(spectrum)
        assert len(peaks) >= 2

    def test_analyze_bands(self):
        recognizer = SpectralPatternRecognizer(n_frequency_bands=4)
        spectrum = np.random.randn(128)
        bands = recognizer.analyze_bands(spectrum)
        assert len(bands) == 4

    def test_full_analysis(self):
        recognizer = SpectralPatternRecognizer()
        spectrum = np.random.randn(256) + 1.0
        analysis = recognizer.full_analysis(spectrum)
        assert "peaks" in analysis
        assert "bands" in analysis
        assert "total_energy" in analysis


class TestSpectralReasoningEngine:
    def test_basic_reasoning(self):
        engine = SpectralReasoningEngine(enable_counterfactuals=False)
        spectrum = np.random.randn(128) + 2.0
        result = engine.reason(spectrum)
        assert result.chain_id == "chain_1"
        assert len(result.steps) >= 3
        assert result.overall_confidence > 0

    def test_full_reasoning_with_counterfactuals(self):
        engine = SpectralReasoningEngine()
        spectrum = np.random.randn(64) + 1.0
        result = engine.reason(spectrum)
        assert result.final_conclusion != ""
        assert result.n_steps >= 4

    def test_multiple_reasoning_chains(self):
        engine = SpectralReasoningEngine(enable_counterfactuals=False, enable_causal=False)
        for _ in range(3):
            engine.reason(np.random.randn(64))
        assert engine.n_reasoning_chains == 3
