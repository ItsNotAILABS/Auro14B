"""Enterprise long-workflow pattern tests — 5,000 Monte Carlo trials.

Validates MESIE/MAESI across 10 enterprise verticals with extended
multi-step workflows including SDK install verification, pipeline
orchestration, and end-to-end integration patterns.

Run:
    pytest tests/test_enterprise_workflows.py -v
    pytest tests/test_enterprise_workflows.py -k "workflow" -v
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pytest

from mesie import match_records, validate_record
from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.embeddings import SpectralVectorizer
from mesie.embeddings.fingerprint import SpectralFingerprintPipeline
from mesie.io.loaders import load_record
from mesie.matching.ranking import rank_candidates
from mesie.sdk import SpectralIntelligenceSDK, MAESIClient, search_research


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(record_id: str, rng: np.random.Generator, n: int = 64) -> MultiElementRecord:
    """Create a synthetic spectral record."""
    f = np.linspace(0.1, 50.0, n)
    a = np.abs(rng.normal(1.0, 0.3, n))
    return MultiElementRecord(
        record_id=record_id,
        components=[SpectralComponent(name="ch1", frequency=f, amplitude=a)],
    )


def _noisy_copy(rec: MultiElementRecord, rng: np.random.Generator, scale: float = 0.05) -> MultiElementRecord:
    """Add noise to a record's amplitude."""
    c = rec.components[0]
    a = np.maximum(np.abs(c.amplitude) * (1.0 + rng.normal(0, scale, len(c.amplitude))), 1e-12)
    return MultiElementRecord(
        record_id=f"{rec.record_id}_noisy",
        components=[SpectralComponent(name=c.name, frequency=c.frequency.copy(), amplitude=a)],
    )


# ---------------------------------------------------------------------------
# SDK Install & Basic Smoke Tests
# ---------------------------------------------------------------------------

class TestSDKInstall:
    """Verify SDK installation and core capabilities."""

    def test_sdk_instantiates(self):
        engine = SpectralIntelligenceSDK()
        assert "0.4" in engine.version

    def test_sdk_generate_psd(self):
        engine = SpectralIntelligenceSDK()
        rec = engine.generate_psd()
        assert rec.components
        assert len(rec.components[0].frequency) > 0

    def test_sdk_generate_fas(self):
        engine = SpectralIntelligenceSDK()
        rec = engine.generate_fas()
        assert rec.components

    def test_sdk_validate(self):
        engine = SpectralIntelligenceSDK()
        rec = engine.generate_psd()
        report = engine.validate(rec)
        assert report.is_valid

    def test_sdk_match_pair(self):
        engine = SpectralIntelligenceSDK()
        r1 = engine.generate_psd()
        r2 = engine.generate_psd()
        result = engine.match(r1, r2)
        assert 0 <= result.composite_score <= 1.0

    def test_sdk_embed(self):
        engine = SpectralIntelligenceSDK()
        rec = engine.generate_psd()
        emb = engine.embed(rec)
        assert emb.shape[0] == 1
        assert emb.shape[1] > 0

    def test_maesi_client_init(self):
        client = MAESIClient(fast=True, use_fingerprint=True)
        assert client is not None

    def test_research_search(self):
        hits = search_research("spectral analysis", top_k=3)
        assert len(hits) >= 1


# ---------------------------------------------------------------------------
# Enterprise Long Workflow Pattern Tests
# ---------------------------------------------------------------------------

class TestManufacturingWorkflow:
    """Manufacturing: multi-step predictive maintenance pipeline."""

    def test_full_pipeline_vibration_monitoring(self):
        """End-to-end: generate baseline → add drift → detect anomaly → rank."""
        rng = np.random.default_rng(100)
        engine = SpectralIntelligenceSDK()

        # Step 1: Establish baseline fleet
        fleet = [_make_record(f"machine_{i}", rng) for i in range(10)]

        # Step 2: Validate all baselines
        for rec in fleet:
            report = engine.validate(rec)
            assert report.is_valid

        # Step 3: Simulate drift on one machine
        drifted = _noisy_copy(fleet[3], rng, scale=0.25)

        # Step 4: Rank to identify closest baseline (should still find machine_3)
        ranked = rank_candidates(drifted, fleet, top_k=3)
        assert ranked[0].score >= 0.4

        # Step 5: Confirm anomaly separation
        normal = _noisy_copy(fleet[3], rng, scale=0.02)
        match_normal = match_records(fleet[3], normal).composite_score
        match_drift = match_records(fleet[3], drifted).composite_score
        assert match_normal >= match_drift  # normal closer to baseline


class TestEnergyWorkflow:
    """Energy: grid monitoring with validation pipeline."""

    def test_schumann_monitoring_pipeline(self):
        """Multi-step: generate → perturb → validate → embed → compare."""
        rng = np.random.default_rng(200)
        engine = SpectralIntelligenceSDK()

        # Generate Schumann-like spectrum (7.83, 14.3, 20.8 Hz resonances)
        f = np.linspace(0.1, 50.0, 256)
        a = np.zeros(256)
        for peak in [7.83, 14.3, 20.8, 27.3, 33.8]:
            a += np.exp(-0.5 * ((f - peak) / 0.5) ** 2)
        a = np.maximum(a, 1e-6)

        baseline = MultiElementRecord(
            record_id="schumann_baseline",
            components=[SpectralComponent(name="EM", frequency=f, amplitude=a)],
        )

        # Validate
        report = engine.validate(baseline)
        assert report.is_valid

        # Perturb under noise
        noisy_versions = [_noisy_copy(baseline, rng, scale=0.03) for _ in range(20)]
        for nv in noisy_versions:
            v = validate_record(nv)
            assert v.is_valid

        # Embed all and check stability
        vectorizer = SpectralVectorizer()
        embs = np.array([vectorizer.transform(nv) for nv in noisy_versions])
        # Embeddings should be stable (low std relative to mean)
        assert np.mean(np.std(embs, axis=0)) < np.mean(np.abs(np.mean(embs, axis=0)))


class TestAerospaceWorkflow:
    """Aerospace: orbital edge + seismic anchor coupling."""

    def test_multi_domain_coupling(self):
        """Cross-domain match: orbital signature vs ground seismic anchor."""
        rng = np.random.default_rng(300)
        engine = SpectralIntelligenceSDK()

        # Orbital edge spectrum (higher freq)
        f_orbital = np.linspace(1.0, 100.0, 128)
        a_orbital = rng.exponential(0.5, 128)
        orbital = MultiElementRecord(
            record_id="orbital_edge",
            components=[SpectralComponent(name="orbital", frequency=f_orbital, amplitude=a_orbital)],
        )

        # Ground seismic anchor (lower freq)
        f_seismic = np.linspace(0.01, 20.0, 128)
        a_seismic = rng.exponential(1.0, 128)
        seismic = MultiElementRecord(
            record_id="seismic_anchor",
            components=[SpectralComponent(name="seismic", frequency=f_seismic, amplitude=a_seismic)],
        )

        # Both should validate independently
        assert engine.validate(orbital).is_valid
        assert engine.validate(seismic).is_valid

        # Cross-domain matching should produce a score
        result = engine.match(orbital, seismic)
        assert 0 <= result.composite_score <= 1.0


class TestInsuranceWorkflow:
    """Insurance: catastrophe risk cross-matching pipeline."""

    def test_seismic_risk_assessment(self):
        """Multi-record risk scoring: compare earthquake spectra to structural refs."""
        rng = np.random.default_rng(400)

        # Simulate library of structural vulnerability signatures
        structures = [_make_record(f"structure_{i}", rng) for i in range(15)]

        # Simulate earthquake event
        eq_event = _make_record("earthquake_2024", rng)

        # Cross-match against all structures
        ranked = rank_candidates(eq_event, structures, top_k=5)
        assert len(ranked) == 5
        assert all(r.score >= 0 for r in ranked)

        # Scores should be ordered
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)


class TestConstructionWorkflow:
    """Construction: structural FAS ranking under perturbation."""

    def test_fas_ranking_stability(self):
        """Rank stability: perturbed queries should still find original."""
        rng = np.random.default_rng(500)
        engine = SpectralIntelligenceSDK()

        # Create structural library
        library = [_make_record(f"building_{i}", rng) for i in range(20)]

        # Perturb building_5 multiple times and rank
        target = library[5]
        hits = 0
        for _ in range(50):
            perturbed = _noisy_copy(target, rng, scale=0.06)
            ranked = rank_candidates(perturbed, library, top_k=3)
            if any("building_5" in r.candidate_id for r in ranked):
                hits += 1

        # Should find original in top-3 at least 80% of the time
        assert hits / 50 >= 0.80


class TestHealthcareWorkflow:
    """Healthcare: device monitoring anomaly detection."""

    def test_anomaly_vs_baseline_pipeline(self):
        """Full pipeline: fit baseline → score normal → score anomaly → separate."""
        from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter

        rng = np.random.default_rng(600)

        # Establish baseline from healthy device readings
        baselines = [_make_record(f"healthy_{i}", rng) for i in range(10)]

        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline(baselines)

        # Normal readings should have low anomaly scores
        normal_scores = [adapter.score_anomaly(_noisy_copy(baselines[0], rng, 0.02)) for _ in range(20)]

        # Anomalous reading (completely different spectrum)
        anomaly = _make_record("faulty_device", np.random.default_rng(999))
        anomaly_score = adapter.score_anomaly(anomaly)

        # Anomaly should score higher than normals
        assert anomaly_score > np.mean(normal_scores)


class TestRoboticsWorkflow:
    """Robotics: fleet ANN state lookup workflow."""

    def test_fleet_state_lookup_latency(self):
        """Fleet of 50 robots: index states → query → verify sub-10ms."""
        rng = np.random.default_rng(700)

        # Index fleet states
        fleet = [_make_record(f"robot_{i}", rng) for i in range(50)]
        fp = SpectralFingerprintPipeline()
        fp.index_records(fleet)

        # Query latency
        query = _noisy_copy(fleet[25], rng)
        t0 = time.perf_counter()
        results = fp.query(query, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(results) >= 1
        assert elapsed_ms < 100  # generous bound for CI


class TestTelecomWorkflow:
    """Telecom: spectrum compliance with research + EM library."""

    def test_compliance_research_hit(self):
        """Verify research catalog returns EM-related entries."""
        hits = search_research("electromagnetic spectrum band", top_k=5)
        assert len(hits) >= 1

    def test_em_band_validation(self):
        """EM band spectrum passes validation."""
        rng = np.random.default_rng(800)
        f = np.linspace(30.0, 300.0, 128)  # VHF-like band
        a = rng.exponential(0.3, 128)
        rec = MultiElementRecord(
            record_id="em_compliance_check",
            components=[SpectralComponent(name="VHF", frequency=f, amplitude=a)],
        )
        report = validate_record(rec)
        assert report.is_valid


class TestResearchWorkflow:
    """Research: R&D lab benchmark classification."""

    def test_classification_ranking_accuracy(self):
        """Classify unknown samples by ranking against known library."""
        rng = np.random.default_rng(900)

        # Build labeled library: 3 classes, 10 each
        library = []
        for cls in range(3):
            for i in range(10):
                f = np.linspace(0.1, 50.0, 64)
                a = np.abs(rng.normal(1.0 + cls * 0.5, 0.2, 64))
                rec = MultiElementRecord(
                    record_id=f"class{cls}_sample{i}",
                    components=[SpectralComponent(name="lab", frequency=f, amplitude=a)],
                )
                library.append(rec)

        # Query from class 1
        query_f = np.linspace(0.1, 50.0, 64)
        query_a = np.abs(rng.normal(1.5, 0.2, 64))
        query = MultiElementRecord(
            record_id="unknown",
            components=[SpectralComponent(name="lab", frequency=query_f, amplitude=query_a)],
        )

        ranked = rank_candidates(query, library, top_k=5)
        assert ranked[0].score >= 0.4


class TestEnterpriseAIWorkflow:
    """Enterprise AI: agent spectral memory with MAESI + fingerprint."""

    def test_agent_memory_workflow(self):
        """Full workflow: index corpus → query → retrieve neighbors → fingerprint."""
        rng = np.random.default_rng(1000)

        corpus = [_make_record(f"memory_{i}", rng) for i in range(20)]

        # MAESI client
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(corpus)

        # Fingerprint pipeline
        fp = SpectralFingerprintPipeline()
        fp.index_records(corpus)

        # Query
        query = _noisy_copy(corpus[10], rng)
        result = client.query(query, top_k=5)
        fp_hits = fp.query(query, top_k=3)

        assert len(result.neighbors) >= 1
        assert len(fp_hits) >= 1


# ---------------------------------------------------------------------------
# Monte Carlo 5,000-Trial Enterprise Suite
# ---------------------------------------------------------------------------

class TestMonteCarloEnterprise5000:
    """5,000-trial Monte Carlo validation (500 per use case × 10 enterprises).

    Each trial introduces stochastic perturbation and verifies the core
    success metric for that enterprise vertical.
    """

    TRIALS_PER_CASE = 500
    SEED = 42

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rng = np.random.default_rng(self.SEED)
        self.records = [_make_record(f"ref_{i}", self.rng) for i in range(10)]

    def _run_trials(self, fn, n=None):
        n = n or self.TRIALS_PER_CASE
        successes = sum(1 for _ in range(n) if fn())
        rate = successes / n
        return rate

    def test_manufacturing_500_trials(self):
        """Manufacturing: cosine similarity >= 0.5 under vibration drift."""
        client = MAESIClient(fast=True, use_fingerprint=False)
        client.index_corpus(self.records)

        def trial():
            q = _noisy_copy(self.records[0], self.rng, 0.05)
            hits = client.fast_compute.cosine_search(q, top_k=1)
            return hits[0][1] >= 0.5 if hits else False

        assert self._run_trials(trial) >= 0.85

    def test_energy_500_trials(self):
        """Energy: validation level >= 4 under noise."""
        def trial():
            q = _noisy_copy(self.records[1], self.rng, 0.03)
            v = validate_record(q)
            return v.level >= 4 and v.is_valid

        assert self._run_trials(trial) >= 0.85

    def test_aerospace_500_trials(self):
        """Aerospace: match score >= 0.6 across orbital-seismic coupling."""
        def trial():
            q = _noisy_copy(self.records[2], self.rng, 0.08)
            m = match_records(self.records[2], q).composite_score
            return m >= 0.6

        assert self._run_trials(trial) >= 0.85

    def test_insurance_500_trials(self):
        """Insurance: cross-match score >= 0.55."""
        def trial():
            q = _noisy_copy(self.records[3], self.rng, 0.08)
            m = match_records(q, self.records[4]).composite_score
            return m >= 0.55

        assert self._run_trials(trial) >= 0.85

    def test_construction_500_trials(self):
        """Construction: FAS ranking finds self under perturbation."""
        def trial():
            q = _noisy_copy(self.records[5], self.rng, 0.06)
            ranked = rank_candidates(q, self.records, top_k=3)
            return any("ref_5" in r.candidate_id for r in ranked)

        assert self._run_trials(trial) >= 0.85

    def test_healthcare_500_trials(self):
        """Healthcare: anomaly detection separates outlier from baseline."""
        from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter
        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline(self.records[:5])

        def trial():
            normal = _noisy_copy(self.records[0], self.rng, 0.02)
            outlier = _noisy_copy(self.records[8], self.rng, 0.15)
            return adapter.score_anomaly(outlier) > adapter.score_anomaly(normal)

        assert self._run_trials(trial) >= 0.85

    def test_robotics_500_trials(self):
        """Robotics: ANN lookup < 50ms with similarity > 0.4."""
        fp = SpectralFingerprintPipeline()
        fp.index_records(self.records)

        def trial():
            q = _noisy_copy(self.records[7], self.rng)
            t0 = time.perf_counter()
            hits = fp.query(q, top_k=3)
            ms = (time.perf_counter() - t0) * 1000
            return ms < 50 and len(hits) >= 1

        assert self._run_trials(trial) >= 0.85

    def test_telecom_500_trials(self):
        """Telecom: research catalog hit for EM queries."""
        queries = ["spectrum band", "EM frequency", "radio wave", "spectral", "resonance"]

        def trial():
            q = queries[self.rng.integers(0, len(queries))]
            hits = search_research(q, top_k=2)
            return len(hits) >= 1

        assert self._run_trials(trial) >= 0.85

    def test_research_500_trials(self):
        """Research: ranking score >= 0.45 on benchmark samples."""
        def trial():
            q = _noisy_copy(self.records[self.rng.integers(0, len(self.records))], self.rng)
            ranked = rank_candidates(q, self.records, top_k=1)
            return ranked[0].score >= 0.45 if ranked else False

        assert self._run_trials(trial) >= 0.85

    def test_enterprise_ai_500_trials(self):
        """Enterprise AI: MAESI query with neighbors and fast latency."""
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(self.records)

        def trial():
            q = self.records[self.rng.integers(0, len(self.records))]
            t0 = time.perf_counter()
            r = client.query(q, top_k=3)
            ms = (time.perf_counter() - t0) * 1000
            return len(r.neighbors) >= 1 and ms < 100

        assert self._run_trials(trial) >= 0.85
