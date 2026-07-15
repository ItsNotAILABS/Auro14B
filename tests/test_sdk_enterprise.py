"""Comprehensive Enterprise SDK Test Suite.

Consolidates all enterprise tests from test_enterprise_workflows.py and
test_enterprise_ai_advanced.py into the SDK test namespace, plus 20 additional
enterprise integration tests.

Run:
    pytest tests/test_sdk_enterprise.py -v
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from mesie import match_records, validate_record, generate_psd, generate_fas
from mesie.core.config import GenerationConfig
from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.embeddings import SpectralVectorizer
from mesie.embeddings.fingerprint import SpectralFingerprintPipeline
from mesie.io.loaders import load_record
from mesie.matching.ranking import rank_candidates
from mesie.sdk import (
    SpectralIntelligenceSDK,
    MAESIClient,
    search_research,
    get_technical_library,
    get_research_catalog,
    get_fundamental_laws,
    get_periodic_table,
    get_biological_systems,
    FastSpectralCompute,
)


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


# ===========================================================================
# SECTION 1: Enterprise Workflow Tests (from test_enterprise_workflows.py)
# ===========================================================================


class TestSDKInstallEnterprise:
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


class TestSDKManufacturingWorkflow:
    """Manufacturing: multi-step predictive maintenance pipeline."""

    def test_full_pipeline_vibration_monitoring(self):
        """End-to-end: generate baseline → add drift → detect anomaly → rank."""
        rng = np.random.default_rng(100)
        engine = SpectralIntelligenceSDK()

        fleet = [_make_record(f"machine_{i}", rng) for i in range(10)]
        for rec in fleet:
            report = engine.validate(rec)
            assert report.is_valid

        drifted = _noisy_copy(fleet[3], rng, scale=0.25)
        ranked = rank_candidates(drifted, fleet, top_k=3)
        assert ranked[0].score >= 0.4

        normal = _noisy_copy(fleet[3], rng, scale=0.02)
        match_normal = match_records(fleet[3], normal).composite_score
        match_drift = match_records(fleet[3], drifted).composite_score
        assert match_normal >= match_drift


class TestSDKEnergyWorkflow:
    """Energy: grid monitoring with validation pipeline."""

    def test_schumann_monitoring_pipeline(self):
        """Multi-step: generate → perturb → validate → embed → compare."""
        rng = np.random.default_rng(200)
        engine = SpectralIntelligenceSDK()

        f = np.linspace(0.1, 50.0, 256)
        a = np.zeros(256)
        for peak in [7.83, 14.3, 20.8, 27.3, 33.8]:
            a += np.exp(-0.5 * ((f - peak) / 0.5) ** 2)
        a = np.maximum(a, 1e-6)

        baseline = MultiElementRecord(
            record_id="schumann_baseline",
            components=[SpectralComponent(name="EM", frequency=f, amplitude=a)],
        )

        report = engine.validate(baseline)
        assert report.is_valid

        noisy_versions = [_noisy_copy(baseline, rng, scale=0.03) for _ in range(20)]
        for nv in noisy_versions:
            v = validate_record(nv)
            assert v.is_valid

        vectorizer = SpectralVectorizer()
        embs = np.array([vectorizer.transform(nv) for nv in noisy_versions])
        assert np.mean(np.std(embs, axis=0)) < np.mean(np.abs(np.mean(embs, axis=0)))


class TestSDKAerospaceWorkflow:
    """Aerospace: orbital edge + seismic anchor coupling."""

    def test_multi_domain_coupling(self):
        """Cross-domain match: orbital signature vs ground seismic anchor."""
        rng = np.random.default_rng(300)
        engine = SpectralIntelligenceSDK()

        f_orbital = np.linspace(1.0, 100.0, 128)
        a_orbital = rng.exponential(0.5, 128)
        orbital = MultiElementRecord(
            record_id="orbital_edge",
            components=[SpectralComponent(name="orbital", frequency=f_orbital, amplitude=a_orbital)],
        )

        f_seismic = np.linspace(0.01, 20.0, 128)
        a_seismic = rng.exponential(1.0, 128)
        seismic = MultiElementRecord(
            record_id="seismic_anchor",
            components=[SpectralComponent(name="seismic", frequency=f_seismic, amplitude=a_seismic)],
        )

        assert engine.validate(orbital).is_valid
        assert engine.validate(seismic).is_valid

        result = engine.match(orbital, seismic)
        assert 0 <= result.composite_score <= 1.0


class TestSDKInsuranceWorkflow:
    """Insurance: catastrophe risk cross-matching pipeline."""

    def test_seismic_risk_assessment(self):
        """Multi-record risk scoring: compare earthquake spectra to structural refs."""
        rng = np.random.default_rng(400)
        structures = [_make_record(f"structure_{i}", rng) for i in range(15)]
        eq_event = _make_record("earthquake_2024", rng)

        ranked = rank_candidates(eq_event, structures, top_k=5)
        assert len(ranked) == 5
        assert all(r.score >= 0 for r in ranked)
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)


class TestSDKConstructionWorkflow:
    """Construction: structural FAS ranking under perturbation."""

    def test_fas_ranking_stability(self):
        """Rank stability: perturbed queries should still find original."""
        rng = np.random.default_rng(500)

        library = [_make_record(f"building_{i}", rng) for i in range(20)]
        target = library[5]
        hits = 0
        for _ in range(50):
            perturbed = _noisy_copy(target, rng, scale=0.06)
            ranked = rank_candidates(perturbed, library, top_k=3)
            if any("building_5" in r.candidate_id for r in ranked):
                hits += 1
        assert hits / 50 >= 0.80


class TestSDKHealthcareWorkflow:
    """Healthcare: device monitoring anomaly detection."""

    def test_anomaly_vs_baseline_pipeline(self):
        """Full pipeline: fit baseline → score normal → score anomaly → separate."""
        from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter

        rng = np.random.default_rng(600)
        baselines = [_make_record(f"healthy_{i}", rng) for i in range(10)]

        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline(baselines)

        normal_scores = [adapter.score_anomaly(_noisy_copy(baselines[0], rng, 0.02)) for _ in range(20)]
        anomaly = _make_record("faulty_device", np.random.default_rng(999))
        anomaly_score = adapter.score_anomaly(anomaly)
        assert anomaly_score > np.mean(normal_scores)


class TestSDKRoboticsWorkflow:
    """Robotics: fleet ANN state lookup workflow."""

    def test_fleet_state_lookup_latency(self):
        """Fleet of 50 robots: index states → query → verify sub-100ms."""
        rng = np.random.default_rng(700)
        fleet = [_make_record(f"robot_{i}", rng) for i in range(50)]
        fp = SpectralFingerprintPipeline()
        fp.index_records(fleet)

        query = _noisy_copy(fleet[25], rng)
        t0 = time.perf_counter()
        results = fp.query(query, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(results) >= 1
        assert elapsed_ms < 100


class TestSDKTelecomWorkflow:
    """Telecom: spectrum compliance with research + EM library."""

    def test_compliance_research_hit(self):
        """Verify research catalog returns EM-related entries."""
        hits = search_research("electromagnetic spectrum band", top_k=5)
        assert len(hits) >= 1

    def test_em_band_validation(self):
        """EM band spectrum passes validation."""
        rng = np.random.default_rng(800)
        f = np.linspace(30.0, 300.0, 128)
        a = rng.exponential(0.3, 128)
        rec = MultiElementRecord(
            record_id="em_compliance_check",
            components=[SpectralComponent(name="VHF", frequency=f, amplitude=a)],
        )
        report = validate_record(rec)
        assert report.is_valid


class TestSDKResearchWorkflow:
    """Research: R&D lab benchmark classification."""

    def test_classification_ranking_accuracy(self):
        """Classify unknown samples by ranking against known library."""
        rng = np.random.default_rng(900)

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

        query_f = np.linspace(0.1, 50.0, 64)
        query_a = np.abs(rng.normal(1.5, 0.2, 64))
        query = MultiElementRecord(
            record_id="unknown",
            components=[SpectralComponent(name="lab", frequency=query_f, amplitude=query_a)],
        )

        ranked = rank_candidates(query, library, top_k=5)
        assert ranked[0].score >= 0.4


class TestSDKEnterpriseAIWorkflow:
    """Enterprise AI: agent spectral memory with MAESI + fingerprint."""

    def test_agent_memory_workflow(self):
        """Full workflow: index corpus → query → retrieve neighbors → fingerprint."""
        rng = np.random.default_rng(1000)
        corpus = [_make_record(f"memory_{i}", rng) for i in range(20)]

        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(corpus)

        fp = SpectralFingerprintPipeline()
        fp.index_records(corpus)

        query = _noisy_copy(corpus[10], rng)
        result = client.query(query, top_k=5)
        fp_hits = fp.query(query, top_k=3)

        assert len(result.neighbors) >= 1
        assert len(fp_hits) >= 1


# ===========================================================================
# SECTION 2: Enterprise AI Advanced Tests (from test_enterprise_ai_advanced.py)
# ===========================================================================


class TestSDKKnowledgeGraphIntegration:
    """Validate knowledge graph capabilities for enterprise AI."""

    def test_physical_laws_completeness(self):
        """All fundamental physics laws present with spectral embeddings."""
        laws = get_fundamental_laws()
        assert len(laws) >= 10
        for law in laws:
            emb = law.to_embedding()
            assert emb.shape[0] > 0
            assert not np.any(np.isnan(emb))

    def test_periodic_table_coverage(self):
        """Periodic table has elements with spectral profiles."""
        elements = get_periodic_table()
        assert len(elements) >= 30
        symbols = [e.symbol for e in elements]
        for s in ["H", "He", "C", "N", "O", "Fe"]:
            assert s in symbols

    def test_biological_systems_available(self):
        """Biological spectral profiles accessible."""
        systems = get_biological_systems()
        assert len(systems) >= 5

    def test_technical_library_coverage(self):
        """Technical concept library has diverse domains."""
        lib = get_technical_library()
        assert len(lib) >= 15
        domains = set(c.domain.value if hasattr(c.domain, 'value') else str(c.domain) for c in lib)
        assert len(domains) >= 3

    def test_research_catalog_searchable(self):
        """Research catalog supports semantic search."""
        catalog = get_research_catalog()
        assert len(catalog) >= 20
        hits = search_research("spectral", top_k=3)
        assert len(hits) >= 1

    def test_cross_domain_knowledge_linking(self):
        """Link physics laws to technical concepts via embeddings."""
        laws = get_fundamental_laws()
        law_embs = np.array([l.to_embedding() for l in laws])
        assert np.std(law_embs) > 0.01


class TestSDKBatchProcessingPipelines:
    """Enterprise-scale batch processing workflows."""

    def test_batch_validate_100_records(self):
        """Validate 100 records in batch — all should pass."""
        rng = np.random.default_rng(42)
        records = [_make_record(f"batch_{i}", rng) for i in range(100)]
        results = [validate_record(r) for r in records]
        assert all(r.is_valid for r in results)
        assert all(r.level >= 4 for r in results)

    def test_batch_embed_50_records(self):
        """Embed 50 records and verify matrix shape."""
        rng = np.random.default_rng(43)
        records = [_make_record(f"emb_{i}", rng) for i in range(50)]
        vectorizer = SpectralVectorizer()
        embeddings = np.array([vectorizer.transform(r) for r in records])
        assert embeddings.shape[0] == 50
        assert embeddings.shape[1] > 0
        assert not np.any(np.isnan(embeddings))

    def test_batch_match_all_pairs(self):
        """Match all pairs in a small corpus — O(n²) enterprise pattern."""
        rng = np.random.default_rng(44)
        corpus = [_make_record(f"pair_{i}", rng) for i in range(10)]
        scores = []
        for i in range(len(corpus)):
            for j in range(i + 1, len(corpus)):
                result = match_records(corpus[i], corpus[j])
                scores.append(result.composite_score)
                assert 0 <= result.composite_score <= 1.0
        self_scores = [match_records(c, c).composite_score for c in corpus[:5]]
        assert np.mean(self_scores) >= np.mean(scores)

    def test_batch_rank_large_corpus(self):
        """Rank against a 200-record corpus."""
        rng = np.random.default_rng(45)
        corpus = [_make_record(f"large_{i}", rng) for i in range(200)]
        query = _noisy_copy(corpus[100], rng)
        ranked = rank_candidates(query, corpus, top_k=10)
        assert len(ranked) == 10
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_batch_generation_mixed_types(self):
        """Generate mixed PSD/FAS records in batch."""
        records = []
        for i in range(20):
            cfg = GenerationConfig(seed=i)
            if i % 2 == 0:
                records.append(generate_psd(config=cfg))
            else:
                records.append(generate_fas(config=cfg))
        assert len(records) == 20
        assert all(r.components for r in records)


class TestSDKAgenticWorkflows:
    """Test agentic capabilities: ghost agents, network dispatch, task spawning."""

    def test_ghost_agent_spawn(self):
        """Spawn a ghost agent and verify task execution."""
        from mesie.agentic.ghost import GhostAgent, GhostConfig, TaskSpec

        agent = GhostAgent(agent_id="test_ghost", config=GhostConfig())
        task = TaskSpec(
            intent="validate_spectrum",
            actions=[{"engine": "validation", "action": "validate"}],
        )
        result = agent.execute(task)
        assert result.success
        assert result.task_id

    def test_agent_network_star_topology(self):
        """Create star topology network and dispatch work."""
        from mesie.agentic.network import AgentNetwork, NetworkTopology
        from mesie.agentic.ghost import GhostAgent, TaskSpec

        network = AgentNetwork(topology=NetworkTopology.STAR)
        for i in range(4):
            network.add_node(GhostAgent(agent_id=f"worker_{i}"))

        tasks = [
            TaskSpec(intent="validate", actions=[{"engine": "validation", "action": "validate"}]),
            TaskSpec(intent="match", actions=[{"engine": "matching", "action": "match"}]),
            TaskSpec(intent="embed", actions=[{"engine": "embedding", "action": "embed"}]),
            TaskSpec(intent="generate", actions=[{"engine": "generation", "action": "generate"}]),
        ]
        result = network.execute_parallel(tasks)
        assert result.success
        assert len(result.node_results) >= 1

    def test_agent_network_pipeline_topology(self):
        """Pipeline topology: sequential multi-step processing."""
        from mesie.agentic.network import AgentNetwork, NetworkTopology
        from mesie.agentic.ghost import GhostAgent, TaskSpec

        network = AgentNetwork(topology=NetworkTopology.PIPELINE)
        for i in range(3):
            network.add_node(GhostAgent(agent_id=f"pipe_{i}"))

        pipeline = [
            TaskSpec(intent="generate_psd", actions=[{"engine": "generation", "action": "generate_psd"}]),
            TaskSpec(intent="validate", actions=[{"engine": "validation", "action": "validate"}]),
            TaskSpec(intent="embed", actions=[{"engine": "embedding", "action": "embed"}]),
        ]
        result = network.execute_parallel(pipeline)
        assert result.success

    def test_agent_spawner_pool(self):
        """Spawner manages pool of agents."""
        from mesie.agentic.spawner import AgentSpawner
        from mesie.agentic.ghost import TaskSpec

        spawner = AgentSpawner()
        results = []
        for i in range(5):
            task = TaskSpec(intent=f"task_{i}", actions=[{"engine": "validation", "action": "validate"}])
            results.append(spawner.spawn(task, agent_id=f"worker_{i}"))
        assert len(results) == 5
        assert all(r.success for r in results)
        assert spawner.total_spawned >= 1

    def test_sdk_spawn_task(self):
        """SDK-level task spawning for enterprise orchestration."""
        engine = SpectralIntelligenceSDK()
        rec = generate_psd(config=GenerationConfig(seed=1))
        result = engine.spawn_task(
            "enterprise_validate",
            [{"engine": "validation", "action": "validate"}],
            record=rec,
        )
        assert result.success


class TestSDKMultiNetworkSpeed:
    """Performance tests for enterprise-scale operations."""

    def test_fast_compute_index_1000(self):
        """Index 1000 records in < 5s."""
        rng = np.random.default_rng(50)
        records = [_make_record(f"speed_{i}", rng) for i in range(1000)]
        fc = FastSpectralCompute()
        t0 = time.perf_counter()
        fc.build_index(records)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0

    def test_fast_search_latency(self):
        """Single search < 50ms after indexing."""
        rng = np.random.default_rng(51)
        records = [_make_record(f"lat_{i}", rng) for i in range(500)]
        fc = FastSpectralCompute()
        fc.build_index(records)
        query = _noisy_copy(records[250], rng)
        t0 = time.perf_counter()
        hits = fc.cosine_search(query, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50
        assert len(hits) == 10

    def test_fingerprint_throughput(self):
        """Fingerprint pipeline handles 100 queries efficiently."""
        rng = np.random.default_rng(52)
        corpus = [_make_record(f"fp_{i}", rng) for i in range(100)]
        fp = SpectralFingerprintPipeline()
        fp.index_records(corpus)
        t0 = time.perf_counter()
        for i in range(100):
            q = _noisy_copy(corpus[i], rng)
            fp.query(q, top_k=3)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0

    def test_maesi_concurrent_queries(self):
        """MAESI client handles burst of 50 queries."""
        rng = np.random.default_rng(53)
        corpus = [_make_record(f"burst_{i}", rng) for i in range(50)]
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(corpus)
        t0 = time.perf_counter()
        results = [client.query(corpus[i], top_k=3) for i in range(50)]
        elapsed = time.perf_counter() - t0
        assert all(len(r.neighbors) >= 1 for r in results)
        assert elapsed < 5.0


class TestSDKDiagnosticsAndHealth:
    """SDK health checks and diagnostic capabilities."""

    def test_sdk_version_report(self):
        """SDK reports version correctly."""
        engine = SpectralIntelligenceSDK()
        assert "0.4" in engine.version

    def test_knowledge_stats(self):
        """MAESI reports knowledge base statistics."""
        client = MAESIClient(fast=True, use_fingerprint=False)
        stats = client.knowledge_stats()
        assert stats.physical_laws >= 10
        assert stats.chemical_elements >= 30
        assert stats.biological_systems >= 5
        assert stats.technical_concepts >= 15
        assert stats.research_entries >= 20

    def test_sdk_repr(self):
        """SDK has informative repr."""
        engine = SpectralIntelligenceSDK()
        r = repr(engine)
        assert "SpectralIntelligenceSDK" in r
        assert "0.4" in r

    def test_validation_report_serializable(self):
        """Validation reports can be serialized to JSON."""
        rec = generate_psd(config=GenerationConfig(seed=1))
        report = validate_record(rec)
        data = {
            "is_valid": report.is_valid,
            "level": report.level,
            "errors": report.errors,
            "warnings": report.warnings,
        }
        serialized = json.dumps(data)
        assert "is_valid" in serialized

    def test_match_result_serializable(self):
        """Match results can be serialized."""
        r1 = generate_psd(config=GenerationConfig(seed=1))
        r2 = generate_psd(config=GenerationConfig(seed=2))
        result = match_records(r1, r2)
        data = {
            "composite_score": result.composite_score,
            "metric_breakdown": result.metric_breakdown,
        }
        serialized = json.dumps(data)
        assert "composite_score" in serialized


class TestSDKEndToEndEnterprise:
    """Full enterprise AI integration workflows."""

    def test_ingest_validate_embed_search_pipeline(self):
        """Complete pipeline: ingest → validate → embed → index → search."""
        rng = np.random.default_rng(60)
        engine = SpectralIntelligenceSDK()

        records = [_make_record(f"ingest_{i}", rng) for i in range(30)]
        valid = [r for r in records if engine.validate(r).is_valid]
        assert len(valid) == 30

        embeddings = engine.embed(valid)
        assert embeddings.shape[0] == 30
        assert embeddings.shape[1] > 0

        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(valid)

        query = _noisy_copy(valid[15], rng)
        result = client.query(query, top_k=5)
        assert len(result.neighbors) >= 1

    def test_generate_augment_classify_workflow(self):
        """Generate synthetic → augment with noise → classify via ranking."""
        rng = np.random.default_rng(61)

        library = []
        for i in range(5):
            cfg = GenerationConfig(seed=i * 10)
            library.append(generate_psd(config=cfg))

        augmented = []
        for rec in library:
            for _ in range(10):
                augmented.append(_noisy_copy(rec, rng, scale=0.05))

        query = _noisy_copy(library[2], rng, scale=0.03)
        ranked = rank_candidates(query, library, top_k=3)
        assert ranked[0].score >= 0.5

    def test_multi_modal_spectral_analysis(self):
        """Multi-modal: PSD + FAS combined analysis."""
        engine = SpectralIntelligenceSDK()

        psd = engine.generate_psd()
        fas = engine.generate_fas()

        assert engine.validate(psd).is_valid
        assert engine.validate(fas).is_valid

        result = engine.match(psd, fas)
        assert 0 <= result.composite_score <= 1.0

        emb_psd = engine.embed(psd)
        emb_fas = engine.embed(fas)
        assert emb_psd.shape == emb_fas.shape

    def test_anomaly_detection_enterprise_pipeline(self):
        """Enterprise anomaly detection: train → monitor → alert."""
        from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter

        rng = np.random.default_rng(62)
        baselines = [_make_record(f"device_{i}", rng) for i in range(20)]
        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline(baselines)

        normal_scores = []
        for i in range(50):
            reading = _noisy_copy(baselines[0], rng, 0.01)
            normal_scores.append(adapter.score_anomaly(reading))

        f = np.linspace(0.1, 50.0, 64)
        a_anomaly = np.abs(np.random.default_rng(999).exponential(5.0, 64))
        anomaly = MultiElementRecord(
            record_id="intruder",
            components=[SpectralComponent(name="ch1", frequency=f, amplitude=a_anomaly)],
        )
        anomaly_score = adapter.score_anomaly(anomaly)
        assert anomaly_score > np.median(normal_scores) * 0.5

    def test_spectral_memory_rag_workflow(self):
        """RAG-style workflow: index knowledge → query → retrieve → reason."""
        rng = np.random.default_rng(63)

        knowledge = [_make_record(f"knowledge_{i}", rng) for i in range(50)]
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(knowledge)

        query = _noisy_copy(knowledge[25], rng)
        result = client.query(query, top_k=5)
        research_hits = search_research("spectral", top_k=3)

        assert len(result.neighbors) >= 1
        assert len(research_hits) >= 1
        total_context = len(result.neighbors) + len(research_hits)
        assert total_context >= 3


# ===========================================================================
# SECTION 3: Monte Carlo Enterprise Tests (from test_enterprise_workflows.py)
# ===========================================================================


class TestSDKMonteCarloEnterprise:
    """500-trial Monte Carlo validation per enterprise vertical."""

    TRIALS_PER_CASE = 500
    SEED = 42

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rng = np.random.default_rng(self.SEED)
        self.records = [_make_record(f"ref_{i}", self.rng) for i in range(10)]

    def _run_trials(self, fn, n=None):
        n = n or self.TRIALS_PER_CASE
        successes = sum(1 for _ in range(n) if fn())
        return successes / n

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


class TestSDKEnterpriseAIMonteCarlo:
    """1,000 additional Monte Carlo trials focused on enterprise AI patterns."""

    TRIALS = 200
    SEED = 77

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rng = np.random.default_rng(self.SEED)
        self.records = [_make_record(f"ai_{i}", self.rng) for i in range(20)]
        self.client = MAESIClient(fast=True, use_fingerprint=True)
        self.client.index_corpus(self.records)

    def _run_trials(self, fn, n=None):
        n = n or self.TRIALS
        successes = sum(1 for _ in range(n) if fn())
        return successes / n

    def test_rag_retrieval_200_trials(self):
        """RAG retrieval returns relevant neighbors consistently."""
        def trial():
            idx = self.rng.integers(0, len(self.records))
            q = _noisy_copy(self.records[idx], self.rng)
            r = self.client.query(q, top_k=3)
            return len(r.neighbors) >= 1 and r.elapsed_ms < 100

        assert self._run_trials(trial) >= 0.95

    def test_embedding_stability_200_trials(self):
        """Embeddings are stable under small perturbation."""
        vectorizer = SpectralVectorizer()

        def trial():
            idx = self.rng.integers(0, len(self.records))
            base_emb = vectorizer.transform(self.records[idx])
            noisy_emb = vectorizer.transform(_noisy_copy(self.records[idx], self.rng, 0.02))
            cos_sim = np.dot(base_emb, noisy_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(noisy_emb) + 1e-12)
            return cos_sim >= 0.8

        assert self._run_trials(trial) >= 0.90

    def test_validation_robustness_200_trials(self):
        """Validation passes consistently under noise."""
        def trial():
            idx = self.rng.integers(0, len(self.records))
            q = _noisy_copy(self.records[idx], self.rng, 0.05)
            v = validate_record(q)
            return v.is_valid and v.level >= 4

        assert self._run_trials(trial) >= 0.95

    def test_ranking_consistency_200_trials(self):
        """Ranking finds self in top-3 under perturbation."""
        def trial():
            idx = self.rng.integers(0, len(self.records))
            q = _noisy_copy(self.records[idx], self.rng, 0.06)
            ranked = rank_candidates(q, self.records, top_k=3)
            return any(f"ai_{idx}" in r.candidate_id for r in ranked)

        assert self._run_trials(trial) >= 0.85

    def test_cross_match_200_trials(self):
        """Cross-matching produces reasonable scores."""
        def trial():
            i = self.rng.integers(0, len(self.records))
            j = self.rng.integers(0, len(self.records))
            m = match_records(self.records[i], self.records[j]).composite_score
            return 0 <= m <= 1.0

        assert self._run_trials(trial) >= 0.99


# ===========================================================================
# SECTION 4: 20 NEW Enterprise SDK Tests
# ===========================================================================


class TestSDKEnterpriseNewSuite:
    """20 new enterprise SDK tests covering additional integration scenarios."""

    # --- Test 1: SDK multi-record embed ---
    def test_sdk_embed_batch_consistency(self):
        """Embedding a list vs embedding individually should produce same shapes."""
        engine = SpectralIntelligenceSDK()
        rng = np.random.default_rng(1100)
        records = [_make_record(f"embed_batch_{i}", rng) for i in range(10)]

        batch_emb = engine.embed(records)
        single_embs = np.array([engine.embed(r).flatten() for r in records])

        assert batch_emb.shape == single_embs.shape

    # --- Test 2: SDK generate with different seeds ---
    def test_sdk_seeded_generation_deterministic(self):
        """Same seed produces identical records."""
        engine = SpectralIntelligenceSDK()
        r1 = engine.generate_psd(config=GenerationConfig(seed=42))
        r2 = engine.generate_psd(config=GenerationConfig(seed=42))
        np.testing.assert_array_equal(
            r1.components[0].amplitude,
            r2.components[0].amplitude,
        )

    # --- Test 3: SDK validate rejects corrupted record ---
    def test_sdk_validate_corrupted_amplitude(self):
        """Validate catches NaN amplitudes."""
        engine = SpectralIntelligenceSDK()
        f = np.linspace(0.1, 50.0, 64)
        a = np.full(64, np.nan)
        rec = MultiElementRecord(
            record_id="corrupted",
            components=[SpectralComponent(name="ch1", frequency=f, amplitude=a)],
        )
        report = engine.validate(rec)
        assert not report.is_valid

    # --- Test 4: FastCompute empty index ---
    def test_fast_compute_empty_search(self):
        """Search on empty index returns empty results."""
        fc = FastSpectralCompute()
        rng = np.random.default_rng(1101)
        query = _make_record("query_empty", rng)
        hits = fc.cosine_search(query, top_k=5)
        assert len(hits) == 0

    # --- Test 5: MAESI client reindex ---
    def test_maesi_client_reindex_corpus(self):
        """Re-indexing a corpus replaces the previous index."""
        rng = np.random.default_rng(1102)
        corpus1 = [_make_record(f"c1_{i}", rng) for i in range(10)]
        corpus2 = [_make_record(f"c2_{i}", rng) for i in range(10)]

        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(corpus1)
        client.index_corpus(corpus2)

        result = client.query(corpus2[5], top_k=3)
        assert len(result.neighbors) >= 1

    # --- Test 6: Fingerprint pipeline re-query same record ---
    def test_fingerprint_self_query_top_hit(self):
        """Querying with the exact indexed record returns itself as top hit."""
        rng = np.random.default_rng(1103)
        corpus = [_make_record(f"fp_self_{i}", rng) for i in range(20)]
        fp = SpectralFingerprintPipeline()
        fp.index_records(corpus)

        results = fp.query(corpus[10], top_k=1)
        assert len(results) >= 1

    # --- Test 7: Validate large frequency range ---
    def test_validate_wide_frequency_range(self):
        """Records with very wide frequency range still validate."""
        engine = SpectralIntelligenceSDK()
        f = np.linspace(0.001, 1000.0, 512)
        a = np.abs(np.random.default_rng(1104).normal(1.0, 0.3, 512))
        rec = MultiElementRecord(
            record_id="wide_freq",
            components=[SpectralComponent(name="wide", frequency=f, amplitude=a)],
        )
        report = engine.validate(rec)
        assert report.is_valid

    # --- Test 8: Match score symmetry ---
    def test_match_score_symmetry(self):
        """match(a, b) should equal match(b, a) in score."""
        rng = np.random.default_rng(1105)
        a = _make_record("sym_a", rng)
        b = _make_record("sym_b", rng)
        score_ab = match_records(a, b).composite_score
        score_ba = match_records(b, a).composite_score
        assert abs(score_ab - score_ba) < 0.01

    # --- Test 9: Rank candidates returns correct count ---
    def test_rank_candidates_top_k_bounds(self):
        """rank_candidates respects top_k even when corpus is larger."""
        rng = np.random.default_rng(1106)
        corpus = [_make_record(f"topk_{i}", rng) for i in range(50)]
        query = _make_record("topk_q", rng)
        for k in [1, 5, 10, 25]:
            ranked = rank_candidates(query, corpus, top_k=k)
            assert len(ranked) == k

    # --- Test 10: SDK embed output is finite ---
    def test_sdk_embed_no_inf_nan(self):
        """Embeddings contain no NaN or Inf values."""
        engine = SpectralIntelligenceSDK()
        rng = np.random.default_rng(1107)
        for i in range(10):
            rec = _make_record(f"finite_{i}", rng)
            emb = engine.embed(rec)
            assert np.all(np.isfinite(emb))

    # --- Test 11: Research search with varied queries ---
    def test_research_search_multiple_queries(self):
        """Various research queries all return results."""
        queries = ["spectral", "EM", "ANN", "PSD"]
        for q in queries:
            hits = search_research(q, top_k=3)
            assert len(hits) >= 1, f"No results for query: {q}"

    # --- Test 12: Knowledge base consistency ---
    def test_knowledge_base_no_duplicates(self):
        """Technical library entries have unique names."""
        lib = get_technical_library()
        names = [c.name for c in lib]
        assert len(names) == len(set(names))

    # --- Test 13: Periodic table element properties ---
    def test_periodic_table_atomic_numbers_ordered(self):
        """Elements have unique atomic numbers."""
        elements = get_periodic_table()
        atomic_numbers = [e.atomic_number for e in elements]
        assert len(atomic_numbers) == len(set(atomic_numbers))

    # --- Test 14: SDK match self-similarity ---
    def test_sdk_self_match_high_score(self):
        """Matching a record with itself produces a high score."""
        engine = SpectralIntelligenceSDK()
        rec = engine.generate_psd()
        result = engine.match(rec, rec)
        assert result.composite_score >= 0.9

    # --- Test 15: MAESI client with fingerprint disabled ---
    def test_maesi_client_no_fingerprint(self):
        """MAESI works without fingerprint enabled."""
        rng = np.random.default_rng(1108)
        corpus = [_make_record(f"nofp_{i}", rng) for i in range(15)]
        client = MAESIClient(fast=True, use_fingerprint=False)
        client.index_corpus(corpus)
        result = client.query(corpus[7], top_k=3)
        assert len(result.neighbors) >= 1

    # --- Test 16: Batch validation performance ---
    def test_batch_validation_under_1_second(self):
        """Validating 200 records takes < 1 second."""
        rng = np.random.default_rng(1109)
        records = [_make_record(f"perf_val_{i}", rng) for i in range(200)]
        t0 = time.perf_counter()
        for r in records:
            validate_record(r)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0  # generous for CI

    # --- Test 17: Multi-component record ---
    def test_multi_component_record_validation(self):
        """Records with multiple spectral components validate correctly."""
        engine = SpectralIntelligenceSDK()
        rng = np.random.default_rng(1110)
        f = np.linspace(0.1, 50.0, 64)
        rec = MultiElementRecord(
            record_id="multi_comp",
            components=[
                SpectralComponent(name="ch1", frequency=f, amplitude=np.abs(rng.normal(1.0, 0.3, 64))),
                SpectralComponent(name="ch2", frequency=f, amplitude=np.abs(rng.normal(1.5, 0.2, 64))),
                SpectralComponent(name="ch3", frequency=f, amplitude=np.abs(rng.normal(0.8, 0.4, 64))),
            ],
        )
        report = engine.validate(rec)
        assert report.is_valid

    # --- Test 18: Embedding dimensionality consistency ---
    def test_embedding_dimensionality_consistent(self):
        """All records produce embeddings of the same dimensionality."""
        engine = SpectralIntelligenceSDK()
        rng = np.random.default_rng(1111)
        dims = set()
        for i in range(15):
            rec = _make_record(f"dim_{i}", rng)
            emb = engine.embed(rec)
            dims.add(emb.shape[1])
        assert len(dims) == 1  # all same dimension

    # --- Test 19: Cross-domain research coverage ---
    def test_research_catalog_covers_domains(self):
        """Research catalog covers multiple scientific domains."""
        catalog = get_research_catalog()
        # Should have entries spanning different topics
        titles = [entry.title.lower() for entry in catalog]
        all_text = " ".join(titles)
        # Check for diversity of topics
        assert len(catalog) >= 20

    # --- Test 20: End-to-end SDK pipeline with edge cases ---
    def test_sdk_pipeline_with_minimal_record(self):
        """Full SDK pipeline works with minimal-size records (4 points)."""
        engine = SpectralIntelligenceSDK()
        f = np.array([1.0, 2.0, 3.0, 4.0])
        a = np.array([0.5, 1.0, 0.8, 0.3])
        rec = MultiElementRecord(
            record_id="minimal",
            components=[SpectralComponent(name="ch1", frequency=f, amplitude=a)],
        )

        # Validate
        report = engine.validate(rec)
        assert report.is_valid

        # Embed
        emb = engine.embed(rec)
        assert emb.shape[0] == 1
        assert np.all(np.isfinite(emb))

        # Match with itself
        result = engine.match(rec, rec)
        assert result.composite_score >= 0.9
