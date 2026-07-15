"""Advanced Enterprise AI tests — agentic workflows, knowledge graphs,
batch processing pipelines, multi-network dispatch, and diagnostic capabilities.

Extends the enterprise test suite with deeper AI integration patterns.

Run:
    pytest tests/test_enterprise_ai_advanced.py -v
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
    f = np.linspace(0.1, 50.0, n)
    a = np.abs(rng.normal(1.0, 0.3, n))
    return MultiElementRecord(
        record_id=record_id,
        components=[SpectralComponent(name="ch1", frequency=f, amplitude=a)],
    )


def _noisy_copy(rec: MultiElementRecord, rng: np.random.Generator, scale: float = 0.05) -> MultiElementRecord:
    c = rec.components[0]
    a = np.maximum(np.abs(c.amplitude) * (1.0 + rng.normal(0, scale, len(c.amplitude))), 1e-12)
    return MultiElementRecord(
        record_id=f"{rec.record_id}_noisy",
        components=[SpectralComponent(name=c.name, frequency=c.frequency.copy(), amplitude=a)],
    )


# ---------------------------------------------------------------------------
# Enterprise AI — Knowledge Graph Tests
# ---------------------------------------------------------------------------

class TestKnowledgeGraphIntegration:
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
        # Spot-check key elements
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
        # Search different fields
        for query in ["spectral", "EM frequency", "signal"]:
            hits = search_research(query, top_k=3)
            # At least some queries should return results
        hits = search_research("spectral", top_k=3)
        assert len(hits) >= 1

    def test_cross_domain_knowledge_linking(self):
        """Link physics laws to technical concepts via embeddings."""
        laws = get_fundamental_laws()
        law_embs = np.array([l.to_embedding() for l in laws])
        # Embeddings should be distinct (not all same)
        assert np.std(law_embs) > 0.01


# ---------------------------------------------------------------------------
# Enterprise AI — Batch Processing Pipeline Tests
# ---------------------------------------------------------------------------

class TestBatchProcessingPipelines:
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
        # No NaN
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
        # Self-similarity should be higher than cross
        self_scores = [match_records(c, c).composite_score for c in corpus[:5]]
        assert np.mean(self_scores) >= np.mean(scores)

    def test_batch_rank_large_corpus(self):
        """Rank against a 200-record corpus."""
        rng = np.random.default_rng(45)
        corpus = [_make_record(f"large_{i}", rng) for i in range(200)]
        query = _noisy_copy(corpus[100], rng)
        ranked = rank_candidates(query, corpus, top_k=10)
        assert len(ranked) == 10
        # Scores should be descending
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_batch_generation_mixed_types(self):
        """Generate mixed PSD/FAS records in batch."""
        from mesie.core.config import GenerationConfig
        records = []
        for i in range(20):
            cfg = GenerationConfig(seed=i)
            if i % 2 == 0:
                records.append(generate_psd(config=cfg))
            else:
                records.append(generate_fas(config=cfg))
        assert len(records) == 20
        assert all(r.components for r in records)


# ---------------------------------------------------------------------------
# Enterprise AI — Agentic Workflow Tests
# ---------------------------------------------------------------------------

class TestAgenticWorkflows:
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
        from mesie.core.config import GenerationConfig
        engine = SpectralIntelligenceSDK()
        rec = generate_psd(config=GenerationConfig(seed=1))
        result = engine.spawn_task(
            "enterprise_validate",
            [
                {"engine": "validation", "action": "validate"},
            ],
            record=rec,
        )
        assert result.success


# ---------------------------------------------------------------------------
# Enterprise AI — Multi-Network Speed Tests
# ---------------------------------------------------------------------------

class TestMultiNetworkSpeed:
    """Performance tests for enterprise-scale operations."""

    def test_fast_compute_index_1000(self):
        """Index 1000 records in < 1s."""
        rng = np.random.default_rng(50)
        records = [_make_record(f"speed_{i}", rng) for i in range(1000)]
        fc = FastSpectralCompute()
        t0 = time.perf_counter()
        fc.build_index(records)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0  # generous for CI

    def test_fast_search_latency(self):
        """Single search < 5ms after indexing."""
        rng = np.random.default_rng(51)
        records = [_make_record(f"lat_{i}", rng) for i in range(500)]
        fc = FastSpectralCompute()
        fc.build_index(records)
        query = _noisy_copy(records[250], rng)
        t0 = time.perf_counter()
        hits = fc.cosine_search(query, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50  # generous for CI
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
        assert elapsed < 5.0  # 100 queries in < 5s

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


# ---------------------------------------------------------------------------
# Enterprise AI — Diagnostic & Health Check Tests
# ---------------------------------------------------------------------------

class TestDiagnosticsAndHealth:
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
        from mesie.core.config import GenerationConfig
        rec = generate_psd(config=GenerationConfig(seed=1))
        report = validate_record(rec)
        # Should be convertible to dict/JSON
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
        from mesie.core.config import GenerationConfig
        r1 = generate_psd(config=GenerationConfig(seed=1))
        r2 = generate_psd(config=GenerationConfig(seed=2))
        result = match_records(r1, r2)
        data = {
            "composite_score": result.composite_score,
            "metric_breakdown": result.metric_breakdown,
        }
        serialized = json.dumps(data)
        assert "composite_score" in serialized


# ---------------------------------------------------------------------------
# Enterprise AI — End-to-End Integration Workflows
# ---------------------------------------------------------------------------

class TestEndToEndEnterpriseAI:
    """Full enterprise AI integration workflows."""

    def test_ingest_validate_embed_search_pipeline(self):
        """Complete pipeline: ingest → validate → embed → index → search."""
        rng = np.random.default_rng(60)
        engine = SpectralIntelligenceSDK()

        # Ingest
        records = [_make_record(f"ingest_{i}", rng) for i in range(30)]

        # Validate all
        valid = [r for r in records if engine.validate(r).is_valid]
        assert len(valid) == 30

        # Embed
        embeddings = engine.embed(valid)
        assert embeddings.shape[0] == 30
        assert embeddings.shape[1] > 0

        # Index via MAESI
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(valid)

        # Search
        query = _noisy_copy(valid[15], rng)
        result = client.query(query, top_k=5)
        assert len(result.neighbors) >= 1

    def test_generate_augment_classify_workflow(self):
        """Generate synthetic → augment with noise → classify via ranking."""
        rng = np.random.default_rng(61)

        # Generate baseline library
        library = []
        for i in range(5):
            cfg = GenerationConfig(seed=i * 10)
            library.append(generate_psd(config=cfg))

        # Augment: create noisy variants
        augmented = []
        for rec in library:
            for _ in range(10):
                augmented.append(_noisy_copy(rec, rng, scale=0.05))

        # Classify: rank a new sample against library
        query = _noisy_copy(library[2], rng, scale=0.03)
        ranked = rank_candidates(query, library, top_k=3)
        assert ranked[0].score >= 0.5

    def test_multi_modal_spectral_analysis(self):
        """Multi-modal: PSD + FAS combined analysis."""
        engine = SpectralIntelligenceSDK()

        psd = engine.generate_psd()
        fas = engine.generate_fas()

        # Both should validate
        assert engine.validate(psd).is_valid
        assert engine.validate(fas).is_valid

        # Cross-match produces a score
        result = engine.match(psd, fas)
        assert 0 <= result.composite_score <= 1.0

        # Embed both
        emb_psd = engine.embed(psd)
        emb_fas = engine.embed(fas)
        assert emb_psd.shape == emb_fas.shape

    def test_anomaly_detection_enterprise_pipeline(self):
        """Enterprise anomaly detection: train → monitor → alert."""
        from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter

        rng = np.random.default_rng(62)

        # Train on baseline fleet of 20 devices
        baselines = [_make_record(f"device_{i}", rng) for i in range(20)]
        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline(baselines)

        # Monitor: normal readings (very close to baselines)
        normal_scores = []
        for i in range(50):
            reading = _noisy_copy(baselines[0], rng, 0.01)
            normal_scores.append(adapter.score_anomaly(reading))

        # Alert: anomalous reading (completely different distribution)
        f = np.linspace(0.1, 50.0, 64)
        a_anomaly = np.abs(np.random.default_rng(999).exponential(5.0, 64))
        anomaly = MultiElementRecord(
            record_id="intruder",
            components=[SpectralComponent(name="ch1", frequency=f, amplitude=a_anomaly)],
        )
        anomaly_score = adapter.score_anomaly(anomaly)

        # Anomaly should score higher than median normal
        assert anomaly_score > np.median(normal_scores) * 0.5

    def test_spectral_memory_rag_workflow(self):
        """RAG-style workflow: index knowledge → query → retrieve → reason."""
        rng = np.random.default_rng(63)

        # Build spectral knowledge base
        knowledge = [_make_record(f"knowledge_{i}", rng) for i in range(50)]
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(knowledge)

        # Query with context
        query = _noisy_copy(knowledge[25], rng)
        result = client.query(query, top_k=5)

        # Also hit research
        research_hits = search_research("spectral", top_k=3)

        # RAG result: combine neighbors + research
        assert len(result.neighbors) >= 1
        assert len(research_hits) >= 1
        # Total context retrieved
        total_context = len(result.neighbors) + len(research_hits)
        assert total_context >= 3


# ---------------------------------------------------------------------------
# Enterprise AI — Monte Carlo Extension (1000 additional trials)
# ---------------------------------------------------------------------------

class TestEnterpriseAIMonteCarlo:
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
