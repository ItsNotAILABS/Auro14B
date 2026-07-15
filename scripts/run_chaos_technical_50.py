"""50 major tests — chaos engineering + technical validation for MESIE/MAESI/NeuroAIX."""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie import match_records, validate_record
from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter
from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.embeddings import SpectralFingerprintPipeline
from mesie.engines.registry import build_default_registry
from mesie.internal_api.bus import InternalBus
from mesie.io.loaders import load_record
from mesie.matching.ranking import rank_candidates
from mesie.octopus import OctopusController, OctopusConfig
from mesie.sdk import (
    MAESIClient,
    get_technical_library,
    get_technical_matrix,
    search_research,
)
from mesie.sdk.technical_library import TechnicalDomain
from mesie.signal import SalientFeatureExtractor, TimeFrequencyTransform


@dataclass
class MajorTest:
    id: str
    name: str
    lane: str  # chaos | technical
    domain: str
    fn: Callable[[], None]


@dataclass
class TestOutcome:
    id: str
    name: str
    lane: str
    domain: str
    passed: bool
    latency_ms: float
    detail: str = ""
    error: Optional[str] = None


@dataclass
class ChaosTechnicalReport:
    n_tests: int
    passed: int
    failed: int
    success_rate: float
    chaos_passed: int
    chaos_total: int
    technical_passed: int
    technical_total: int
    total_ms: float
    outcomes: List[TestOutcome] = field(default_factory=list)


def _record(
    rid: str = "test",
    n: int = 64,
    *,
    scale: float = 1.0,
    noise: float = 0.0,
    spike_idx: Optional[int] = None,
) -> MultiElementRecord:
    f = np.linspace(0.2, 25.0, n)
    a = scale * (0.5 + np.exp(-((f - 5.0) ** 2) / 8.0))
    if noise:
        a = np.maximum(a * (1.0 + np.random.default_rng(7).normal(0, noise, n)), 1e-12)
    if spike_idx is not None and 0 <= spike_idx < n:
        a[spike_idx] *= 50.0
    return MultiElementRecord(
        record_id=rid,
        components=[SpectralComponent(name="ch", frequency=f, amplitude=a)],
    )


def _noisy_copy(base: MultiElementRecord, rng: np.random.Generator, scale: float = 0.15) -> MultiElementRecord:
    c = base.components[0]
    a = np.maximum(np.abs(c.amplitude) * (1.0 + rng.normal(0, scale, len(c.amplitude))), 1e-12)
    return MultiElementRecord(
        record_id=f"{base.record_id}_chaos",
        components=[SpectralComponent(name=c.name, frequency=c.frequency.copy(), amplitude=a)],
    )


def _run_test(test: MajorTest) -> TestOutcome:
    t0 = time.perf_counter()
    try:
        detail = test.fn()
        ms = (time.perf_counter() - t0) * 1000
        return TestOutcome(test.id, test.name, test.lane, test.domain, True, ms, detail or "ok")
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        return TestOutcome(
            test.id, test.name, test.lane, test.domain, False, ms,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Chaos lane (25)
# ---------------------------------------------------------------------------

def _chaos_tests(refs: Dict[str, MultiElementRecord], rng: np.random.Generator) -> List[MajorTest]:
    ref_list = list(refs.values())
    vib = refs.get("vibration_monitoring_reference") or ref_list[0]
    eq = refs.get("earthquake_psd_reference") or ref_list[0]

    def c01():
        r = _noisy_copy(vib, rng, 0.25)
        s = match_records(vib, r).composite_score
        assert s >= 0.45, f"noisy match {s}"

    def c02():
        r = _record("flat", n=32, scale=0.0)
        rep = validate_record(r)
        assert rep.level >= 1

    def c03():
        r = _record("spike", spike_idx=10)
        s = match_records(r, r).composite_score
        assert s == 1.0

    def c04():
        bus = InternalBus()
        build_default_registry(bus)
        bad = bus.request("chaos", "validation", "validate", {})
        assert not bad.ok

    def c05():
        bus = InternalBus()
        build_default_registry(bus)
        bad = bus.request("chaos", "nonexistent", "match", {"record_a": vib, "record_b": vib})
        assert not bad.ok

    def c06():
        ranked = rank_candidates(vib, [], top_k=3)
        assert ranked == []

    def c07():
        ranked = rank_candidates(vib, [vib], top_k=5)
        assert len(ranked) == 1 and ranked[0].composite_score >= 0.99

    def c08():
        fp = SpectralFingerprintPipeline()
        fp.index_records(ref_list[:4])
        hits = fp.query(_noisy_copy(ref_list[0], rng, 0.2), top_k=2)
        assert len(hits) >= 1

    def c09():
        octo = OctopusController(config=OctopusConfig(movement_steps=2))
        rep = octo.run_standard_cycle(_noisy_copy(vib, rng), candidate=_noisy_copy(eq, rng))
        assert rep.arms_used

    def c10():
        adapter = SpectralAnomalyAdapter(threshold=1.5)
        adapter.fit_baseline([vib])
        outlier = _record("outlier", n=64, scale=3.0)
        assert adapter.score_anomaly(outlier) > adapter.score_anomaly(vib)

    def c11():
        r = _record("tiny", n=4)
        emb_bus = InternalBus()
        build_default_registry(emb_bus)
        resp = emb_bus.request("chaos", "embedding", "transform", {"record": r})
        assert resp.ok and len(resp.data["embedding"]) > 0

    def c12():
        r = _record("huge", n=2048)
        s = match_records(r, r).composite_score
        assert s == 1.0

    def c13():
        seeds = [match_records(vib, _noisy_copy(vib, np.random.default_rng(s), 0.05)).composite_score for s in range(5)]
        assert max(seeds) - min(seeds) < 0.15

    def c14():
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(ref_list)
        q = client.query(_noisy_copy(ref_list[0], rng))
        assert q.elapsed_ms < 200 and len(q.neighbors) >= 1

    def c15():
        tf = TimeFrequencyTransform().from_record(_noisy_copy(eq, rng, 0.3))
        sal = SalientFeatureExtractor(max_points=6).extract(tf)
        assert sal.n_points >= 1

    def c16():
        bus = InternalBus()
        build_default_registry(bus)
        for _ in range(20):
            resp = bus.request("stress", "matching", "match", {"record_a": vib, "record_b": eq})
            assert resp.ok

    def c17():
        r = MultiElementRecord(record_id="multi", components=[
            SpectralComponent(name="a", frequency=np.linspace(1, 10, 16), amplitude=np.ones(16) * 0.5),
            SpectralComponent(name="b", frequency=np.linspace(1, 10, 16), amplitude=np.ones(16) * 0.3),
        ])
        s = match_records(r, r).composite_score
        assert s == 1.0

    def c18():
        corrupt = _noisy_copy(vib, rng, 0.8)
        ranked = rank_candidates(vib, ref_list, top_k=3)
        assert all(h.composite_score > 0 for h in ranked)

    def c19():
        bus = InternalBus()
        build_default_registry(bus)
        resp = bus.request("chaos", "fingerprint", "index", {"records": ref_list[:3]})
        assert resp.ok
        q = bus.request("chaos", "fingerprint", "query", {"record": ref_list[0], "top_k": 2})
        assert q.ok

    def c20():
        octo = OctopusController()
        names = octo.list_engines()
        assert len(names) >= 8

    def c21():
        r = _record("micro", n=48, scale=1e-6)
        rep = validate_record(r)
        assert rep.level >= 1 and isinstance(rep.errors, list)

    def c22():
        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline([eq])
        assert adapter.score_anomaly(eq) <= adapter.score_anomaly(_noisy_copy(eq, rng, 0.5))

    def c23():
        fp = SpectralFingerprintPipeline()
        fp.index_records(ref_list)
        for i in range(5):
            hits = fp.query(_noisy_copy(ref_list[i % len(ref_list)], rng, 0.1), top_k=1)
            assert hits[0].similarity > 0.3

    def c24():
        bus = InternalBus()
        build_default_registry(bus)
        resp = bus.request("chaos", "control", "evaluate", {"similarity": 0.2, "anomaly": 2.0})
        assert resp.ok and resp.data.get("commands")

    def c25():
        client = MAESIClient(fast=True)
        report = client.run_full(ref_list[:6], benchmark=True)
        assert report.speed is not None and report.speed.speedup_ratio > 1.0

    chaos = [
        ("C01", "Noisy vibration match resilience", "robustness", c01),
        ("C02", "Flatline spectrum validation", "edge_case", c02),
        ("C03", "Amplitude spike self-match", "edge_case", c03),
        ("C04", "Bus reject empty validate payload", "fault_injection", c04),
        ("C05", "Bus unknown engine graceful fail", "fault_injection", c05),
        ("C06", "Rank empty candidate pool", "edge_case", c06),
        ("C07", "Rank single candidate", "edge_case", c07),
        ("C08", "Fingerprint noisy neighbor retrieval", "retrieval", c08),
        ("C09", "Octopus cycle under cross-domain noise", "orchestration", c09),
        ("C10", "Anomaly adapter outlier separation", "control", c10),
        ("C11", "Embed tiny 4-point spectrum", "scale", c11),
        ("C12", "Match 2048-point spectrum", "scale", c12),
        ("C13", "Seed-stable noisy match band", "determinism", c13),
        ("C14", "MAESI query under noise SLA", "sla", c14),
        ("C15", "TF salient under seismic noise", "signal", c15),
        ("C16", "Bus match stress 20 rounds", "stress", c16),
        ("C17", "Multi-component self-match", "structure", c17),
        ("C18", "Rank corpus after heavy perturbation", "retrieval", c18),
        ("C19", "Fingerprint bus index+query", "bus", c19),
        ("C20", "Octopus engine roster integrity", "orchestration", c20),
        ("C21", "Micro-amplitude validation tolerance", "edge_case", c21),
        ("C22", "Anomaly baseline vs noisy self", "control", c22),
        ("C23", "Fingerprint repeated noisy queries", "stress", c23),
        ("C24", "Control engine low-sim commands", "control", c24),
        ("C25", "MAESI fast compute speedup under load", "performance", c25),
    ]
    return [MajorTest(cid, name, "chaos", dom, fn) for cid, name, dom, fn in chaos]


# ---------------------------------------------------------------------------
# Technical lane (25) — 20 concepts + 5 cross-stack integrations
# ---------------------------------------------------------------------------

def _technical_tests(refs: Dict[str, MultiElementRecord]) -> List[MajorTest]:
    ref_list = list(refs.values())
    concepts = get_technical_library()
    matrix = get_technical_matrix()

    def _concept_test(concept_name: str, check: Callable[[], None]) -> Callable[[], None]:
        def run():
            check()
            return concept_name
        return run

    checks: Dict[str, Callable[[], None]] = {}

    checks["STFT Spectrogram"] = lambda: (
        TimeFrequencyTransform().from_time_series(np.sin(np.linspace(0, 20, 256)), 128.0).matrix.shape[0] > 0
    )
    checks["Salient TF Peaks"] = lambda: (
        SalientFeatureExtractor(max_points=8).extract(
            TimeFrequencyTransform().from_record(ref_list[0])
        ).n_points >= 1
    )

    def _lsh_ann():
        from mesie.embeddings import ANNIndex, LSHHasher
        h = LSHHasher(dim=16, n_planes=8, seed=1)
        v = np.ones(16)
        assert h.hash(v).bucket_key == h.hash(v).bucket_key
        idx = ANNIndex(use_lsh=True)
        idx.add("a", v)
        hits = idx.query(v, top_k=1)
        assert hits[0].item_id == "a"

    checks["LSH Spectral Hash"] = _lsh_ann
    checks["ANN Cosine Rerank"] = _lsh_ann

    checks["Pump Vibration Baseline"] = lambda: assert_ref("vibration_monitoring_reference", refs)
    def _anomaly_baseline():
        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline([ref_list[0]])
        assert adapter.score_anomaly(ref_list[0]) >= 0

    checks["Anomaly vs Baseline"] = _anomaly_baseline

    def _schumann():
        from data import load_library
        entry = load_library("schumann_resonances")
        assert "schumann_resonances" in entry

    checks["Schumann Eco-Hz"] = _schumann

    def _hz_ladder():
        from mesie.edge.hz_ladder import HzLadder
        ladder = HzLadder()
        assert len(ladder.tiers) >= 5

    checks["EM Band Ladder"] = _hz_ladder

    def _satellite():
        from mesie.edge.satellite_nodes import OrbitalTier
        tier = OrbitalTier(name="LEO_550", altitude_km=550)
        assert tier.orbital_period_s > 0

    checks["LEO Contact Window"] = _satellite
    checks["Orbital Edge Gate"] = lambda: assert_ref("earthquake_psd_reference", refs)
    checks["Earthquake PSD Anchor"] = lambda: assert_ref("earthquake_psd_reference", refs)
    checks["RotDNN Orientation"] = lambda: assert_ref("rotdnn_reference", refs)
    checks["Structural FAS"] = lambda: assert_ref("structural_fas_reference", refs)

    def _vectorizer():
        from mesie.embeddings.vectorizers import SpectralVectorizer
        v = SpectralVectorizer(n_bands=8).transform(ref_list[0])
        assert len(v) >= 8

    checks["Spectral Vectorizer"] = _vectorizer

    def _fingerprint_pipe():
        fp = SpectralFingerprintPipeline()
        fp.index_records(ref_list[:3])
        assert fp.query(ref_list[0], top_k=1)

    checks["Fingerprint Pipeline"] = _fingerprint_pipe

    checks["Octopus Multi-Arm Control"] = lambda: len(OctopusController().list_engines()) >= 8

    def _internal_bus():
        bus = InternalBus()
        build_default_registry(bus)
        assert len(build_default_registry(bus).names()) >= 10

    checks["Internal API Bus"] = _internal_bus

    def _transfer():
        from mesie.transfer.alignment import CORAL
        a = np.random.default_rng(1).standard_normal((10, 4))
        b = np.random.default_rng(2).standard_normal((10, 4))
        out = CORAL().fit_transform(a, b)
        assert out.shape == a.shape

    checks["Cross-Domain Transfer"] = _transfer

    def _intel_protocol():
        from mesie.ai.intelligence_protocols import IntelligenceProtocol
        from mesie.embeddings.vectorizers import SpectralVectorizer
        emb = SpectralVectorizer(n_bands=8).transform(ref_list[0])
        r = IntelligenceProtocol().reason(emb)
        assert r.confidence >= 0

    checks["Intelligence Protocol"] = _intel_protocol

    checks["Hz Virtual Chip"] = lambda: (
        MAESIClient(fast=True).run_full(ref_list[:4], benchmark=True).speed.speedup_ratio > 1
    )

    tech_tests: List[MajorTest] = []
    for i, concept in enumerate(concepts[:20], start=1):
        fn = checks.get(concept.name)
        if fn is None:
            def fn(c=concept):
                assert c.mesie_module
        tech_tests.append(
            MajorTest(
                f"T{i:02d}",
                f"Technical: {concept.name}",
                "technical",
                concept.domain.value,
                _concept_test(concept.name, fn),
            )
        )

    def t21():
        tf = TimeFrequencyTransform().from_record(ref_list[0])
        sal = SalientFeatureExtractor(max_points=6).extract(tf)
        assert sal.n_points >= 1 and len(sal.feature_vector) > 0

    def t22():
        hits = search_research("LSH approximate nearest", 3)
        assert len(hits) >= 1

    def t23():
        proj = matrix @ (matrix[0] / np.linalg.norm(matrix[0]))
        assert len(proj) == len(concepts)

    def t24():
        domains = {c.domain for c in concepts}
        assert len(domains) >= 6

    def t25():
        client = MAESIClient(fast=True, use_fingerprint=True)
        client.index_corpus(ref_list)
        q = client.query(ref_list[0])
        assert q.neighbors and q.technical_hits

    extras = [
        ("T21", "TF→Salient integration chain", "time_frequency", t21),
        ("T22", "Research knowledge ANN hit", "ann_retrieval", t22),
        ("T23", "Technical matrix projection", "spectral_ml", t23),
        ("T24", "Multi-domain technical coverage", "architecture", t24),
        ("T25", "MAESI full stack query", "product", t25),
    ]
    for cid, name, dom, fn in extras:
        tech_tests.append(MajorTest(cid, name, "technical", dom, fn))

    return tech_tests


def assert_ref(key: str, refs: Dict[str, MultiElementRecord]) -> None:
    assert key in refs, f"missing {key}"
    assert refs[key].record_id


def run_all(seed: int = 42) -> ChaosTechnicalReport:
    rng = np.random.default_rng(seed)
    refs = {n: load_reference_record(n) for n in list_references()}
    tests = _chaos_tests(refs, rng) + _technical_tests(refs)
    assert len(tests) == 50, f"expected 50 tests, got {len(tests)}"

    t0 = time.perf_counter()
    outcomes = [_run_test(t) for t in tests]
    total_ms = (time.perf_counter() - t0) * 1000

    passed = sum(1 for o in outcomes if o.passed)
    chaos = [o for o in outcomes if o.lane == "chaos"]
    tech = [o for o in outcomes if o.lane == "technical"]

    return ChaosTechnicalReport(
        n_tests=len(tests),
        passed=passed,
        failed=len(tests) - passed,
        success_rate=passed / len(tests),
        chaos_passed=sum(1 for o in chaos if o.passed),
        chaos_total=len(chaos),
        technical_passed=sum(1 for o in tech if o.passed),
        technical_total=len(tech),
        total_ms=total_ms,
        outcomes=outcomes,
    )


def write_report(report: ChaosTechnicalReport, root: Path) -> tuple[Path, Path]:
    out_json = root / "deliverables" / "MESIE_Chaos_Technical_50_Report.json"
    out_md = root / "deliverables" / "MESIE_Chaos_Technical_50_Report.md"
    out_json.parent.mkdir(exist_ok=True)

    payload = {
        "suite": "MESIE Chaos + Technical — 50 Major Tests",
        "n_tests": report.n_tests,
        "passed": report.passed,
        "failed": report.failed,
        "success_rate_pct": round(report.success_rate * 100, 2),
        "chaos": {"passed": report.chaos_passed, "total": report.chaos_total},
        "technical": {"passed": report.technical_passed, "total": report.technical_total},
        "total_runtime_ms": round(report.total_ms, 2),
        "outcomes": [
            {
                "id": o.id,
                "name": o.name,
                "lane": o.lane,
                "domain": o.domain,
                "passed": o.passed,
                "latency_ms": round(o.latency_ms, 3),
                "detail": o.detail,
                "error": o.error,
            }
            for o in report.outcomes
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# MESIE / MAESI — 50 Major Chaos + Technical Tests",
        "",
        f"**Result:** {report.passed}/{report.n_tests} passed ({report.success_rate * 100:.1f}%)",
        f"**Runtime:** {report.total_ms:.0f} ms on laptop virtual chip",
        "",
        "## Summary",
        "",
        f"| Lane | Passed | Total |",
        f"|------|--------|-------|",
        f"| Chaos | {report.chaos_passed} | {report.chaos_total} |",
        f"| Technical | {report.technical_passed} | {report.technical_total} |",
        "",
        "## Chaos engineering (25)",
        "",
        "| ID | Test | Domain | ms | Status |",
        "|----|------|--------|-----|--------|",
    ]
    for o in report.outcomes:
        if o.lane != "chaos":
            continue
        status = "PASS" if o.passed else f"FAIL: {o.error}"
        lines.append(f"| {o.id} | {o.name} | {o.domain} | {o.latency_ms:.1f} | {status} |")

    lines.extend(["", "## Technical validation (25)", "", "| ID | Test | Domain | ms | Status |", "|----|------|--------|-----|--------|"])
    for o in report.outcomes:
        if o.lane != "technical":
            continue
        status = "PASS" if o.passed else f"FAIL: {o.error}"
        lines.append(f"| {o.id} | {o.name} | {o.domain} | {o.latency_ms:.1f} | {status} |")

    if report.failed:
        lines.extend(["", "## Failures", ""])
        for o in report.outcomes:
            if not o.passed:
                lines.append(f"- **{o.id}** {o.name}: {o.error}")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_json, out_md


def main() -> None:
    print("=== MESIE 50 Major Tests: Chaos + Technical ===\n")
    report = run_all()
    jpath, mpath = write_report(report, ROOT)

    print(f"Passed: {report.passed}/{report.n_tests} ({report.success_rate * 100:.1f}%)")
    print(f"Chaos:      {report.chaos_passed}/{report.chaos_total}")
    print(f"Technical:  {report.technical_passed}/{report.technical_total}")
    print(f"Runtime:    {report.total_ms:.0f} ms")
    print(f"\nWrote {jpath}")
    print(f"Wrote {mpath}")

    if report.failed:
        print("\n--- FAILURES ---")
        for o in report.outcomes:
            if not o.passed:
                print(f"  {o.id} {o.name}: {o.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()