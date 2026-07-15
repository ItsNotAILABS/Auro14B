"""MESIE SDK test drive — exercises core API + bundled datasets."""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mesie
from data import list_benchmarks, list_references, load_benchmark, load_reference, load_reference_record


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def run(name: str, fn: Callable[[], Any]) -> Check:
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        detail = str(result) if result is not None else "ok"
        if len(detail) > 200:
            detail = detail[:200] + "..."
        return Check(name, True, detail, {"elapsed_s": round(elapsed, 4)})
    except Exception as e:
        return Check(name, False, f"{type(e).__name__}: {e}", {"trace": traceback.format_exc()[-400:]})

def main() -> None:
    from mesie import (
        GenerationConfig,
        generate_fas,
        generate_psd,
        generate_rotdnn,
        load_record,
        match_records,
        normalize_record,
        validate_record,
    )
    from mesie.cognitive import (
        AgentStateSpectralAdapter,
        SpectralAnomalyAdapter,
        SpectralAttentionAdapter,
        SpectralMemoryAdapter,
    )
    from mesie.embeddings import SpectralRetriever, SpectralVectorizer
    from mesie.matching.ranking import rank_candidates

    checks: List[Check] = []

    checks.append(Check("mesie_version", True, mesie.__version__))

    refs = list_references()
    benches = list_benchmarks()
    checks.append(Check("dataset_catalog", True, f"{len(refs)} refs, {len(benches)} benchmarks", {"refs": refs, "benches": benches}))

    for ref_name in refs:
        def _load_ref(n=ref_name):
            r = load_reference_record(n)
            v = validate_record(r)
            return f"{r.record_id} components={len(r.components)} valid={v.is_valid} level={v.level}"

        checks.append(run(f"reference:{ref_name}", _load_ref))

    for bench_name in benches:
        def _bench(n=bench_name):
            b = load_benchmark(n)
            keys = list(b.keys())[:8]
            extra = ""
            if "n_samples" in b:
                extra = f" n_samples={b['n_samples']}"
            if "samples" in b and isinstance(b["samples"], list):
                extra += f" len(samples)={len(b['samples'])}"
            return f"keys={keys}{extra}"

        checks.append(run(f"benchmark:{bench_name}", _bench))

    # Cross-reference matching
    def match_refs():
        a = load_reference_record("earthquake_psd_reference")
        b = load_reference_record("structural_fas_reference")
        m = match_records(a, b)
        return f"score={m.composite_score:.4f} metrics={m.metric_breakdown}"

    checks.append(run("match:earthquake_vs_structural_fas", match_refs))

    def match_self():
        r = load_reference_record("rotdnn_reference")
        m = match_records(r, r)
        return f"self_score={m.composite_score:.4f}"

    checks.append(run("match:rotdnn_self", match_self))

    # Ranking
    def rank_demo():
        ref = load_reference_record("vibration_monitoring_reference")
        cands = [load_reference_record("rotdnn_reference"), load_reference_record("structural_fas_reference")]
        ranked = rank_candidates(ref, cands)
        return [(x.candidate_id, round(x.composite_score, 4)) for x in ranked]

    checks.append(run("rank_candidates", rank_demo))

    # Generation
    import numpy as np

    cfg = GenerationConfig(seed=42, target_frequency=np.linspace(0.1, 50.0, 64), amplitude_shape="power_law")
    for gen_name, gen_fn in [
        ("generate_psd", lambda: generate_psd(cfg)),
        ("generate_fas", lambda: generate_fas(cfg)),
        ("generate_rotdnn", lambda: generate_rotdnn(cfg)),
    ]:
        checks.append(run(gen_name, lambda f=gen_fn: f"{f().record_id} rep={f().representation} comps={len(f().components)}"))

    # Embeddings + retrieval
    def embedding_pipeline():
        records = [load_reference_record(n) for n in refs[:3]]
        vec = SpectralVectorizer(n_bands=8)
        matrix = vec.batch_transform(records)
        retriever = SpectralRetriever(vec)
        retriever.index(records)
        hits = retriever.query(records[0], top_k=2)
        return {"shape": list(matrix.shape), "top_hit": hits[0][0] if hits else None}

    checks.append(run("embeddings+retrieval", embedding_pipeline))

    # Cognitive
    def cognitive_loop():
        adapter = SpectralMemoryAdapter()
        anomaly = SpectralAnomalyAdapter(threshold=3.0)
        attention = SpectralAttentionAdapter()
        state = AgentStateSpectralAdapter()
        baseline = [load_reference_record("vibration_monitoring_reference")]
        anomaly.fit_baseline(baseline)
        rec = load_reference_record("rotdnn_reference")
        mem = adapter.to_memory_object(rec)
        st = state.to_state_vector(rec)
        w = attention.compute_attention_weights(baseline, query=rec)
        return {
            "embedding_len": len(mem["spectral_embedding"]),
            "state_dim": len(st),
            "anomaly": anomaly.score_anomaly(rec),
            "attention_sum": float(w.sum()),
        }

    checks.append(run("cognitive_adapters", cognitive_loop))

    # Intelligence protocols (numpy path)
    def intel_protocol():
        from mesie.ai.intelligence_protocols import IntelligenceConfig, IntelligenceProtocol, IntelligenceLevel

        rec = load_reference_record("earthquake_psd_reference")
        amp = rec.components[0].amplitude
        proto = IntelligenceProtocol(IntelligenceConfig(level=IntelligenceLevel.ADAPTIVE))
        proto.observe(amp)
        out = proto.reason(amp)
        return f"conclusion={out.conclusion[:60]!r} confidence={out.confidence:.3f}"

    checks.append(run("intelligence_protocol", intel_protocol))

    # Protocols / helix smoke
    def protocol_serialize():
        from mesie.protocols.serialization import SerializationFormat, SpectralSerializer

        rec = load_reference_record("structural_fas_reference")
        c = rec.components[0]
        ser = SpectralSerializer(default_format=SerializationFormat.JSON)
        blob = ser.serialize(c.frequency, c.amplitude, metadata={"record_id": rec.record_id})
        return f"format={blob.format.value} size_bytes={blob.size_bytes}"

    checks.append(run("protocol_serialization", protocol_serialize))

    def helix_encode():
        from mesie.helix.vector_helix import VectorHelix

        helix = VectorHelix()
        nodes = helix.insert_batch([load_reference_record(n) for n in refs[:3]])
        return f"nodes={len(nodes)} phase={nodes[0].phase:.3f}"

    checks.append(run("helix_insert_batch", helix_encode))

    # Report JSON for parsing
    report = {
        "version": mesie.__version__,
        "passed": sum(1 for c in checks if c.ok),
        "failed": sum(1 for c in checks if not c.ok),
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail, "metrics": c.metrics}
            for c in checks
        ],
    }
    out_path = ROOT / "scripts" / "sdk_test_drive_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()