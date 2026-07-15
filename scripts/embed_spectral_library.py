"""Build an embeddable spectral library index from bundled + generated records."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_benchmarks, list_references, load_benchmark, load_reference_record
from mesie import GenerationConfig, generate_fas, generate_psd, load_record, match_records, validate_record
from mesie.cognitive import SpectralAnomalyAdapter, SpectralMemoryAdapter
from mesie.embeddings import SpectralRetriever, SpectralVectorizer
from mesie.ai.intelligence_protocols import IntelligenceConfig, IntelligenceLevel, IntelligenceProtocol


LIBRARY_DIR = ROOT / "library"
INDEX_PATH = LIBRARY_DIR / "spectral_index.json"


@dataclass
class LibraryEntry:
    id: str
    source: str
    category: str
    embedding: List[float]
    valid: bool
    validation_level: int
    tags: List[str]


def _record_from_benchmark_sample(sample: dict, dataset_id: str) -> dict:
    return {
        "record_id": sample.get("sample_id", "unknown"),
        "representation": "single",
        "components": [
            {
                "name": "channel",
                "frequency": sample["frequencies"],
                "amplitude": sample["amplitudes"],
            }
        ],
        "metadata": {"library": dataset_id, "parameters": sample.get("parameters", {})},
    }


def build_library_entries() -> List[LibraryEntry]:
    vectorizer = SpectralVectorizer(n_bands=8)
    entries: List[LibraryEntry] = []

    for name in list_references():
        rec = load_reference_record(name)
        report = validate_record(rec)
        emb = vectorizer.transform(rec)
        entries.append(
            LibraryEntry(
                id=rec.record_id,
                source=f"reference:{name}",
                category=_category_from_name(name),
                embedding=emb.tolist(),
                valid=report.is_valid,
                validation_level=report.level,
                tags=["reference", report.is_valid and "production" or "needs_review"],
            )
        )

    for bench_name in list_benchmarks():
        data = load_benchmark(bench_name)
        for sample in data.get("samples", []):
            raw = _record_from_benchmark_sample(sample, data.get("dataset_id", bench_name))
            rec = load_record(raw)
            report = validate_record(rec)
            emb = vectorizer.transform(rec)
            cls = sample.get("parameters", {}).get("class", bench_name)
            entries.append(
                LibraryEntry(
                    id=rec.record_id,
                    source=f"benchmark:{bench_name}",
                    category=str(cls) if cls else "training",
                    embedding=emb.tolist(),
                    valid=report.is_valid,
                    validation_level=report.level,
                    tags=["benchmark", bench_name],
                )
            )

    cfg = GenerationConfig(seed=7, target_frequency=np.linspace(0.2, 40, 64))
    for label, gen in [("synthetic_psd", generate_psd), ("synthetic_fas", generate_fas)]:
        rec = gen(cfg)
        report = validate_record(rec)
        emb = vectorizer.transform(rec)
        entries.append(
            LibraryEntry(
                id=f"{label}_{rec.record_id}",
                source="generated",
                category=label,
                embedding=emb.tolist(),
                valid=report.is_valid,
                validation_level=report.level,
                tags=["generated", "seed_7"],
            )
        )

    return entries


def _category_from_name(name: str) -> str:
    if "earthquake" in name:
        return "seismic"
    if "vibration" in name:
        return "machinery"
    if "structural" in name:
        return "structural"
    if "rotdnn" in name:
        return "seismic_design"
    return "general"


def run_use_case_scenarios(entries: List[LibraryEntry], vectorizer: SpectralVectorizer) -> Dict[str, Any]:
    records_by_id = {}
    for e in entries:
        if e.source.startswith("reference:"):
            name = e.source.split(":", 1)[1]
            records_by_id[e.id] = load_reference_record(name)

    retriever = SpectralRetriever(vectorizer)
    ref_records = list(records_by_id.values())
    retriever.index(ref_records)

    earthquake = load_reference_record("earthquake_psd_reference")
    vibration = load_reference_record("vibration_monitoring_reference")

    eq_match = match_records(earthquake, vibration)
    mem = SpectralMemoryAdapter()
    eq_memory = mem.to_memory_object(earthquake)

    anomaly = SpectralAnomalyAdapter(threshold=2.5)
    anomaly.fit_baseline(ref_records[:2])
    vib_anomaly = anomaly.score_anomaly(vibration)

    amp = earthquake.components[0].amplitude
    intel = IntelligenceProtocol(IntelligenceConfig(level=IntelligenceLevel.ADAPTIVE))
    intel.observe(amp)
    reasoning = intel.reason(amp)

    neighbors = retriever.query(earthquake, top_k=3)

    bench = load_benchmark("spectral_classification_benchmark")
    class_counts: Dict[str, int] = {}
    for s in bench.get("samples", [])[:50]:
        c = s.get("parameters", {}).get("class", "unknown")
        class_counts[c] = class_counts.get(c, 0) + 1

    by_category: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for e in entries:
        by_category[e.category] = by_category.get(e.category, 0) + 1
        src = e.source.split(":")[0]
        by_source[src] = by_source.get(src, 0) + 1

    return {
        "library_size": len(entries),
        "embedding_dimensions": len(entries[0].embedding) if entries else 0,
        "entries_by_category": by_category,
        "entries_by_source": by_source,
        "retrieval_neighbors_for_earthquake": neighbors,
        "cross_domain_match": {
            "reference_a": "earthquake_psd",
            "reference_b": "vibration_monitoring",
            "similarity_score": round(eq_match.composite_score, 4),
            "plain_read": _plain_match_read(eq_match.composite_score),
        },
        "memory_object_keys": list(eq_memory.keys()),
        "anomaly_on_vibration_vs_seismic_baseline": round(vib_anomaly, 3),
        "intelligence_conclusion": reasoning.conclusion,
        "intelligence_confidence": round(reasoning.confidence, 3),
        "benchmark_class_mix_sample": class_counts,
    }


def _plain_match_read(score: float) -> str:
    if score >= 0.85:
        return "Very similar — likely same type of event or machine state."
    if score >= 0.65:
        return "Related — some shared patterns, not a duplicate."
    if score >= 0.45:
        return "Weak link — different situations with occasional overlap."
    return "Different — treat as separate cases."


def main() -> None:
    LIBRARY_DIR.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    entries = build_library_entries()
    build_ms = (time.perf_counter() - t0) * 1000

    vectorizer = SpectralVectorizer(n_bands=8)
    scenarios = run_use_case_scenarios(entries, vectorizer)

    t1 = time.perf_counter()
    n_match = 2000
    a = load_reference_record("earthquake_psd_reference")
    b = load_reference_record("structural_fas_reference")
    for _ in range(n_match):
        match_records(a, b)
    match_per_ms = (time.perf_counter() - t1) / n_match * 1000

    embeds_per_sec = round(len(entries) / (build_ms / 1000), 0) if build_ms > 0 else 0

    index = {
        "version": "0.2.1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry_count": len(entries),
        "embedding_dim": scenarios["embedding_dimensions"],
        "build_time_ms": round(build_ms, 1),
        "embeds_per_second": embeds_per_sec,
        "avg_match_time_ms": round(match_per_ms, 4),
        "entries": [asdict(e) for e in entries],
        "scenarios": scenarios,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({k: index[k] for k in index if k != "entries"}, indent=2))
    print(f"\nWrote {INDEX_PATH} ({len(entries)} embedded entries)")


if __name__ == "__main__":
    main()