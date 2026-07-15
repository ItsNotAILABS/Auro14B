"""Run MAESI SDK v1.1 — knowledge libraries + fast compute benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie.sdk import MAESIClient


def main() -> None:
    refs = [load_reference_record(n) for n in list_references()]
    client = MAESIClient(fast=True, use_fingerprint=True)
    report = client.run_full(refs, benchmark=True)

    out = ROOT / "deliverables" / "MAESI_SDK_Run_Report.json"
    payload = {
        "version": report.version,
        "knowledge": report.knowledge.__dict__,
        "speed": report.speed.__dict__ if report.speed else None,
        "fingerprint_hits": report.fingerprint_hits,
        "neuroaix_available": report.neuroaix_available,
        "plain_summary": report.plain_summary,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== MAESI SDK Run ===\n")
    k = report.knowledge
    print(f"SDK {report.version}")
    print(f"Knowledge: {k.physical_laws} laws, {k.chemical_elements} elements, {k.biological_systems} bio,")
    print(f"           {k.technical_concepts} technical, {k.research_entries} research")
    if report.speed:
        s = report.speed
        print(f"\nSpeed: loop {s.loop_match_ms} ms/match | batch {s.batch_match_ms} ms | speedup {s.speedup_ratio}x")
        print(f"       embed_batch({s.n_items}) {s.embed_batch_ms} ms | ANN query {s.ann_query_ms} ms")
    print(f"\nNeuroAIX encoder: {'ok' if report.neuroaix_available else 'skip'}")
    print(f"\n{report.plain_summary}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()