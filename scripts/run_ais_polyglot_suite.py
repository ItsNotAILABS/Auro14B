"""Run AISVectorPolyglot full test + use + integration suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie.polyglot import AISVectorPolyglotSuite, SUITE_NAME, run_integration_suite


def main() -> None:
    refs = [load_reference_record(n) for n in list_references()]
    print(f"=== {SUITE_NAME} — Test / Use / Integration Suite ===\n")

    suite = AISVectorPolyglotSuite()
    health = suite.health()
    print("Runtimes:")
    for name, info in health.runtimes.items():
        print(f"  {name:12} mode={info['mode']:8} available={info['available']}")

    report = run_integration_suite(refs)
    print(f"\nIntegration: {report.passed}/{report.total} passed ({report.runtime_ms:.0f} ms)")

    parity = suite.parity_matrix(refs[0], refs[1] if len(refs) > 1 else refs[0])
    print(f"Parity spread: {parity.get('score_spread')}")

    n = suite.index_corpus(refs)
    q = suite.vector_query(refs[0])
    print(f"Vector index: {n} records | query neighbors: {len(q.neighbors)} | fp hits: {len(q.fingerprint_hits)}")

    tools = suite.third_party.openai_tools_schema()
    print(f"Third-party AI tools: {len(tools)}")

    out = ROOT / "deliverables" / "AISVectorPolyglot_Integration_Report.json"
    payload = {
        "suite": SUITE_NAME,
        "health": {"runtimes": health.runtimes, "vector_indexed": health.vector_indexed},
        "integration": {
            "passed": report.passed,
            "failed": report.failed,
            "total": report.total,
            "runtime_ms": report.runtime_ms,
            "outcomes": [
                {
                    "id": o.id,
                    "name": o.name,
                    "category": o.category,
                    "passed": o.passed,
                    "latency_ms": o.latency_ms,
                    "error": o.error,
                }
                for o in report.outcomes
            ],
        },
        "parity": parity,
        "vector_query_ms": q.elapsed_ms,
        "third_party_tools": [t["function"]["name"] for t in tools],
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    if report.failed:
        print("\n--- FAILURES ---")
        for o in report.outcomes:
            if not o.passed:
                print(f"  {o.id} {o.name}: {o.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()