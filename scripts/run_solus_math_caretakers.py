"""Run SOLUS math caretakers — Logic Prover + Pattern Forge inside SDK organism."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie.sdk import MAESIClient, SDKSolusOrganism, SOLUS_BRAND


def main() -> None:
    print(f"=== {SOLUS_BRAND} Math Caretakers — Logic Prover + Pattern Forge ===\n")
    organism = SDKSolusOrganism()
    vitals = organism.pulse()
    print(f"Organism: {vitals.organism} | health={vitals.sdk_health} | sovereign={vitals.sovereign}")
    print(f"Caretakers: {', '.join(vitals.caretakers)}\n")

    prove = organism.logic_action("prove", theorem="forall x in spectra: embed(x) enables match(x, corpus)")
    print(f"Logic Prover: ok={prove.ok} | brain={prove.brain['conclusion'][:60]}...")
    print(f"  heart bpm={prove.heart['bpm']} coherence={prove.heart['coherence']}\n")

    import numpy as np
    demo = np.exp(-((np.linspace(0.5, 20, 32) - 5) ** 2) / 4)
    xray = organism.pattern_action("xray", values=demo.tolist())
    print(f"Pattern Forge: ok={xray.ok} | signals={xray.data.get('signal_count')}/{xray.data.get('n')}")
    print(f"  brain={xray.brain['conclusion'][:60]}...\n")

    refs = [load_reference_record(n) for n in list_references()[:2]]
    if refs:
        c = refs[0].components[0]
        analysis = organism.analyze_spectrum(c.frequency.tolist(), c.amplitude.tolist())
        print(f"Spectral bridge: xray ratio={analysis['xray'].get('signal_ratio')}")

    client = MAESIClient(fast=True, use_solus_caretakers=True)
    report = client.run_full(refs, benchmark=True)
    print(f"\nMAESI SDK + organism: {report.plain_summary[:200]}...")
    if report.solus_organism:
        print(f"Organism tend: {report.solus_organism['sdk_health']}")

    out = ROOT / "deliverables" / "SOLUS_Math_Caretakers_Report.json"
    payload = {
        "brand": SOLUS_BRAND,
        "vitals": vitals.__dict__,
        "logic_prover": {"ok": prove.ok, "data": prove.data, "heart": prove.heart, "brain": prove.brain},
        "pattern_forge": {"ok": xray.ok, "data": xray.data, "heart": xray.heart, "brain": xray.brain},
        "maesi_organism": report.solus_organism,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()