"""Repair bundled reference JSON: non-negative PSD amplitudes, units, representation."""

from __future__ import annotations

import json
from pathlib import Path

DATA_REF = Path(__file__).resolve().parents[1] / "data" / "reference"
FLOOR = 1e-12


def fix_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    record_type = str(data.get("record_type", "")).lower()
    rep = {"psd": "psd", "fas": "fas"}.get(record_type, data.get("representation", "single"))
    if record_type.startswith("rotd"):
        rep = "rotdnn"
    data["representation"] = rep

    meta = data.setdefault("metadata", {})
    units = meta.get("units", {})
    amp_unit = units.get("amplitude", "linear") if isinstance(units, dict) else "linear"

    clipped = 0
    for comp in data.get("components", []):
        key = "amplitudes" if "amplitudes" in comp else "amplitude"
        amps = list(comp.get(key, []))
        for i, v in enumerate(amps):
            if v < 0:
                amps[i] = FLOOR
                clipped += 1
        comp[key] = amps
        comp["units"] = amp_unit

    meta["data_fix"] = {
        "version": "0.2.1",
        "negative_amplitudes_clipped": clipped,
        "note": "Tiny negative PSD values clipped to physical floor for MESIE validation.",
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"file": path.name, "clipped": clipped, "representation": rep}


if __name__ == "__main__":
    for p in sorted(DATA_REF.glob("*.json")):
        print(fix_file(p))