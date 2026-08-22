"""Atomic persistence for the HIM neuromorphic runtime state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .feline_neuromorphic import FelineNeuromorphicEngine


class NeuromorphicStateStore:
    schema = "him.neuromorphic-state.v2"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def for_brain_state(cls, brain_state_path: str | Path) -> "NeuromorphicStateStore":
        base = Path(brain_state_path)
        return cls(base.with_suffix(base.suffix + ".neuromorphic.json"))

    def save(self, engine: FelineNeuromorphicEngine, timing: Any | None = None) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "engine_schema": engine.schema,
            "cycle": engine.cycle_number,
            "region_ids": list(engine.region_ids),
            "membrane": engine.membrane,
            "threshold": engine.threshold,
            "trace": engine.trace,
            "refractory": engine.refractory,
            "synaptic_gain": engine.synaptic_gain,
            "edge_gain": [
                {
                    "source": key[0],
                    "target": key[1],
                    "kind": key[2],
                    "pathway": key[3],
                    "gain": value,
                }
                for key, value in sorted(engine.edge_gain.items())
            ],
            "timing_plasticity": {
                "last_spike_cycle": dict(getattr(timing, "last_spike_cycle", {})),
            } if timing is not None else None,
            "previous_hash": engine.previous_hash,
            "total_energy_ceu": engine.total_energy_ceu,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        body["state_sha256"] = hashlib.sha256(canonical).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)
        return {"path": str(self.path), "state_sha256": body["state_sha256"]}

    def load(self, engine: FelineNeuromorphicEngine, timing: Any | None = None) -> bool:
        if not self.path.exists():
            return False
        body = json.loads(self.path.read_text(encoding="utf-8"))
        supplied = str(body.pop("state_sha256", ""))
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if not supplied or hashlib.sha256(canonical).hexdigest() != supplied:
            raise ValueError("neuromorphic state integrity mismatch")
        if body.get("schema") not in {"him.neuromorphic-state.v1", self.schema}:
            raise ValueError("unsupported neuromorphic state schema")
        if tuple(body.get("region_ids") or ()) != engine.region_ids:
            raise ValueError("neuromorphic state region inventory mismatch")

        for name in ("membrane", "threshold", "trace", "refractory", "synaptic_gain"):
            values = body.get(name) or {}
            target = getattr(engine, name)
            for key, value in values.items():
                if key in target:
                    target[key] = int(value) if name == "refractory" else float(value)

        known_edges = set(engine.edge_gain)
        for item in body.get("edge_gain") or []:
            key = (str(item["source"]), str(item["target"]), str(item["kind"]), str(item["pathway"]))
            if key in known_edges:
                engine.edge_gain[key] = float(item["gain"])

        if timing is not None:
            saved_timing = body.get("timing_plasticity") or {}
            saved_spikes = saved_timing.get("last_spike_cycle") or {}
            for region, value in saved_spikes.items():
                if region in timing.last_spike_cycle:
                    timing.last_spike_cycle[region] = int(value)

        engine.cycle_number = int(body.get("cycle", 0))
        engine.previous_hash = str(body.get("previous_hash", engine.previous_hash))
        engine.total_energy_ceu = float(body.get("total_energy_ceu", 0.0))
        return True
