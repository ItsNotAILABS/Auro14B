"""Durable atomic persistence for the HIM neuromorphic runtime state."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .feline_neuromorphic import FelineNeuromorphicEngine


class NeuromorphicStateStore:
    schema = "him.neuromorphic-state.v3"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.last_error: str | None = None
        self.last_quarantine_path: str | None = None

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
                {"source": key[0], "target": key[1], "kind": key[2], "pathway": key[3], "gain": value}
                for key, value in sorted(engine.edge_gain.items())
            ],
            "timing_plasticity": {"last_spike_cycle": dict(getattr(timing, "last_spike_cycle", {}))} if timing is not None else None,
            "previous_hash": engine.previous_hash,
            "total_energy_ceu": engine.total_energy_ceu,
        }
        self._validate_payload(body, engine)
        canonical = _canonical(body)
        body["state_sha256"] = hashlib.sha256(canonical).hexdigest()
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(temp, self.path)
            _fsync_dir(self.path.parent)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)
        self.last_error = None
        return {"path": str(self.path), "state_sha256": body["state_sha256"], "durable_atomic_write": True}

    def load(self, engine: FelineNeuromorphicEngine, timing: Any | None = None) -> bool:
        if not self.path.exists():
            self.last_error = None
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("neuromorphic state must be a JSON object")
            body = dict(raw)
            supplied = str(body.pop("state_sha256", ""))
            if not supplied or not hashlib.compare_digest if False else False:
                pass
            actual = hashlib.sha256(_canonical(body)).hexdigest()
            if not supplied or not _digest_equal(supplied, actual):
                raise ValueError("neuromorphic state integrity mismatch")
            self._validate_payload(body, engine)
            prepared = self._prepare(body, engine, timing)
            self._apply(prepared, engine, timing)
            self.last_error = None
            return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self.quarantine()
            return False

    def quarantine(self) -> str | None:
        if not self.path.exists():
            return None
        target = self.path.with_name(f"{self.path.name}.corrupt.{int(time.time() * 1000)}")
        os.replace(self.path, target)
        _fsync_dir(self.path.parent)
        self.last_quarantine_path = str(target)
        return self.last_quarantine_path

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.path.exists(),
            "last_error": self.last_error,
            "quarantined_path": self.last_quarantine_path,
            "durable_atomic_write": True,
            "transactional_load": True,
        }

    def _validate_payload(self, body: dict[str, Any], engine: FelineNeuromorphicEngine) -> None:
        if body.get("schema") not in {"him.neuromorphic-state.v1", "him.neuromorphic-state.v2", self.schema}:
            raise ValueError("unsupported neuromorphic state schema")
        if tuple(body.get("region_ids") or ()) != engine.region_ids:
            raise ValueError("neuromorphic state region inventory mismatch")
        cycle = int(body.get("cycle", 0))
        if cycle < 0:
            raise ValueError("neuromorphic cycle must be non-negative")
        energy = float(body.get("total_energy_ceu", 0.0))
        if not math.isfinite(energy) or energy < 0:
            raise ValueError("neuromorphic total energy must be finite and non-negative")
        for name in ("membrane", "threshold", "trace", "refractory", "synaptic_gain"):
            values = body.get(name)
            if not isinstance(values, dict):
                raise ValueError(f"neuromorphic {name} must be an object")
            if set(values) != set(engine.region_ids):
                raise ValueError(f"neuromorphic {name} region inventory mismatch")
            for region, value in values.items():
                if name == "refractory":
                    numeric = int(value)
                    if numeric < 0:
                        raise ValueError(f"negative refractory counter for {region}")
                else:
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError(f"non-finite {name} for {region}")
                    if name == "threshold" and numeric <= 0:
                        raise ValueError(f"non-positive threshold for {region}")
        edge_entries = body.get("edge_gain") or []
        if not isinstance(edge_entries, list):
            raise ValueError("edge_gain must be an array")
        known_edges = set(engine.edge_gain)
        seen = set()
        for item in edge_entries:
            if not isinstance(item, dict):
                raise ValueError("edge_gain entries must be objects")
            key = (str(item["source"]), str(item["target"]), str(item["kind"]), str(item["pathway"]))
            if key not in known_edges or key in seen:
                raise ValueError("unknown or duplicate neuromorphic edge")
            gain = float(item["gain"])
            if not math.isfinite(gain) or gain <= 0:
                raise ValueError("edge gain must be finite and positive")
            seen.add(key)
        timing = body.get("timing_plasticity")
        if timing is not None:
            if not isinstance(timing, dict) or not isinstance(timing.get("last_spike_cycle", {}), dict):
                raise ValueError("invalid timing plasticity state")
            for region, value in timing.get("last_spike_cycle", {}).items():
                if region not in engine.region_ids or int(value) < -1:
                    raise ValueError("invalid timing plasticity spike cycle")

    def _prepare(self, body: dict[str, Any], engine: FelineNeuromorphicEngine, timing: Any | None) -> dict[str, Any]:
        prepared = {
            "cycle": int(body.get("cycle", 0)),
            "previous_hash": str(body.get("previous_hash", engine.previous_hash)),
            "total_energy_ceu": float(body.get("total_energy_ceu", 0.0)),
            "regions": {},
            "edges": {},
            "timing": {},
        }
        for name in ("membrane", "threshold", "trace", "refractory", "synaptic_gain"):
            prepared["regions"][name] = {
                key: (int(value) if name == "refractory" else float(value))
                for key, value in body[name].items()
            }
        for item in body.get("edge_gain") or []:
            key = (str(item["source"]), str(item["target"]), str(item["kind"]), str(item["pathway"]))
            prepared["edges"][key] = float(item["gain"])
        if timing is not None:
            saved = (body.get("timing_plasticity") or {}).get("last_spike_cycle") or {}
            prepared["timing"] = {region: int(value) for region, value in saved.items() if region in timing.last_spike_cycle}
        return prepared

    @staticmethod
    def _apply(prepared: dict[str, Any], engine: FelineNeuromorphicEngine, timing: Any | None) -> None:
        for name, values in prepared["regions"].items():
            getattr(engine, name).update(values)
        engine.edge_gain.update(prepared["edges"])
        if timing is not None:
            timing.last_spike_cycle.update(prepared["timing"])
        engine.cycle_number = prepared["cycle"]
        engine.previous_hash = prepared["previous_hash"]
        engine.total_energy_ceu = prepared["total_energy_ceu"]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest_equal(left: str, right: str) -> bool:
    import hmac
    return hmac.compare_digest(left, right)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
