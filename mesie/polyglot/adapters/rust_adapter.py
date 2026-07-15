"""Rust adapter — native CLI or NumPy fallback."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mesie.polyglot.adapters.base import PolyglotAdapter
from mesie.polyglot.contract import AISVectorMessage, PolyglotAction, RuntimeId

ROOT = Path(__file__).resolve().parents[3]
RUST_BIN = ROOT / "bindings" / "rust" / "mesie-spectral-core" / "target" / "release" / "mesie-spectral-core.exe"
RUST_BIN_UNIX = ROOT / "bindings" / "rust" / "mesie-spectral-core" / "target" / "release" / "mesie-spectral-core"


def _primary_arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    comps = record.get("components") or []
    if comps:
        c = comps[0]
        return np.asarray(c.get("frequency", []), dtype=np.float64), np.asarray(c.get("amplitude", []), dtype=np.float64)
    return np.asarray(record.get("frequency", []), dtype=np.float64), np.asarray(record.get("amplitude", []), dtype=np.float64)


def _fallback_validate(record: dict) -> dict:
    freq, amp = _primary_arrays(record)
    errors = []
    if len(freq) == 0 or len(amp) == 0:
        errors.append("missing frequency/amplitude")
    if len(freq) != len(amp):
        errors.append("length mismatch")
    if any(a < 0 for a in amp):
        errors.append("negative amplitudes")
    return {"is_valid": not errors, "level": 4 if not errors else 2, "errors": errors, "warnings": []}


def _fallback_match(a: dict, b: dict) -> dict:
    fa, aa = _primary_arrays(a)
    fb, ab = _primary_arrays(b)
    n = min(len(fa), len(fb), len(aa), len(ab))
    if n == 0:
        return {"composite_score": 0.0, "metrics": {"cosine": 0.0, "rmse": 1.0}}
    va, vb = aa[:n], ab[:n]
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    cosine = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean((va - vb) ** 2)))
    score = max(0.0, min(1.0, 0.6 * cosine + 0.4 * (1.0 / (1.0 + rmse))))
    return {"composite_score": score, "metrics": {"cosine": cosine, "rmse": rmse}}


class RustAdapter(PolyglotAdapter):
    runtime = RuntimeId.RUST

    def _bin_path(self) -> Optional[Path]:
        if RUST_BIN.exists():
            return RUST_BIN
        if RUST_BIN_UNIX.exists():
            return RUST_BIN_UNIX
        return None

    def available(self) -> bool:
        return self._bin_path() is not None

    def _native(self, payload: dict) -> dict:
        proc = subprocess.run(
            [str(self._bin_path())],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return json.loads(proc.stdout)

    def _handle(self, message: AISVectorMessage) -> Tuple[dict, Optional[List[float]]]:
        if message.action == PolyglotAction.HEALTH:
            return {"status": "ok", "mode": self.mode}, None

        payload = {
            "action": message.action.value,
            "record": message.record,
            "record_a": message.record_a,
            "record_b": message.record_b,
        }

        if self.available():
            out = self._native(payload)
            return out.get("data", out), out.get("vector")

        if message.action == PolyglotAction.VALIDATE:
            return _fallback_validate(message.record or {}), None
        if message.action == PolyglotAction.MATCH:
            return _fallback_match(message.record_a or message.record or {}, message.record_b or {}), None
        raise ValueError(f"rust fallback unsupported action: {message.action}")