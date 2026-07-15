"""TypeScript adapter — mirrors workers/mesie-api spectral.ts logic."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mesie.polyglot.adapters.base import PolyglotAdapter
from mesie.polyglot.contract import AISVectorMessage, PolyglotAction, RuntimeId


def _primary(record: dict) -> tuple[np.ndarray, np.ndarray, str]:
    comps = record.get("components") or []
    if comps:
        c = comps[0]
        return (
            np.asarray(c.get("frequency", []), dtype=np.float64),
            np.asarray(c.get("amplitude", []), dtype=np.float64),
            c.get("name", "component"),
        )
    return (
        np.asarray(record.get("frequency", []), dtype=np.float64),
        np.asarray(record.get("amplitude", []), dtype=np.float64),
        "component_0",
    )


class TypeScriptAdapter(PolyglotAdapter):
    """In-process mirror of workers/mesie-api/src/spectral.ts for parity without HTTP."""

    runtime = RuntimeId.TYPESCRIPT

    def available(self) -> bool:
        return True

    def _validate(self, record: dict) -> dict:
        errors, warnings = [], []
        level = 1
        if not record:
            return {"is_valid": False, "level": 0, "errors": ["empty"], "warnings": []}
        freq, amp, name = _primary(record)
        if len(freq) == 0 or len(amp) == 0:
            return {"is_valid": False, "level": 2, "errors": [f"{name} missing arrays"], "warnings": []}
        if len(freq) != len(amp):
            return {"is_valid": False, "level": 2, "errors": ["length mismatch"], "warnings": []}
        level = 3
        for i in range(len(freq)):
            if not np.isfinite(freq[i]) or not np.isfinite(amp[i]):
                errors.append("non-finite values")
                break
        for i in range(1, len(freq)):
            if freq[i] < freq[i - 1]:
                warnings.append("non-monotonic frequency grid")
                break
        if any(a < 0 for a in amp):
            errors.append("negative amplitudes")
        level = 6 if not errors else 4
        return {"is_valid": not errors, "level": level, "errors": errors, "warnings": warnings}

    def _match(self, a: dict, b: dict) -> dict:
        fa, aa, _ = _primary(a)
        fb, ab, _ = _primary(b)
        n = min(len(fa), len(fb), len(aa), len(ab))
        va, vb = aa[:n], ab[:n]
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        cosine = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
        rmse = float(np.sqrt(np.mean((va - vb) ** 2))) if n else 1.0
        score = max(0.0, min(1.0, 0.7 * cosine + 0.3 * (1.0 / (1.0 + rmse))))
        return {
            "composite_score": score,
            "metrics": {"cosine": cosine, "rmse": rmse},
            "reference_id": a.get("record_id", "ref"),
            "candidate_id": b.get("record_id", "cand"),
        }

    def _handle(self, message: AISVectorMessage) -> Tuple[dict, Optional[List[float]]]:
        if message.action == PolyglotAction.HEALTH:
            return {"status": "ok", "engine": "typescript-mirror"}, None
        if message.action == PolyglotAction.VALIDATE:
            return self._validate(message.record or {}), None
        if message.action == PolyglotAction.MATCH:
            return self._match(message.record_a or message.record or {}, message.record_b or {}), None
        raise ValueError(f"typescript unsupported action: {message.action}")