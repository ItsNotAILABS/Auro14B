"""Julia adapter — subprocess bridge or NumPy fallback."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mesie.polyglot.adapters.base import PolyglotAdapter
from mesie.polyglot.adapters.rust_adapter import _fallback_match, _fallback_validate
from mesie.polyglot.contract import AISVectorMessage, PolyglotAction, RuntimeId

ROOT = Path(__file__).resolve().parents[3]
JULIA_SCRIPT = ROOT / "bindings" / "julia" / "MESIEPolyglot" / "cli.jl"


class JuliaAdapter(PolyglotAdapter):
    runtime = RuntimeId.JULIA

    def available(self) -> bool:
        if shutil.which("julia") is None or not JULIA_SCRIPT.exists():
            return False
        try:
            proc = subprocess.run(
                ["julia", str(JULIA_SCRIPT)],
                input=json.dumps({"action": "health"}),
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            json.loads(proc.stdout)
            return True
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            return False

    def _native(self, payload: dict) -> dict:
        proc = subprocess.run(
            ["julia", str(JULIA_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=60,
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
            try:
                out = self._native(payload)
                return out.get("data", out), out.get("vector")
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass

        if message.action == PolyglotAction.VALIDATE:
            return _fallback_validate(message.record or {}), None
        if message.action == PolyglotAction.MATCH:
            return _fallback_match(message.record_a or message.record or {}, message.record_b or {}), None
        if message.action == PolyglotAction.EMBED:
            from mesie.polyglot.adapters.python_adapter import PythonAdapter
            return PythonAdapter()._handle(message)
        raise ValueError(f"julia fallback unsupported action: {message.action}")