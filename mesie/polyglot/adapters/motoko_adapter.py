"""Motoko adapter — canister contract mirror for on-chain / IC validation."""

from __future__ import annotations

from typing import List, Optional, Tuple

from mesie.polyglot.adapters.base import PolyglotAdapter
from mesie.polyglot.adapters.rust_adapter import _fallback_match, _fallback_validate
from mesie.polyglot.contract import AISVectorMessage, PolyglotAction, RuntimeId


class MotokoAdapter(PolyglotAdapter):
    """Motoko logic mirrored locally; native mode when dfx canister endpoint is configured."""

    runtime = RuntimeId.MOTOKO

    def __init__(self, canister_url: str = "") -> None:
        self.canister_url = canister_url

    def available(self) -> bool:
        return bool(self.canister_url)

    def _motoko_validate(self, record: dict) -> dict:
        """Mirror of bindings/motoko/mesie_canister validation rules."""
        comps = record.get("components") or []
        if not comps and not record.get("frequency"):
            return {"is_valid": False, "level": 1, "errors": ["no components"], "warnings": []}
        c = comps[0] if comps else record
        freq = c.get("frequency", record.get("frequency", []))
        amp = c.get("amplitude", record.get("amplitude", []))
        if len(freq) != len(amp):
            return {"is_valid": False, "level": 2, "errors": ["length mismatch"], "warnings": []}
        if any(x < 0 for x in amp):
            return {"is_valid": False, "level": 3, "errors": ["negative amplitude"], "warnings": []}
        return {"is_valid": True, "level": 5, "errors": [], "warnings": [], "chain_ready": True}

    def _handle(self, message: AISVectorMessage) -> Tuple[dict, Optional[List[float]]]:
        if message.action == PolyglotAction.HEALTH:
            return {"status": "ok", "mode": self.mode, "canister": self.canister_url or "local-mirror"}, None

        if message.action == PolyglotAction.VALIDATE:
            data = self._motoko_validate(message.record or {})
            data["runtime_note"] = "motoko-canister-mirror"
            return data, None

        if message.action == PolyglotAction.MATCH:
            data = _fallback_match(message.record_a or message.record or {}, message.record_b or {})
            data["chain_attested"] = False
            return data, None

        raise ValueError(f"motoko unsupported action: {message.action}")