"""Signed promotion evidence for AURO checkpoints."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Dict, Mapping


class PromotionEvidenceError(RuntimeError):
    pass


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_signed_promotion_evidence(path: str | Path, signing_key: str) -> Dict[str, Any]:
    if not signing_key:
        raise PromotionEvidenceError("AURO checkpoint signing key is required")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = str(data.get("authorization_hmac_sha256") or "")
    unsigned = dict(data)
    unsigned.pop("authorization_hmac_sha256", None)
    actual = hmac.new(signing_key.encode("utf-8"), _canonical(unsigned), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(supplied, actual):
        raise PromotionEvidenceError("promotion evidence authorization signature mismatch")
    required = ("benchmark", "rollback", "capabilities", "forgetting")
    missing = [name for name in required if not isinstance(unsigned.get(name), dict)]
    if missing:
        raise PromotionEvidenceError(f"promotion evidence missing sections: {missing}")
    return unsigned


def evidence_flags(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    benchmark = dict(receipt.get("benchmark") or {})
    rollback = dict(receipt.get("rollback") or {})
    capabilities = dict(receipt.get("capabilities") or {})
    forgetting = dict(receipt.get("forgetting") or {})
    architecture = dict(receipt.get("architecture") or {})
    tools = dict(receipt.get("tools") or {})
    return {
        "optimization_candidate": bool(benchmark.get("candidate")),
        "matched_benchmark": bool(benchmark.get("passed") and benchmark.get("artifact_sha256")),
        "protected_capabilities_pass": bool(capabilities.get("passed") and capabilities.get("artifact_sha256")),
        "continual_learning": bool(forgetting.get("applicable")),
        "replay_or_forgetting_eval": bool(forgetting.get("passed") and forgetting.get("artifact_sha256")),
        "architecture_changed": bool(architecture.get("changed")),
        "reversible_module_boundary": bool(architecture.get("rollback_passed") and architecture.get("artifact_sha256")),
        "tool_capabilities_present": bool(tools.get("present")),
        "tool_registry_receipt": bool(tools.get("passed") and tools.get("artifact_sha256")),
        "rollback_target": rollback.get("checkpoint_id"),
        "rollback_verified": bool(rollback.get("passed") and rollback.get("artifact_sha256")),
    }
