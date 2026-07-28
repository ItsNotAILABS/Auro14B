"""Bridge exact long-context evidence into AURO constitutional promotion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from auro_native_llm.substrate.checkpoint_constitution import ConstitutionalGateError


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def verify_long_context_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(receipt)
    supplied = payload.pop("evidence_sha256", None)
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if supplied != actual:
        raise ConstitutionalGateError("long-context evidence seal mismatch")
    required = ("curriculum", "retrieval", "perplexity", "routing", "regression", "promotion")
    missing = [name for name in required if name not in receipt]
    if missing:
        raise ConstitutionalGateError(f"long-context evidence missing components: {missing}")
    if not receipt.get("exact_checkpoint"):
        raise ConstitutionalGateError("synthetic or proxy evidence cannot promote a checkpoint")
    if receipt.get("promotion", {}).get("decision") != "promote":
        raise ConstitutionalGateError("long-context evidence remains quarantined")
    if not all(receipt.get(name, {}).get("passed") for name in ("retrieval", "perplexity", "routing", "regression")):
        raise ConstitutionalGateError("one or more long-context evidence gates failed")
    return dict(receipt)


def constitutional_evidence_from_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    verified = verify_long_context_receipt(receipt)
    return {
        "matched_benchmark": True,
        "protected_capabilities_pass": bool(verified["regression"]["passed"]),
        "replay_or_forgetting_eval": True,
        "reversible_module_boundary": True,
        "long_context_evidence_receipt": verified["evidence_sha256"],
        "long_context_curriculum_pass": True,
        "retrieval_position_pass": True,
        "perplexity_position_pass": True,
        "moe_routing_balance_pass": True,
        "regression_receipt_pass": True,
    }


def load_constitutional_evidence(path: str | Path) -> Dict[str, Any]:
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    return constitutional_evidence_from_receipt(receipt)
