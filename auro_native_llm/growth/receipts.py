"""Signed receipts for growth-ladder runs.

Follows the repo's receipt convention (``auro_native_llm/receipt.py``,
``auro_native_llm/streaming/receipts.py``): canonical JSON, sha256 digest,
explicit claim boundary. A rung receipt proves the rung's config, growth
provenance, training budget, and loss trajectory -- mechanics evidence only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RECEIPT_SCHEMA = "auro.growth.rung_receipt.v1"

CLAIM_BOUNDARY = (
    "Mechanics receipt for progressive-growth training on synthetic data. "
    "Proves the growth operators preserved the function at growth time, the "
    "training budget spent, and the loss trajectory observed. NOT a claim of "
    "capability, intelligence, or downstream-task gains: loss on synthetic "
    "data measures optimization mechanics only."
)


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _peak_rss_mb() -> float:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            rss /= 1024.0
        return round(rss / 1024.0, 2)
    except Exception:
        return -1.0


def finalize_rung_receipt(
    *,
    rung: int,
    model_id: str,
    config: Dict[str, Any],
    params: int,
    parent: Optional[Dict[str, Any]],
    growth_plan: Optional[str],
    preservation_max_abs_diff: Optional[float],
    steps: int,
    tokens_seen: int,
    loss_first: float,
    loss_last: float,
    loss_curve: List[float],
    elapsed_s: float,
    weights_sha256: str,
    data_provenance: str,
) -> Dict[str, Any]:
    """Build the signed receipt dict for one finished rung."""
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "kind": "growth-rung",
        "rung": int(rung),
        "model_id": model_id,
        "config": config,
        "config_sha256": _sha256_text(_canonical(config)),
        "params": int(params),
        "parent": parent,  # None for rung 0; else {rung, model_id, plan, ...}
        "growth_plan": growth_plan,
        "preservation_max_abs_diff": preservation_max_abs_diff,
        "steps": int(steps),
        "tokens_seen": int(tokens_seen),
        "loss_first": float(loss_first),
        "loss_last": float(loss_last),
        "loss_decreased": bool(loss_last < loss_first),
        "loss_curve": [float(x) for x in loss_curve],
        "elapsed_s": round(float(elapsed_s), 3),
        "peak_rss_mb": _peak_rss_mb(),
        "weights_sha256": weights_sha256,
        "data_provenance": data_provenance,
        "claim_boundary": CLAIM_BOUNDARY,
        "finished_unix": int(time.time()),
    }
    receipt["receipt_sha256"] = _sha256_text(_canonical(receipt))
    return receipt


def save_receipt(receipt: Dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return out
