"""Signed receipts for streaming-inference benchmark runs.

Follows the repo's receipt convention (``auro_native_llm/receipt.py``):
canonical JSON, sha256 digest, explicit claim boundary. A streaming receipt
proves *which* experts fired for every token, what the pager moved, and what
it cost -- the routing-quality evidence no one else publishes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

RECEIPT_SCHEMA = "auro.streaming.receipt.v1"
MAX_TRACE_TOKENS = 4096


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StreamingRun:
    """Accumulates per-token expert traces for one benchmark run."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.started_unix = int(time.time())
        self._traces: List[Dict[str, Any]] = []
        self._truncated = False

    def log_token(
        self,
        token_idx: int,
        fired: List[Tuple[int, int]],
        prefetch_hits_before: int,
        prefetch_hits_after: int,
        demand_stall_ms: float,
    ) -> None:
        """Record which (layer, expert) pairs fired for one token."""
        if len(self._traces) >= MAX_TRACE_TOKENS:
            self._truncated = True
            return
        self._traces.append(
            {
                "token": int(token_idx),
                "fired": [[int(layer), int(expert)] for layer, expert in fired],
                "num_fired": len(fired),
                "prefetch_hit_delta": int(prefetch_hits_after - prefetch_hits_before),
                "demand_stall_ms": round(float(demand_stall_ms), 4),
            }
        )

    @property
    def num_tokens(self) -> int:
        return len(self._traces)


def _peak_rss_mb() -> float:
    try:
        import resource

        # ru_maxrss is KiB on Linux, bytes on macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            rss /= 1024.0
        return round(rss / 1024.0, 2)
    except Exception:
        return -1.0


def finalize_receipt(
    run: StreamingRun,
    pager_snapshot: Dict[str, Any],
    router_stats: Dict[str, Any],
    elapsed_s: float,
    weights_provenance: str,
) -> Dict[str, Any]:
    """Build the signed receipt dict for a finished run."""
    tokens = run.num_tokens
    fired_counts = [t["num_fired"] for t in run._traces]
    total_fired = sum(fired_counts)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "kind": "streaming-bench",
        "config": run.config,
        "config_sha256": _sha256_text(_canonical(run.config)),
        "weights_provenance": weights_provenance,
        "started_unix": run.started_unix,
        "elapsed_s": round(float(elapsed_s), 3),
        "tokens": tokens,
        "tokens_per_s": round(tokens / elapsed_s, 3) if elapsed_s > 0 else 0.0,
        "ms_per_token": round(elapsed_s * 1000.0 / tokens, 3) if tokens else 0.0,
        "peak_rss_mb": _peak_rss_mb(),
        "pager": pager_snapshot,
        "router": router_stats,
        "experts_fired_total": total_fired,
        "experts_fired_per_token_mean": round(total_fired / tokens, 3) if tokens else 0.0,
        "bytes_moved_per_token": round(
            pager_snapshot.get("bytes_moved", 0) / tokens, 1
        ) if tokens else 0.0,
        "trace_truncated": run._truncated,
        "trace": run._traces,
        "claim_boundary": (
            "Mechanics benchmark on the stated weight provenance; measures routing, "
            "paging and memory behavior only. Not a model-quality, perplexity, or "
            "downstream-task claim. Synthetic weights prove the pipeline, not the model."
            if weights_provenance == "synthetic-random-init"
            else "Mechanics benchmark; weight provenance stated above, quality claims out of scope."
        ),
    }
    digest = _sha256_text(_canonical(receipt))
    receipt["receipt_sha256"] = digest
    return receipt


def save_receipt(receipt: Dict[str, Any], path: str | Path) -> Path:
    """Write the receipt JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return out
