"""Streaming-inference benchmark: stale-by-default expert paging, measured.

Builds a small ``MixtureOfExperts`` stack, cold-stores every expert, then
decodes a token sequence paging experts in on demand:

* router fires experts per token (real ``TopKRouter`` routing)
* ``ExpertPager`` pages fired experts in under a byte budget (LRU + prefetch)
* expert forward passes run against the paged (mmap'd) weights
* a locality predictor prefetches the current token's experts for the next one

Emits a signed receipt (``receipts.finalize_receipt``) with per-token expert
traces, pager stats, router stats and memory numbers.

Weights are synthetic random-init unless the caller wires in trained ones;
the receipt's claim boundary says so.
"""
from __future__ import annotations

import argparse
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mesie.foundation.models.mixture_of_experts import MixtureOfExperts

from .cold_store import ExpertColdStore
from .pager import ExpertPager
from .receipts import StreamingRun, finalize_receipt, save_receipt


def _activate(x: np.ndarray, activation: str) -> np.ndarray:
    if activation in ("swiglu", "silu"):
        return x * (1.0 / (1.0 + np.exp(-x)))
    if activation in ("geglu", "gelu"):
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    if activation == "relu":
        return np.maximum(0, x)
    return x


def _expert_forward(weights: Dict[str, np.ndarray], x: np.ndarray, activation: str) -> np.ndarray:
    gate = _activate(np.einsum("...d,dh->...h", x, weights["gate_proj"]), activation)
    up = np.einsum("...d,dh->...h", x, weights["up_proj"])
    return np.einsum("...h,ho->...o", gate * up, weights["down_proj"])


def run_bench(
    layers: int = 2,
    num_experts: int = 4,
    top_k: int = 2,
    hidden: int = 64,
    expert_dim: int = 128,
    seq_len: int = 16,
    budget_mb: float = 1.0,
    prefetch: bool = True,
    evict_policy: str = "lru",
    seed: int = 0,
    store_dir: str | None = None,
) -> Tuple[Dict[str, Any], ExpertPager, List[MixtureOfExperts]]:
    """Run the streaming bench. Returns (receipt, pager, moe_layers)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    config = {
        "layers": layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden": hidden,
        "expert_dim": expert_dim,
        "seq_len": seq_len,
        "budget_bytes": int(budget_mb * 1024 * 1024),
        "prefetch": bool(prefetch),
        "prefetch_predictor": "temporal-locality(previous-token-fired)",
        "prefetch_policy": "never-evicts-demand-loaded-experts",
        "eviction_policy": evict_policy,
        "shared_expert": "pinned-resident-always-active",
        "seed": seed,
    }

    tmp = tempfile.TemporaryDirectory() if store_dir is None else None
    root = Path(store_dir) if store_dir else Path(tmp.name)
    store = ExpertColdStore(root)
    moes = [
        MixtureOfExperts(
            hidden_dim=hidden,
            num_experts=num_experts,
            top_k=top_k,
            expert_dim=expert_dim,
            modality_aware=False,
            adaptive_compute=False,
        )
        for _ in range(layers)
    ]
    for li, moe in enumerate(moes):
        store.save_moe_layer(li, moe)
    # release the in-RAM originals: from here on, experts only exist cold
    for moe in moes:
        for e in moe.experts:
            del e.gate_proj, e.up_proj, e.down_proj

    pager = ExpertPager(store, config["budget_bytes"], policy=evict_policy)
    run = StreamingRun(config)

    t0 = time.perf_counter()
    prev_fired: List[List[int]] = [[] for _ in range(layers)]
    for t in range(seq_len):
        if prefetch:
            # Policy: prefetch never evicts demand-loaded experts. It only
            # spends headroom, so a bad prediction can waste bytes but never
            # thrash the hot set.
            for li, experts in enumerate(prev_fired):
                for e in experts:
                    need = store.expert_nbytes(li, e)
                    if pager.stats.hot_bytes + need <= pager.budget_bytes:
                        pager.prefetch(li, e)
        fired_all: List[Tuple[int, int]] = []
        stall_before = pager.stats.demand_stall_ms
        ph_before = pager.stats.prefetch_hits
        # Fresh synthetic token embedding each step: exercises diverse routing
        # the way real decode hidden states vary per token. Seeded -> deterministic.
        x = rng.standard_normal(hidden).reshape(1, hidden)
        for li, moe in enumerate(moes):
            indices, weights, _info = moe.router.route(x, training=False)
            fired: List[int] = []
            out = np.zeros_like(x)
            for k in range(moe.top_k):
                eid = int(indices[0, k])
                w = float(weights[0, k])
                if eid < 0 or w <= 0.0:
                    continue
                arrs = pager.acquire(li, eid)
                out += _expert_forward(arrs, x, moe.experts[0].activation) * w
                fired.append(eid)
                fired_all.append((li, eid))
            # shared expert is pinned resident (always active, disclosed in config)
            out += moe.shared_expert.forward(x) * moe.shared_weight
            x = out
            prev_fired[li] = fired
        run.log_token(
            t,
            fired_all,
            ph_before,
            pager.stats.prefetch_hits,
            pager.stats.demand_stall_ms - stall_before,
        )
    elapsed = time.perf_counter() - t0

    router_stats: Dict[str, Any] = {"per_layer": []}
    entropies: List[float] = []
    for moe in moes:
        st = moe.get_expert_statistics()
        router_stats["per_layer"].append(
            {
                "router_entropy": st["router_entropy"],
                "expert_firings": {
                    k: v for k, v in st.items() if k.startswith("expert_")
                },
            }
        )
        entropies.append(st["router_entropy"])
    router_stats["mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0

    receipt = finalize_receipt(
        run,
        pager.snapshot(),
        router_stats,
        elapsed,
        weights_provenance="synthetic-random-init",
    )
    if tmp is not None:
        tmp.cleanup()
    return receipt, pager, moes


def main() -> None:
    ap = argparse.ArgumentParser(description="Auro streaming-inference bench")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--expert-dim", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--budget-mb", type=float, default=1.0)
    ap.add_argument("--prefetch", dest="prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--evict-policy", choices=["lru", "tinylfu"], default="lru")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="streaming_receipt.json")
    args = ap.parse_args()

    receipt, _pager, _moes = run_bench(
        layers=args.layers,
        num_experts=args.experts,
        top_k=args.top_k,
        hidden=args.hidden,
        expert_dim=args.expert_dim,
        seq_len=args.seq_len,
        budget_mb=args.budget_mb,
        prefetch=args.prefetch,
        evict_policy=args.evict_policy,
        seed=args.seed,
    )
    path = save_receipt(receipt, args.out)
    p = receipt["pager"]
    print(f"receipt: {path}")
    print(f"tokens={receipt['tokens']} tok/s={receipt['tokens_per_s']} "
          f"peak_rss_mb={receipt['peak_rss_mb']}")
    print(f"pager: hit_rate={p['hit_rate']:.2f} prefetch_hit_rate={p['prefetch_hit_rate']:.2f} "
          f"bytes_moved={p['bytes_moved']} peak_hot_bytes={p['peak_hot_bytes']} "
          f"evictions={p['evictions']}")
    print(f"receipt_sha256={receipt['receipt_sha256']}")


if __name__ == "__main__":
    main()
