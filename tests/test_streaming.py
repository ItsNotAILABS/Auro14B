"""Tests for the streaming-inference prototype (cold store + pager + receipts)."""
import json

import numpy as np
import pytest

from auro_native_llm.streaming import ExpertColdStore, ExpertPager, StreamingRun, finalize_receipt
from auro_native_llm.streaming.bench import run_bench
from auro_native_llm.streaming.receipts import _canonical, _sha256_text


def _weights(hidden=32, dim=64, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "gate_proj": rng.standard_normal((hidden, dim)),
        "up_proj": rng.standard_normal((hidden, dim)),
        "down_proj": rng.standard_normal((dim, hidden)),
    }


def test_cold_store_roundtrip(tmp_path):
    store = ExpertColdStore(tmp_path)
    w = _weights()
    store.save_expert(0, 3, w, meta={"note": "t"})
    back = store.load(0, 3)
    for k in ("gate_proj", "up_proj", "down_proj"):
        assert np.array_equal(np.asarray(back[k]), w[k])
    assert store.expert_nbytes(0, 3) == sum(a.nbytes for a in w.values())
    assert store.loads == 1


def test_cold_store_verify_and_manifest(tmp_path):
    store = ExpertColdStore(tmp_path)
    store.save_expert(1, 0, _weights())
    assert store.verify(1, 0) is True
    assert store.verify(1, 7) is False  # never saved
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["1:0"]["sha256"]
    assert manifest["1:0"]["shapes"]["gate_proj"] == [32, 64]


def test_cold_store_missing_expert(tmp_path):
    store = ExpertColdStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load(0, 0)


def test_pager_hit_miss_and_prefetch(tmp_path):
    store = ExpertColdStore(tmp_path)
    store.save_expert(0, 0, _weights(seed=1))
    store.save_expert(0, 1, _weights(seed=2))
    pager = ExpertPager(store, budget_bytes=10**9)

    pager.acquire(0, 0)  # miss
    pager.acquire(0, 0)  # hit
    assert pager.stats.misses == 1
    assert pager.stats.hits == 1

    assert pager.prefetch(0, 1) is True   # cold -> issued
    assert pager.prefetch(0, 1) is False  # already hot -> not re-issued
    pager.acquire(0, 1)                   # served from prefetch
    assert pager.stats.prefetch_hits == 1
    snap = pager.snapshot()
    assert snap["prefetch_hit_rate"] == pytest.approx(1.0)


def test_pager_respects_budget(tmp_path):
    store = ExpertColdStore(tmp_path)
    for e in range(4):
        store.save_expert(0, e, _weights(seed=e))
    one = store.expert_nbytes(0, 0)
    pager = ExpertPager(store, budget_bytes=int(one * 1.5))
    for e in range(4):
        pager.acquire(0, e)
    snap = pager.snapshot()
    assert snap["evictions"] > 0
    assert snap["peak_hot_bytes"] <= pager.budget_bytes
    assert snap["bytes_moved"] == 4 * one


def test_bench_end_to_end_receipt(tmp_path):
    # tight budget forces real paging: experts must be evicted and re-loaded
    receipt, pager, _moes = run_bench(
        layers=2, num_experts=4, top_k=2, hidden=32, expert_dim=64,
        seq_len=8, budget_mb=0.05, prefetch=True, seed=7,
    )
    assert receipt["schema"] == "auro.streaming.receipt.v1"
    assert receipt["tokens"] == 8
    assert receipt["tokens_per_s"] > 0
    assert receipt["pager"]["bytes_moved"] > 0
    assert len(receipt["trace"]) == 8
    assert all(t["num_fired"] > 0 for t in receipt["trace"])
    # digest verifies over canonical JSON
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert _sha256_text(_canonical(body)) == receipt["receipt_sha256"]
    assert "synthetic" in receipt["claim_boundary"].lower()
    # tight budget forces evictions; the receipt must say so honestly
    assert receipt["pager"]["evictions"] > 0


def test_bench_deterministic(tmp_path):
    kwargs = dict(layers=2, num_experts=4, top_k=2, hidden=32, expert_dim=64,
                  seq_len=8, budget_mb=4, prefetch=False, seed=123)
    r1, _, _ = run_bench(**kwargs)
    r2, _, _ = run_bench(**kwargs)
    t1 = [[e for e in t["fired"]] for t in r1["trace"]]
    t2 = [[e for e in t["fired"]] for t in r2["trace"]]
    assert t1 == t2


def test_lru_thrashes_on_cyclic_scan(tmp_path):
    """Documents a measured finding: LRU misses ~everything when the expert
    firing pattern cycles through more experts than fit in the hot set.

    This is the textbook LRU scan pathology, reproduced against the real
    pager. It is the baseline the next eviction policy (e.g. a TinyLFU-style
    admission filter) has to beat -- the bench exists to measure that race.
    """
    store = ExpertColdStore(tmp_path)
    rng = np.random.default_rng(0)
    for e in range(8):
        store.save_expert(0, e, {"gate_proj": rng.standard_normal((8, 8)),
                                 "up_proj": rng.standard_normal((8, 8)),
                                 "down_proj": rng.standard_normal((8, 8))})
    one = store.expert_nbytes(0, 0)
    pager = ExpertPager(store, budget_bytes=int(one * 2.5))  # room for ~2
    pattern = [0, 1, 2, 3, 0, 4, 5, 6, 7]
    for _ in range(10):
        for e in pattern:
            pager.acquire(0, e)
    snap = pager.snapshot()
    assert snap["evictions"] > 0
    assert snap["hit_rate"] < 0.05
    assert snap["eviction_policy"] == "lru"


def test_streaming_run_truncation():
    run = StreamingRun({"x": 1})
    for i in range(5000):
        run.log_token(i, [(0, 0)], 0, 0, 0.0)
    receipt = finalize_receipt(run, {"bytes_moved": 0}, {}, 1.0, "synthetic-random-init")
    assert receipt["trace_truncated"] is True
    assert len(receipt["trace"]) == 4096
