"""Tests for Auro exponential-growth mechanics (career ladder).

Mechanics only: function preservation of the growth operators, schedule
math, checkpoint round-trips, and a tiny train -> grow -> train run on
synthetic data. No capability claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from auro_native_llm.growth.core import (
    GrowthConfig,
    GrowthTransformer,
    causal_lm_loss,
    count_params,
)
from auro_native_llm.growth import operators as ops
from auro_native_llm.growth.schedule import (
    LANE_2B_TOTAL,
    apply_plan,
    num_rungs,
    plan_growth,
    rung_targets,
)
from auro_native_llm.growth.career import (
    CheckpointError,
    find_156k_checkpoint,
    latest_completed_rung,
    load_156k_checkpoint,
    load_rung,
    run_career,
    save_rung,
    synthetic_batch,
)


def _small_cfg(**kw):
    base = dict(vocab=64, d=32, layers=2, heads=2, kv_heads=2, head_dim=16,
                ffn=64, seq=16, seed=11)
    base.update(kw)
    return GrowthConfig(**base)


def _preservation_diff(cfg, grow_fn, **kw):
    rng = np.random.default_rng(7)
    m = GrowthTransformer(cfg)
    tok = rng.integers(0, cfg.vocab, size=(2, cfg.seq))
    before = m.forward(tok).data
    p2, c2 = grow_fn(m.params, cfg, **kw)
    assert sum(v.size for v in p2.values()) == count_params(c2)
    after = GrowthTransformer(c2, p2).forward(tok).data
    return float(np.max(np.abs(before - after))), count_params(cfg), count_params(c2)


def test_width_preserves_function():
    diff, p0, p1 = _preservation_diff(_small_cfg(), ops.grow_width)
    assert diff < 1e-4, diff
    assert p1 > p0


def test_width_preserves_function_gqa():
    cfg = _small_cfg(heads=4, kv_heads=2)  # genuine GQA
    diff, _, _ = _preservation_diff(cfg, ops.grow_width)
    assert diff < 1e-4, diff


def test_depth_preserves_function():
    diff, p0, p1 = _preservation_diff(_small_cfg(), ops.grow_depth)
    assert diff < 1e-4, diff
    assert p1 > p0


def test_moe_introduction_preserves_function():
    diff, _, _ = _preservation_diff(_small_cfg(), ops.grow_moe)
    assert diff < 1e-4, diff


def test_moe_duplication_preserves_function():
    cfg = _small_cfg(layers=3, moe_layers=(1,), experts=2, kv_heads=1)
    diff, _, _ = _preservation_diff(cfg, ops.grow_moe)
    assert diff < 1e-4, diff


def test_chained_growth_preserves_function():
    rng = np.random.default_rng(7)
    cfg = _small_cfg(layers=3, moe_layers=(1,), experts=2, kv_heads=1)
    m = GrowthTransformer(cfg)
    tok = rng.integers(0, cfg.vocab, size=(2, cfg.seq))
    before = m.forward(tok).data
    p, c = ops.grow_width(m.params, cfg)
    p, c = ops.grow_depth(p, c)
    p, c = ops.grow_moe(p, c)
    after = GrowthTransformer(c, p).forward(tok).data
    assert float(np.max(np.abs(before - after))) < 1e-4


def test_schedule_doubles_and_reaches_2b():
    rungs = rung_targets(156_000)
    assert rungs[0] == 156_000
    assert rungs[1] == 312_000
    assert rungs[-1] == LANE_2B_TOTAL
    assert num_rungs(156_000) == 15


def test_plan_growth_lands_near_target():
    cfg = _small_cfg()
    p0 = count_params(cfg)
    name, new_cfg, achieved = plan_growth(cfg, p0 * 2)
    assert 1.2 * p0 < achieved < 3.0 * p0, (name, achieved)
    # the plan must be applicable to real params and preserve function
    diff, _, _ = _preservation_diff(cfg, apply_plan, plan_name=name)
    assert diff < 1e-4, (name, diff)


def test_checkpoint_roundtrip(tmp_path):
    cfg = _small_cfg()
    m = GrowthTransformer(cfg)
    meta = {"model_id": "Auro-156K-growth-r0", "rung": 0}
    save_rung(m, tmp_path / "rung-0", meta)
    assert (tmp_path / "rung-0" / "weights.npz").exists()
    assert (tmp_path / "rung-0" / "config.json").exists()
    m2 = load_rung(tmp_path / "rung-0")
    assert m2.param_count() == m.param_count()
    rng = np.random.default_rng(3)
    tok = rng.integers(0, cfg.vocab, size=(2, cfg.seq))
    assert np.allclose(m.forward(tok).data, m2.forward(tok).data, atol=1e-6)


def test_synthetic_data_is_learnable_signal():
    rng = np.random.default_rng(0)
    inputs, targets = synthetic_batch(rng, 8, 16, 64)
    assert inputs.shape == (8, 16) and targets.shape == (8, 16)
    # modular-arithmetic successor dominates: most targets follow the rule
    rule = (37 * inputs + 11) % 64
    assert float((targets == rule).mean()) > 0.4


def test_tiny_train_grow_train_mechanics(tmp_path):
    """End-to-end mechanics: loss falls, growth preserves, loss falls again."""
    cfg = GrowthConfig(vocab=64, d=32, layers=2, heads=2, kv_heads=2,
                       head_dim=16, ffn=32, seq=32, seed=5)
    summaries = run_career(cfg, num_growths=1, steps_per_rung=60, batch=16,
                           lr=5e-4, seed=1, out_dir=tmp_path / "ckpt")
    assert len(summaries) == 2
    r0, r1 = summaries
    assert r0["loss_last"] < r0["loss_first"], "rung-0 loss did not decrease"
    assert r1["params"] > r0["params"], "growth did not add params"
    assert r1["loss_last"] < r1["loss_first"], "rung-1 loss did not decrease"
    # every rung checkpointed as a new model with a receipt
    for s in summaries:
        ckpt = Path(s["ckpt"])
        assert (ckpt / "weights.npz").exists()
        assert (ckpt / "config.json").exists()
        receipt = json.loads((ckpt / "receipt.json").read_text())
        assert receipt["schema"] == "auro.growth.rung_receipt.v1"
        assert "receipt_sha256" in receipt
        assert "mechanics" in receipt["claim_boundary"].lower() or \
            "Mechanics" in receipt["claim_boundary"]


# ----------------------------------------------------------------------------
# Regression tests for the three fixed bugs + resume compounding
# ----------------------------------------------------------------------------

def _softmax_np(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    e = np.exp(a - m)
    return e / e.sum(axis=axis, keepdims=True)


def test_schedule_never_overshoots_target():
    """Bug 1 regression: the rung schedule is monotonic, capped at the
    target, and its length matches num_rungs() + 1."""
    for p0 in (156_000, 1_000_000, LANE_2B_TOTAL, LANE_2B_TOTAL + 1):
        rungs = rung_targets(p0)
        assert all(b >= a for a, b in zip(rungs, rungs[1:])), \
            f"non-monotonic schedule for p0={p0}"
        if p0 <= LANE_2B_TOTAL:
            # a schedule starting past the target cannot shrink to it, but it
            # must never *grow past* it either
            assert rungs[-1] <= LANE_2B_TOTAL, f"overshoot for p0={p0}"
        assert len(rungs) == num_rungs(p0) + 1, (p0, len(rungs))
    rungs = rung_targets(156_000)
    assert rungs[0] == 156_000
    assert rungs[-1] == LANE_2B_TOTAL  # capped exactly on target, no overshoot rung
    assert len(rungs) == 16           # 15 doublings, matching the README


def test_moe_expert_duplication_router_columns():
    """Bug 2 regression: growing experts on an EXISTING MoE layer duplicates
    router *columns* (d, E) -> (d, 2E); the routing distribution halves per
    copy and the full forward is preserved. (The old code duplicated rows,
    producing (2d, E) and crashing the matmul.)"""
    cfg = _small_cfg(layers=3, moe_layers=(1,), experts=2, kv_heads=1)
    rng = np.random.default_rng(7)
    m = GrowthTransformer(cfg)
    p2, c2 = ops.grow_moe(m.params, cfg, layers=[1])
    assert c2.experts == 4
    assert c2.moe_layers == (1,)
    r_before = m.params["b1.router"]
    r_after = p2["b1.router"]
    assert r_before.shape == (cfg.d, 2)
    assert r_after.shape == (cfg.d, 4), r_after.shape
    # duplicated experts are exact copies
    for i in range(2):
        for suf in ("g", "u", "d"):
            assert np.array_equal(p2[f"b1.e{2 + i}.{suf}"], m.params[f"b1.e{i}.{suf}"])
    # routing distribution: each pair gets half the original mass
    x = rng.standard_normal((2, cfg.seq, cfg.d)).astype(np.float32)
    pb = _softmax_np(x @ r_before, axis=-1)
    pa = _softmax_np(x @ r_after, axis=-1)
    assert pa.shape == (2, cfg.seq, 4)
    assert np.allclose(pa[..., :2], pa[..., 2:], atol=1e-6), "pair mass not halved"
    assert np.allclose(pa[..., :2] * 2.0, pb, atol=1e-6), "distribution changed"
    # full function preservation through the fixed path
    tok = rng.integers(0, cfg.vocab, size=(2, cfg.seq))
    before = m.forward(tok).data
    after = GrowthTransformer(c2, p2).forward(tok).data
    assert float(np.max(np.abs(before - after))) < 1e-4
    assert sum(v.size for v in p2.values()) == count_params(c2)


def test_moe_introduction_matches_expert_count():
    """Dense->MoE conversion creates cfg.experts experts (not hardcoded 2),
    so the per-layer count always matches the config (fixes a latent
    KeyError when growing MoE onto a config that already had E > 2)."""
    cfg = _small_cfg(layers=2, moe_layers=(0,), experts=4)
    m = GrowthTransformer(cfg)
    p2, c2 = ops.grow_moe(m.params, cfg, layers=[1])
    assert c2.experts == 4
    assert p2["b1.router"].shape == (cfg.d, 4)
    tok = np.random.default_rng(9).integers(0, cfg.vocab, size=(2, cfg.seq))
    before = m.forward(tok).data
    after = GrowthTransformer(c2, p2).forward(tok).data
    assert float(np.max(np.abs(before - after))) < 1e-4


def test_156k_checkpoint_discovery_and_fallback(tmp_path, monkeypatch):
    """Bug 3: a real 156K checkpoint is discovered and loaded when present;
    incompatible weights raise CheckpointError; absence falls back cleanly."""
    from auro_native_llm.growth.career import _init_rung_zero

    d = tmp_path / "ckpt156"
    d.mkdir()
    monkeypatch.setenv("AURO_156K_CHECKPOINT_DIR", str(d))
    assert find_156k_checkpoint() is None  # nothing dropped in yet

    cfg = _small_cfg()
    m = GrowthTransformer(cfg)
    save_rung(m, d, {"model_id": "Auro-156K"})
    assert find_156k_checkpoint() == d
    m2 = load_156k_checkpoint(d)
    assert m2.param_count() == m.param_count()
    tok = np.random.default_rng(3).integers(0, cfg.vocab, size=(2, cfg.seq))
    assert np.allclose(m.forward(tok).data, m2.forward(tok).data, atol=1e-6)

    # incompatible weights -> clear error, not silent corruption
    np.savez(d / "weights.npz", **{"bogus": np.zeros(3, dtype=np.float32)})
    with pytest.raises(CheckpointError):
        load_156k_checkpoint(d)

    # absence -> clean fallback to random init (mechanics mode)
    (d / "weights.npz").unlink()
    assert find_156k_checkpoint() is None
    m3 = _init_rung_zero(cfg, init_from_156k=True)
    assert m3.param_count() == count_params(cfg)


def test_latest_completed_rung_ignores_incomplete(tmp_path):
    cfg = _small_cfg()
    m = GrowthTransformer(cfg)
    out = tmp_path / "ckpt"
    save_rung(m, out / "rung-0", {"rung": 0})
    (out / "rung-0" / "receipt.json").write_text("{}")
    # rung-1 interrupted mid-training: no receipt -> not resumable
    (out / "rung-1").mkdir(parents=True)
    np.savez(out / "rung-1" / "weights.npz", **{"x": np.zeros(2)})
    (out / "rung-1" / "config.json").write_text("{}")
    assert latest_completed_rung(out)[0] == 0
    assert latest_completed_rung(tmp_path / "nope") is None


def test_resume_compounds_from_latest_rung(tmp_path):
    """Resume loads the latest *completed* rung and continues the ladder:
    rung numbering continues, old checkpoints are untouched, the receipt
    chain links back, and params keep growing."""
    cfg = GrowthConfig(vocab=64, d=32, layers=2, heads=2, kv_heads=2,
                       head_dim=16, ffn=32, seq=32, seed=5)
    out = tmp_path / "ckpt"
    s1 = run_career(cfg, num_growths=1, steps_per_rung=40, batch=16,
                    lr=5e-4, seed=1, out_dir=out, init_from_156k=False)
    assert [s["rung"] for s in s1] == [0, 1]
    w0_before = (out / "rung-0" / "weights.npz").read_bytes()
    w1_before = (out / "rung-1" / "weights.npz").read_bytes()

    s2 = run_career(cfg, num_growths=1, steps_per_rung=40, batch=16,
                    lr=5e-4, seed=2, out_dir=out, resume=True,
                    init_from_156k=False)
    assert [s["rung"] for s in s2] == [2, 3], "rung numbering did not continue"
    # compounding, not restarting: earlier checkpoints byte-identical
    assert (out / "rung-0" / "weights.npz").read_bytes() == w0_before
    assert (out / "rung-1" / "weights.npz").read_bytes() == w1_before
    # new rungs actually trained and still growing
    for s in s2:
        assert s["loss_last"] < s["loss_first"], f"rung {s['rung']} did not learn"
    assert s2[0]["params"] > s1[-1]["params"], "params did not grow on resume"
    # receipt chain links rung-2 back to the resumed rung-1
    meta2 = json.loads((out / "rung-2" / "rung.json").read_text())
    assert meta2["parent"]["rung"] == 1
    assert meta2["parent"]["model_id"] == "Auro-156K-growth-r1"
    assert (out / "rung-2" / "receipt.json").exists()
    assert (out / "rung-3" / "receipt.json").exists()
