"""Tests for the universal growth + train-leg engine.

Covers: Net2Net widening preservation (exact function match), growth
planning (nearest-to-2x subset), MLPLeg training (loss falls below the
chance floor on the teacher-student task), checkpoint save/load roundtrip,
and leg receipt schema/signature.
"""

import json
import math

import numpy as np
import pytest

from auro_native_llm.growth.engine import (
    MLPLeg,
    ce_loss,
    finalize_leg_receipt,
    grow_mlp_params,
    latest_leg_dir,
    mlp_param_count,
    net2net_widen_pair,
    plan_mlp_growth,
    relu,
    run_legs,
    synthetic_classes_task,
)
from auro_native_llm.growth.core import Tensor


def test_net2net_widen_pair_preserves_function():
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((16, 24)).astype(np.float32)
    b1 = rng.standard_normal(24).astype(np.float32)
    W2 = rng.standard_normal((24, 8)).astype(np.float32)
    X = rng.standard_normal((32, 16)).astype(np.float32)

    def fwd(X, W1, b1, W2):
        h = X @ W1 + b1
        h = h / (1.0 + np.exp(-h))  # silu
        return h @ W2

    before = fwd(X, W1, b1, W2)
    W1n, b1n, W2n = net2net_widen_pair(W1, b1, W2, factor=2)
    after = fwd(X, W1n, b1n, W2n)
    assert W1n.shape == (16, 48) and W2n.shape == (48, 8)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)


def test_net2net_widen_pair_factor3():
    rng = np.random.default_rng(1)
    W1 = rng.standard_normal((8, 8)).astype(np.float32)
    b1 = rng.standard_normal(8).astype(np.float32)
    W2 = rng.standard_normal((8, 4)).astype(np.float32)
    X = rng.standard_normal((16, 8)).astype(np.float32)
    before = (np.maximum(X @ W1 + b1, 0)) @ W2  # relu here for variety
    W1n, b1n, W2n = net2net_widen_pair(W1, b1, W2, factor=3)
    after = (np.maximum(X @ W1n + b1n, 0)) @ W2n
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)


def test_plan_mlp_growth_targets_2x():
    flags, ratio = plan_mlp_growth([32, 64, 64, 8], 2.0)
    assert 1.5 < ratio <= 2.5
    # widening must actually change dims
    assert any(flags)
    # achieved ratio matches recount
    dims = [32, 64, 64, 8]
    grown = [d * 2 if f else d for d, f in zip([None] + dims[1:-1] + [None], [False] + flags + [False])]
    # simpler direct check:
    new_dims = list(dims)
    for i, f in enumerate(flags):
        if f:
            new_dims[i + 1] *= 2
    assert abs(mlp_param_count(new_dims) / mlp_param_count(dims) - ratio) < 1e-9


def test_grow_mlp_params_end_to_end():
    task = synthetic_classes_task(16, 4, seed=0)
    leg = MLPLeg([16, 32, 32, 4], task=task, seed=0)
    rng = np.random.default_rng(0)
    X, _ = task(rng, 16)
    before = leg.forward_np(X)
    grown, plan = leg.grow(2.0)
    after = grown.forward_np(X)
    assert grown.param_count() > leg.param_count()
    assert abs(grown.param_count() / leg.param_count() - plan["achieved_ratio"]) < 1e-9
    np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-4)


def test_mlp_leg_learns_below_chance():
    task = synthetic_classes_task(n_in=16, n_classes=4, seed=0)
    leg = MLPLeg([16, 48, 48, 4], task=task, lr=3e-3, seed=1)
    rng = np.random.default_rng(0)
    losses = [leg.train_step(*task(rng, 128)) for _ in range(40)]
    assert losses[-1] < losses[0]
    assert losses[-1] < math.log(4) - 0.2  # well below chance floor


def test_ce_loss_gradient_correctness():
    # finite-difference check of ce_loss backward
    rng = np.random.default_rng(0)
    logits = Tensor(rng.standard_normal((8, 5)).astype(np.float32), requires_grad=True)
    y = rng.integers(0, 5, size=8)
    loss = ce_loss(logits, y)
    loss.backward()
    analytic = logits.grad.copy()
    eps = 1e-3
    numeric = np.zeros_like(analytic)
    for i in range(8):
        for j in range(5):
            lp = logits.data.copy()
            lp[i, j] += eps
            l_plus = -np.mean(
                (lp - np.log(np.exp(lp).sum(-1, keepdims=True)))[np.arange(8), y]
            )
            lp[i, j] -= 2 * eps
            l_minus = -np.mean(
                (lp - np.log(np.exp(lp).sum(-1, keepdims=True)))[np.arange(8), y]
            )
            numeric[i, j] = (l_plus - l_minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1e-3)


def test_save_load_roundtrip(tmp_path):
    task = synthetic_classes_task(16, 4, seed=0)
    leg = MLPLeg([16, 32, 4], task=task, seed=0)
    meta = {"model_id": "test", "leg": 0}
    saved = leg.save(tmp_path / "leg-0", meta)
    assert len(saved["sha256"]) == 64
    leg2 = MLPLeg.load(tmp_path / "leg-0", task=task, lr=1e-3)
    assert leg2.param_count() == leg.param_count()
    for k in leg.params:
        np.testing.assert_array_equal(leg2.params[k], leg.params[k])


def test_run_legs_compounds(tmp_path):
    task = synthetic_classes_task(16, 4, seed=0)
    leg = MLPLeg([16, 32, 32, 4], task=task, lr=3e-3, seed=1)
    summaries = run_legs(
        leg, model_id="test-compound", num_legs=2, steps_per_leg=30,
        batch=64, seed=0, ckpt_root=tmp_path, log_every=30,
    )
    assert len(summaries) == 2
    assert summaries[1]["params"] > summaries[0]["params"]
    for s in summaries:
        ckpt = tmp_path / f"leg-{s['leg']}"
        receipt = json.loads((ckpt / "receipt.json").read_text())
        assert receipt["schema"] == "auro.engine.leg_receipt.v1"
        assert receipt["loss_decreased"] is True
        assert receipt["receipt_sha256"]
    # leg 0 starts fresh: no parent, no growth yet. leg 1 grew from leg 0.
    r0 = json.loads((tmp_path / "leg-0" / "receipt.json").read_text())
    assert r0["parent"] is None
    assert r0["preservation_max_abs_diff"] is None
    r1 = json.loads((tmp_path / "leg-1" / "receipt.json").read_text())
    assert r1["preservation_max_abs_diff"] is not None
    assert r1["preservation_max_abs_diff"] < 1e-4
    # leg-1's parent chains to leg-0
    r1 = json.loads((tmp_path / "leg-1" / "receipt.json").read_text())
    assert r1["parent"]["leg"] == 0
    # resume continues numbering instead of restarting
    task2 = synthetic_classes_task(16, 4, seed=0)
    leg_fresh = MLPLeg([16, 32, 32, 4], task=task2, lr=3e-3, seed=1)
    more = run_legs(leg_fresh, model_id="test-compound", num_legs=1,
                    steps_per_leg=10, batch=32, seed=0, ckpt_root=tmp_path,
                    log_every=10)
    assert more[0]["leg"] == 2
    assert latest_leg_dir(tmp_path)[0] == 2


def test_finalize_leg_receipt_schema():
    r = finalize_leg_receipt(
        leg=0, model_id="m", family="mlp", config={"a": 1}, params=100,
        parent=None, growth_plan=None, preservation_max_abs_diff=None,
        steps=10, units_seen=640, unit="examples",
        loss_first=2.0, loss_last=1.5, loss_curve=[2.0, 1.5],
        elapsed_s=1.0, steps_per_s=10.0, weights_sha256="ab" * 32,
        data_provenance="synthetic",
    )
    assert r["schema"] == "auro.engine.leg_receipt.v1"
    assert r["kind"] == "train-leg"
    assert r["loss_decreased"] is True
    assert len(r["receipt_sha256"]) == 64
    assert "NOT a claim" in r["claim_boundary"]
