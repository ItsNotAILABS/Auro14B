"""Universal exponential-growth + train-leg engine.

Every real checkpoint in the model registry can take its next leg through a
:class:`ModelAdapter`. One leg = resume the latest checkpoint -> train N
steps -> checkpoint + signed receipt -> grow ~2x with function preservation
verified numerically -> the next leg starts from the grown model. Legs
compound; nothing restarts from scratch.

Why this is the answer to the cost objection
--------------------------------------------
From-scratch pretraining quotes (hundreds of thousands to millions of
dollars for billion-parameter models) price the *naive* path: random init,
giant corpus, giant cluster. The millions of models on Hugging Face exist
because almost nobody pays that price -- the ecosystem runs on fine-tunes,
distillations, merges, adapters, and small models that cost dollars to
train. The ladder here is that economy taken to its logical end: you never
pay from-scratch cost at *any* scale, because every rung starts from a
working model and each leg only trains the delta the new capacity unlocks.
Growth is the compute cheat code; the engine below is the substrate that
makes it real instead of a diagram.

Performance is a first-class requirement of this engine, not a comment:
data generation is vectorized, the autograd graph is collected explicitly
per step so RSS stays flat, and every leg reports steps/s and peak RSS in
its receipt. If a leg gets slower, the receipt shows it.

Adapter families
----------------
- ``"mlp"`` -- generic dense stack (``MLPLeg``), Net2Net hidden widening.
  Covers context-MLP style checkpoints (HIM-native) and meaning models
  with dense stacks (auro_web), once their task loaders are wired.
- ``"auro-growth-transformer"`` -- the Auro 156K ladder; legs run in
  ``career.py`` (specialized operators), tracked here in the registry.
- ``"voice-complete"`` -- NumPy port of the pocket-voice-complete
  classifier; added once the inventory pins its exact architecture.

Data honesty: the built-in tasks are synthetic with learnable structure
(mechanics only). Real corpora plug in as ``task`` callables with the same
``(rng, batch) -> (X, y)`` signature; the receipt records the provenance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from auro_native_llm.growth.core import (
    Tensor,
    Adam,
    log_softmax,
    silu,
    _FLOAT,
    _acc,
    _unbroadcast,
)
from auro_native_llm.growth.receipts import (
    _canonical,
    _sha256_text,
    _peak_rss_mb,
    CLAIM_BOUNDARY,
    save_receipt,
)

RECEIPT_SCHEMA = "auro.engine.leg_receipt.v1"
REGISTRY_PATH = Path.home() / "workspace" / "model-registry.json"
PRESERVATION_TOL = 1e-4


# ----------------------------------------------------------------------------
# Tensor helpers (built on the growth core's autograd engine)
# ----------------------------------------------------------------------------

def relu(t: Tensor) -> Tensor:
    """ReLU via the core Tensor machinery (mask captured for backward)."""
    m = (t.data > 0).astype(_FLOAT)
    out = Tensor(t.data * m, _prev=(t,), _op="relu")

    def _backward():
        if out.grad is None:
            return
        if t.requires_grad:
            _acc(t, _unbroadcast(out.grad * m, t.shape))

    out._backward = _backward
    return out


def ce_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy with a correct backward pass.

    loss = -mean(log_softmax(logits)[range(n), targets])
    dL/dlogits = (softmax(logits) - onehot(targets)) / n
    """
    n = int(targets.shape[0])
    lp = log_softmax(logits, axis=-1)
    data = -float(np.mean(lp.data[np.arange(n), targets]))
    out = Tensor(np.asarray(data, dtype=_FLOAT), _prev=(lp,), _op="ce")

    def _backward():
        if out.grad is None:
            return
        g = float(out.grad) if np.ndim(out.grad) == 0 else float(np.asarray(out.grad).ravel()[0])
        # dL/d(lp) : -g/n at the true-class positions
        d_lp = np.zeros_like(lp.data)
        d_lp[np.arange(n), targets] = -g / n
        _acc(lp, d_lp)

    out._backward = _backward
    return out

# ----------------------------------------------------------------------------
# Generic Net2Net growth for dense stacks
# ----------------------------------------------------------------------------
# Convention (matches operators.py): x @ W with W: (in, out).
#
# Widening hidden dim h -> 2h between fc_i (in->h) and fc_{i+1} (h->out):
#   fc_i.W   (in, h) -> [W, W]      (duplicate cols)
#   fc_i.b   (h,)    -> [b, b]
#   fc_{i+1}.W (h, out) -> [W/2 ; W/2]  (duplicate rows, halved)
# Preservation: hidden' = [a(z); a(z)] with z = xW+b, so
#   out' = [h; h] @ [W/2; W/2] = hW/2 + hW/2 = hW = out,
# exactly, for ANY elementwise activation a. No approximations.

def net2net_widen_pair(
    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, factor: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Widen the hidden dim shared by (W1, b1) -> (W2) by ``factor``.

    Returns ``(W1_new, b1_new, W2_new)``; function-preserving up to float
    rounding. Only integer factor 2 is exact-symmetric; other factors tile
    and truncate, still exact because every new neuron copies an old one and
    the outgoing weights are split evenly across the copies.
    """
    if factor < 2:
        raise ValueError("widening factor must be >= 2")
    # tile each column-block `factor` times: [W1, W1, ...]
    W1_new = np.concatenate([W1] * factor, axis=1).astype(_FLOAT)
    b1_new = np.concatenate([b1] * factor, axis=0).astype(_FLOAT)
    # split the outgoing weight evenly: [W2/f; W2/f; ...]
    W2_new = np.concatenate([W2 / factor] * factor, axis=0).astype(_FLOAT)
    return W1_new, b1_new, W2_new


def mlp_param_count(layer_dims: List[int]) -> int:
    """Total params of a dense stack with dims [in, h0, ..., out] (biases incl)."""
    total = 0
    for i in range(len(layer_dims) - 1):
        total += layer_dims[i] * layer_dims[i + 1] + layer_dims[i + 1]
    return total


def plan_mlp_growth(
    layer_dims: List[int], target_ratio: float = 2.0
) -> Tuple[List[int], float]:
    """Choose hidden dims to widen so total params land nearest ``target_ratio``.

    Tries widening every non-empty subset of hidden layers (2x each) and
    returns ``(widen_flags, achieved_ratio)`` for the subset whose achieved
    ratio is closest to the target. ``widen_flags[i]`` corresponds to hidden
    layer ``i`` (there are ``len(layer_dims) - 2`` of them).
    """
    n_hidden = len(layer_dims) - 2
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer to grow")
    base = mlp_param_count(layer_dims)
    best: Optional[Tuple[List[int], float]] = None
    for mask in range(1, 1 << n_hidden):
        dims = list(layer_dims)
        for i in range(n_hidden):
            if mask & (1 << i):
                dims[i + 1] *= 2
        ratio = mlp_param_count(dims) / base
        err = abs(ratio - target_ratio)
        if best is None or err < best[1]:
            best = ([bool(mask & (1 << i)) for i in range(n_hidden)], err)
    flags, _ = best  # type: ignore[misc]
    grown_dims = list(layer_dims)
    for i, f in enumerate(flags):
        if f:
            grown_dims[i + 1] *= 2
    return flags, mlp_param_count(grown_dims) / base


def grow_mlp_params(
    params: Dict[str, np.ndarray],
    layer_dims: List[int],
    widen: List[bool],
    factor: int = 2,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Apply Net2Net widening per ``widen`` flags; returns (new_params, new_dims)."""
    new_dims = list(layer_dims)
    for i, f in enumerate(widen):
        if f:
            new_dims[i + 1] *= factor
    new: Dict[str, np.ndarray] = {}
    # copy untouched leading/trailing keys through first
    for k, v in params.items():
        new[k] = v
    n_layers = len(layer_dims) - 1
    for i in range(n_layers - 1):  # pairs (fc_i, fc_{i+1}); hidden dim = layer i+1
        if not widen[i]:
            continue
        W1 = params[f"fc{i}.W"]
        b1 = params[f"fc{i}.b"]
        W2 = params[f"fc{i + 1}.W"]
        W1n, b1n, W2n = net2net_widen_pair(W1, b1, W2, factor)
        new[f"fc{i}.W"], new[f"fc{i}.b"], new[f"fc{i + 1}.W"] = W1n, b1n, W2n
    return new, new_dims


# ----------------------------------------------------------------------------
# ModelAdapter: the interface every trainable model implements
# ----------------------------------------------------------------------------

# A task batch callable: (rng, batch) -> (X, y) with X float32 (B, in),
# y int64 (B,).
TaskFn = Callable[[np.random.Generator, int], Tuple[np.ndarray, np.ndarray]]


class ModelAdapter(ABC):
    """One model family on the ladder. Implementations own their params,
    forward pass, growth, and checkpoint format; the engine owns the leg
    loop, preservation verification, receipts, and the registry."""

    family: str = "base"

    @abstractmethod
    def param_count(self) -> int: ...

    @abstractmethod
    def config_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def forward_np(self, X: np.ndarray) -> np.ndarray:
        """Pure-NumPy forward (logits), no autograd -- used for probes."""

    @abstractmethod
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One optimizer step on (X, y); returns the loss."""

    @abstractmethod
    def grow(self, target_ratio: float = 2.0) -> Tuple["ModelAdapter", Dict[str, Any]]:
        """Return (grown_adapter, growth_plan_meta). The grown adapter must
        compute the same function as this one (verified by the engine)."""

    @abstractmethod
    def save(self, ckpt_dir: Path, meta: Dict[str, Any]) -> Dict[str, str]:
        """Write weights + config + meta; return {"sha256": ...} of weights."""

    @classmethod
    @abstractmethod
    def load(cls, ckpt_dir: Path, task: TaskFn, lr: float) -> "ModelAdapter":
        """Rebuild the adapter (params + task + optimizer) from a checkpoint."""

# ----------------------------------------------------------------------------
# Synthetic task (mechanics only): learnable structure, honest label
# ----------------------------------------------------------------------------

SYNTHETIC_PROVENANCE = "synthetic-classes-v1 (modular-hash features + planted partitions)"


def synthetic_classes_task(
    n_in: int, n_classes: int, seed: int = 0, noise: float = 0.05
) -> TaskFn:
    """Teacher-student classification with real learnable structure.

    Labels come from a fixed random teacher MLP (silu, 2 layers):
    ``y = argmax(teacher(X))`` with ``noise`` fraction of labels flipped.
    A student MLP of similar capacity can distill the teacher, so
    cross-entropy falls well below the chance floor ``ln(n_classes)``;
    a broken trainer cannot. Fully vectorized -- no Python loops over
    batch or features.
    """
    rng = np.random.default_rng(seed)
    h = 128
    tW1 = (rng.standard_normal((n_in, h)) * np.sqrt(2.0 / n_in)).astype(_FLOAT)
    tb1 = np.zeros(h, dtype=_FLOAT)
    tW2 = (rng.standard_normal((h, n_classes)) * np.sqrt(2.0 / h)).astype(_FLOAT)
    tb2 = np.zeros(n_classes, dtype=_FLOAT)

    def teacher_logits(X: np.ndarray) -> np.ndarray:
        h1 = X @ tW1 + tb1
        h1 = h1 / (1.0 + np.exp(-h1))  # silu
        return (h1 @ tW2 + tb2).astype(_FLOAT)

    def task(brng: np.random.Generator, batch: int) -> Tuple[np.ndarray, np.ndarray]:
        X = brng.uniform(-1.0, 1.0, size=(batch, n_in)).astype(_FLOAT)
        y = np.argmax(teacher_logits(X), axis=1).astype(np.int64)
        if noise > 0:
            flip = brng.random(batch) < noise
            y[flip] = brng.integers(0, n_classes, size=int(flip.sum()))
        return X, y

    return task


# ----------------------------------------------------------------------------
# MLPLeg: generic dense-stack adapter
# ----------------------------------------------------------------------------

class MLPLeg(ModelAdapter):
    """A dense stack ``[in, h0, ..., out]`` with silu hidden activations.

    Params: ``fc{i}.W`` (in, out), ``fc{i}.b`` (out,). Trained with the
    growth core's Tensor autograd + Adam. Growth = Net2Net widening of the
    hidden subset that lands closest to 2x total params.
    """

    family = "mlp"

    def __init__(
        self,
        layer_dims: List[int],
        task: TaskFn,
        lr: float = 1e-3,
        params: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 0,
    ):
        self.layer_dims = list(layer_dims)
        self.task = task
        self.lr = lr
        self.seed = seed
        if params is None:
            rng = np.random.default_rng(seed)
            params = {}
            for i in range(len(layer_dims) - 1):
                fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
                scale = np.sqrt(2.0 / fan_in)
                params[f"fc{i}.W"] = (rng.standard_normal((fan_in, fan_out)) * scale).astype(_FLOAT)
                params[f"fc{i}.b"] = np.zeros(fan_out, dtype=_FLOAT)
        self.params = {k: np.asarray(v, dtype=_FLOAT) for k, v in params.items()}
        self._opt: Optional[Adam] = None

    # -- interface ------------------------------------------------------
    def param_count(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def config_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "layer_dims": list(self.layer_dims),
            "activation": "silu",
            "lr": self.lr,
            "seed": self.seed,
        }

    def forward_np(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(_FLOAT)
        n_layers = len(self.layer_dims) - 1
        for i in range(n_layers):
            h = h @ self.params[f"fc{i}.W"] + self.params[f"fc{i}.b"]
            if i < n_layers - 1:
                h = _silu_np(h)
        return h

    def _forward_graph(self, X: np.ndarray) -> Tuple[Tensor, List[Tuple[str, Tensor]]]:
        t = Tensor(X.astype(_FLOAT))
        leaves: List[Tuple[str, Tensor]] = []
        n_layers = len(self.layer_dims) - 1
        for i in range(n_layers):
            W = Tensor(self.params[f"fc{i}.W"], requires_grad=True)
            b = Tensor(self.params[f"fc{i}.b"], requires_grad=True)
            leaves.append((f"fc{i}.W", W))
            leaves.append((f"fc{i}.b", b))
            t = t @ W + b
            if i < n_layers - 1:
                t = silu(t)
        return t, leaves

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        if self._opt is None:
            self._opt = Adam(self.params, lr=self.lr)
        logits, leaves = self._forward_graph(X)
        loss_t = ce_loss(logits, y)
        loss_t.backward()
        grads = {name: t.grad for name, t in leaves}
        if any(g is None for g in grads.values()):
            missing = [n for n, t in leaves if t.grad is None]
            raise RuntimeError(f"no grad for leaves: {missing}")
        self._opt.step(grads)
        loss_val = float(np.asarray(loss_t.data).ravel()[0])
        # The autograd graph holds reference cycles (Tensor <-> closures);
        # cyclic GC cannot keep up with one graph per step, so RSS would
        # grow without bound. Explicit collection keeps it flat -- this is
        # numerically a no-op (same fix as career.py).
        del logits, leaves, grads, loss_t
        gc.collect()
        return loss_val

    # -- growth ---------------------------------------------------------
    def grow(self, target_ratio: float = 2.0) -> Tuple["MLPLeg", Dict[str, Any]]:
        widen, achieved = plan_mlp_growth(self.layer_dims, target_ratio)
        new_params, new_dims = grow_mlp_params(self.params, self.layer_dims, widen)
        grown = MLPLeg(new_dims, task=self.task, lr=self.lr, params=new_params,
                       seed=self.seed)
        plan = {
            "operator": "net2net-widen-mlp",
            "widen_hidden": widen,
            "target_ratio": target_ratio,
            "achieved_ratio": achieved,
            "from_dims": list(self.layer_dims),
            "to_dims": new_dims,
        }
        return grown, plan

    # -- checkpoints ----------------------------------------------------
    def save(self, ckpt_dir: Path, meta: Dict[str, Any]) -> Dict[str, str]:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "config.json").write_text(
            json.dumps(self.config_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (ckpt_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        weights_path = ckpt_dir / "weights.npz"
        np.savez(weights_path, **self.params)
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        (ckpt_dir / "SHA256SUM").write_text(f"{sha}  weights.npz\n", encoding="utf-8")
        return {"sha256": sha}

    @classmethod
    def load(cls, ckpt_dir: Path, task: TaskFn, lr: float) -> "MLPLeg":
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
        if cfg.get("family") != "mlp":
            raise ValueError(f"{ckpt_dir}: not an mlp leg checkpoint")
        z = np.load(ckpt_dir / "weights.npz")
        params = {k: np.asarray(z[k], dtype=_FLOAT) for k in z.files}
        leg = cls(cfg["layer_dims"], task=task, lr=lr, params=params,
                  seed=int(cfg.get("seed", 0)))
        return leg


def _silu_np(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-x))).astype(_FLOAT)

# ----------------------------------------------------------------------------
# Leg receipts
# ----------------------------------------------------------------------------

def finalize_leg_receipt(
    *,
    leg: int,
    model_id: str,
    family: str,
    config: Dict[str, Any],
    params: int,
    parent: Optional[Dict[str, Any]],
    growth_plan: Optional[Dict[str, Any]],
    preservation_max_abs_diff: Optional[float],
    steps: int,
    units_seen: int,
    unit: str,
    loss_first: float,
    loss_last: float,
    loss_curve: List[float],
    elapsed_s: float,
    steps_per_s: float,
    weights_sha256: str,
    data_provenance: str,
) -> Dict[str, Any]:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "kind": "train-leg",
        "leg": int(leg),
        "model_id": model_id,
        "family": family,
        "config": config,
        "config_sha256": _sha256_text(_canonical(config)),
        "params": int(params),
        "parent": parent,
        "growth_plan": growth_plan,
        "preservation_max_abs_diff": preservation_max_abs_diff,
        "steps": int(steps),
        "units_seen": int(units_seen),
        "unit": unit,
        "loss_first": float(loss_first),
        "loss_last": float(loss_last),
        "loss_decreased": bool(loss_last < loss_first),
        "loss_curve": [float(x) for x in loss_curve],
        "elapsed_s": round(float(elapsed_s), 3),
        "steps_per_s": round(float(steps_per_s), 3),
        "peak_rss_mb": _peak_rss_mb(),
        "weights_sha256": weights_sha256,
        "data_provenance": data_provenance,
        "claim_boundary": CLAIM_BOUNDARY,
        "finished_unix": int(time.time()),
    }
    receipt["receipt_sha256"] = _sha256_text(_canonical(receipt))
    return receipt


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(reg: Dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")


def latest_leg_dir(ckpt_root: Path, prefix: str = "leg-") -> Optional[Tuple[int, Path]]:
    """Highest-numbered completed leg dir (weights.npz + receipt.json present)."""
    best: Optional[Tuple[int, Path]] = None
    if not ckpt_root.exists():
        return None
    for child in ckpt_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        try:
            k = int(child.name[len(prefix):])
        except ValueError:
            continue
        if (child / "weights.npz").exists() and (child / "receipt.json").exists():
            if best is None or k > best[0]:
                best = (k, child)
    return best


# ----------------------------------------------------------------------------
# The leg loop: train -> checkpoint -> grow -> compound
# ----------------------------------------------------------------------------

def run_legs(
    leg: ModelAdapter,
    *,
    model_id: str,
    num_legs: int = 1,
    steps_per_leg: int = 200,
    batch: int = 64,
    seed: int = 0,
    ckpt_root: Path,
    data_provenance: str = SYNTHETIC_PROVENANCE,
    unit: str = "examples",
    resume: bool = True,
    growth_target_ratio: float = 2.0,
    log_every: int = 50,
) -> List[Dict[str, Any]]:
    """Run the compounding leg loop for one model.

    With ``resume=True`` (default), the latest completed leg under
    ``ckpt_root`` is loaded and the loop continues from there -- the new
    legs compound on everything before them. Leg numbering continues;
    each receipt chains to its parent.
    """
    ckpt_root = Path(ckpt_root)
    rng = np.random.default_rng(seed)
    summaries: List[Dict[str, Any]] = []

    if resume:
        found = latest_leg_dir(ckpt_root)
        if found is not None:
            k, ckpt_dir = found
            leg = type(leg).load(ckpt_dir, task=leg.task, lr=leg.lr)
            n_k = leg.param_count()
            print(f"[resume] loaded leg-{k} ({model_id}, {n_k:,} params) from {ckpt_dir}",
                  flush=True)
            # Grow FIRST so the resumed run continues the ladder identically
            # to an uninterrupted one: leg k+1 trains the grown model.
            grown, growth_plan, pres_diff = _grow_verified(leg, rng, growth_target_ratio)
            parent_meta: Optional[Dict[str, Any]] = {
                "leg": k, "model_id": model_id, "params": n_k, "plan": growth_plan,
            }
            leg = grown
            start_leg = k + 1
        else:
            print("[resume] no completed legs found; starting fresh", flush=True)
            parent_meta, growth_plan, pres_diff = None, None, None
            start_leg = 0
    else:
        parent_meta, growth_plan, pres_diff = None, None, None
        start_leg = 0

    for leg_n in range(start_leg, start_leg + num_legs):
        n_params = leg.param_count()
        print(f"[leg {leg_n}] {model_id} ({leg.family}): {n_params:,} params, "
              f"dims={getattr(leg, 'layer_dims', '?')}", flush=True)
        t0 = time.time()
        curve: List[float] = []
        for step in range(steps_per_leg):
            X, y = leg.task(rng, batch)
            loss = leg.train_step(X, y)
            if step % log_every == 0 or step == steps_per_leg - 1:
                curve.append(loss)
        elapsed = time.time() - t0
        sps = steps_per_leg / elapsed if elapsed > 0 else 0.0
        units_seen = steps_per_leg * batch
        print(f"[leg {leg_n}] loss {curve[0]:.4f} -> {curve[-1]:.4f} "
              f"({steps_per_leg} steps, {elapsed:.1f}s, {sps:.1f} steps/s)",
              flush=True)

        ckpt_dir = ckpt_root / f"leg-{leg_n}"
        leg_meta = {
            "model_id": model_id,
            "family": leg.family,
            "leg": leg_n,
            "params": n_params,
            "parent": parent_meta,
            "growth_plan": growth_plan,
            "steps": steps_per_leg,
            "data_provenance": data_provenance,
            "claim": "train-leg checkpoint on synthetic data; not a capability claim",
        }
        saved = leg.save(ckpt_dir, leg_meta)
        receipt = finalize_leg_receipt(
            leg=leg_n, model_id=model_id, family=leg.family,
            config=leg.config_dict(), params=n_params, parent=parent_meta,
            growth_plan=growth_plan, preservation_max_abs_diff=pres_diff,
            steps=steps_per_leg, units_seen=units_seen, unit=unit,
            loss_first=curve[0], loss_last=curve[-1], loss_curve=curve,
            elapsed_s=elapsed, steps_per_s=sps, weights_sha256=saved["sha256"],
            data_provenance=data_provenance,
        )
        save_receipt(receipt, ckpt_dir / "receipt.json")
        print(f"[leg {leg_n}] checkpointed -> {ckpt_dir} "
              f"(receipt {receipt['receipt_sha256'][:12]}...)", flush=True)
        summaries.append({"leg": leg_n, "model_id": model_id,
                          "params": n_params,
                          "loss_first": curve[0], "loss_last": curve[-1],
                          "steps_per_s": sps, "ckpt": str(ckpt_dir)})

        if leg_n < start_leg + num_legs - 1:
            grown, growth_plan, pres_diff = _grow_verified(leg, rng, growth_target_ratio)
            parent_meta = {"leg": leg_n, "model_id": model_id,
                           "params": n_params, "plan": growth_plan}
            leg = grown
    return summaries


def _grow_verified(
    leg: ModelAdapter, rng: np.random.Generator, target_ratio: float
) -> Tuple[ModelAdapter, Dict[str, Any], float]:
    """Grow the adapter and numerically verify function preservation."""
    n_params = leg.param_count()
    target = int(n_params * target_ratio)
    probe_X, _ = leg.task(rng, 16)
    before = leg.forward_np(probe_X)
    grown, plan = leg.grow(target_ratio)
    achieved = grown.param_count()
    after = grown.forward_np(probe_X)
    pres_diff = float(np.max(np.abs(before - after)))
    print(f"[grow] {leg.family}: plan={plan.get('operator')} target={target:,} "
          f"achieved={achieved:,} (ratio {achieved / n_params:.2f}); "
          f"preservation max|diff| = {pres_diff:.3e}", flush=True)
    assert pres_diff < PRESERVATION_TOL, \
        f"growth broke function preservation: {pres_diff}"
    return grown, plan, pres_diff


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Universal train-leg engine (synthetic data).")
    ap.add_argument("--model-id", type=str, default="mlp-demo")
    ap.add_argument("--dims", type=str, default="32,64,64,8",
                    help="comma-separated layer dims for a fresh mlp leg")
    ap.add_argument("--legs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="leg_checkpoints")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    task = synthetic_classes_task(n_in=dims[0], n_classes=dims[-1], seed=args.seed)
    leg = MLPLeg(dims, task=task, lr=args.lr, seed=args.seed)
    print(f"leg-0 params: {leg.param_count():,}")
    summaries = run_legs(
        leg, model_id=args.model_id, num_legs=args.legs,
        steps_per_leg=args.steps, batch=args.batch, seed=args.seed,
        ckpt_root=Path(args.out), resume=not args.no_resume,
    )
    print("\nlegs complete:")
    for s in summaries:
        print(f"  leg {s['leg']} {s['model_id']}: {s['params']:,} params, "
              f"loss {s['loss_first']:.4f} -> {s['loss_last']:.4f}, "
              f"{s['steps_per_s']:.1f} steps/s")


if __name__ == "__main__":
    main()
