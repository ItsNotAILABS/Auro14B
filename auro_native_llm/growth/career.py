"""Career runner: train -> checkpoint-as-new-model -> grow -> continue.

Each rung trains on synthetic data (mechanics verification only), then the
rung is checkpointed as a *new model* (weights + config + rung metadata +
signed receipt) under ``checkpoints/rung-N/``. Growth to the next rung uses
the function-preserving operators; preservation is re-verified numerically
at every growth step and recorded in the receipt.

Nothing here trains on real data. See README.md.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from auro_native_llm.growth.core import (
    Adam,
    GrowthConfig,
    GrowthTransformer,
    _param_shapes,
    causal_lm_loss,
    count_params,
)
from auro_native_llm.growth.schedule import (
    apply_plan,
    default_criteria,
    plan_growth,
)
from auro_native_llm.growth.receipts import finalize_rung_receipt, save_receipt

DATA_PROVENANCE = "synthetic-structured-v1 (modular-arithmetic chain + copy motifs + noise)"


# ----------------------------------------------------------------------------
# Synthetic data (mechanics only -- learnable structure, no real text)
# ----------------------------------------------------------------------------

def synthetic_batch(rng: np.random.Generator, batch: int, seq: int, vocab: int,
                    arith_a: int = 37, arith_b: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """(inputs, targets) with learnable structure.

    Each position is, with probability 0.55, the modular-arithmetic successor
    of the previous token; with 0.15 a copy of the token 3 steps back
    (induction-like motif); else uniform noise. Initial cross-entropy is
    ``ln(vocab)``; a learning model drives it down via the planted structure.
    """
    toks = np.empty((batch, seq + 1), dtype=np.int64)
    for b in range(batch):
        t = int(rng.integers(0, vocab))
        for i in range(seq + 1):
            toks[b, i] = t
            r = rng.random()
            if r < 0.55:
                t = (arith_a * t + arith_b) % vocab
            elif r < 0.70 and i >= 3:
                t = int(toks[b, i - 3])
            else:
                t = int(rng.integers(0, vocab))
    return toks[:, :-1], toks[:, 1:]


# ----------------------------------------------------------------------------
# Checkpointing: every rung is saved AS A NEW MODEL
# ----------------------------------------------------------------------------

def _weights_sha256(params: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(np.ascontiguousarray(params[k]).tobytes())
    return h.hexdigest()


def save_rung(model: GrowthTransformer, ckpt_dir: Path, rung_meta: Dict) -> Dict[str, str]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wpath = ckpt_dir / "weights.npz"
    np.savez(wpath, **{k: v for k, v in model.params.items()})
    (ckpt_dir / "config.json").write_text(
        json.dumps(model.cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (ckpt_dir / "rung.json").write_text(
        json.dumps(rung_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"weights": str(wpath), "sha256": _weights_sha256(model.params)}


def load_rung(ckpt_dir: Path) -> GrowthTransformer:
    cfg = GrowthConfig.from_dict(
        json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    )
    z = np.load(ckpt_dir / "weights.npz")
    params = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    return GrowthTransformer(cfg, params)


def latest_completed_rung(out_dir: Path) -> Optional[Tuple[int, Path]]:
    """Highest rung-N in ``out_dir`` with weights + config + receipt.

    Only rungs with a signed receipt count as completed: a rung whose
    training was interrupted mid-way must not be resumed from.
    """
    out_dir = Path(out_dir)
    best: Optional[Tuple[int, Path]] = None
    if not out_dir.is_dir():
        return None
    for child in out_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("rung-"):
            continue
        try:
            n = int(child.name.split("-", 1)[1])
        except ValueError:
            continue
        if ((child / "weights.npz").is_file()
                and (child / "config.json").is_file()
                and (child / "receipt.json").is_file()):
            if best is None or n > best[0]:
                best = (n, child)
    return best


# ----------------------------------------------------------------------------
# Auro-156K checkpoint as rung-0 initialization
# ----------------------------------------------------------------------------
#
# Honest status (2026-09-10): no trained Auro-156K checkpoint ships with
# this repo. ``native_llm/configs/family/auro_156k.json`` is an
# *architecture target* (status: "architecture-target-not-trained-checkpoint"),
# and the only real trained weights in the repo are HIM-native-v0 (~147K,
# a different model). The plumbing below discovers a real 156K checkpoint
# when one is dropped in, and otherwise falls back to random init with a
# clear log line -- it never fabricates weights.

class CheckpointError(Exception):
    """A candidate Auro-156K checkpoint exists but cannot be used."""


CHECKPOINT_ENV_VAR = "AURO_156K_CHECKPOINT_DIR"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def candidate_156k_dirs() -> List[Path]:
    """Where a real Auro-156K checkpoint is looked for, in order."""
    dirs: List[Path] = []
    env = os.environ.get(CHECKPOINT_ENV_VAR)
    if env:
        dirs.append(Path(env))
    dirs.append(_repo_root() / "checkpoints" / "auro-156k")
    dirs.append(Path(__file__).resolve().parent / "checkpoints" / "auro-156k")
    return dirs


def find_156k_checkpoint() -> Optional[Path]:
    """Return the first candidate dir holding weights.npz + config.json."""
    for d in candidate_156k_dirs():
        if (d / "weights.npz").is_file() and (d / "config.json").is_file():
            return d
    return None


def _growth_config_from_repo_schema(d: Dict) -> GrowthConfig:
    """Best-effort map of the native_llm model_config schema -> GrowthConfig.

    Assumptions (documented, not silent): head_dim = hidden // heads;
    MoE layers from ``moe_every`` (every-other-layer -> odd layers);
    ``trained_seq_len`` must be present in the config because RoPE needs a
    max sequence length and the architecture target does not declare one.
    """
    arch = d.get("architecture", {})
    layers = int(arch["layers"])
    heads = int(arch["attention_heads"])
    hidden = int(arch["hidden_size"])
    moe_every = int(arch.get("moe_every") or 0)
    moe_layers = tuple(
        l for l in range(layers) if moe_every and l % moe_every == moe_every - 1
    )
    seq = d.get("trained_seq_len") or arch.get("trained_seq_len")
    if seq is None:
        raise CheckpointError(
            "repo-schema config has no 'trained_seq_len'; add it so the "
            "growth core knows the RoPE cache size."
        )
    return GrowthConfig(
        vocab=int(arch["vocab_size_target"]),
        d=hidden,
        layers=layers,
        heads=heads,
        kv_heads=int(arch["kv_heads"]),
        head_dim=hidden // heads,
        ffn=int(arch["intermediate_size"]),
        seq=int(seq),
        moe_layers=moe_layers,
        experts=int(arch.get("experts", 2)),
        seed=0,
    )


def load_156k_checkpoint(ckpt_dir: Path) -> GrowthTransformer:
    """Load a real Auro-156K checkpoint into a GrowthTransformer.

    Expected layout (mirrors :func:`save_rung`): ``weights.npz`` with
    growth-core key names plus ``config.json`` in either the growth-core
    schema or the repo's native_llm ``model_config`` schema. Every key and
    shape is validated against the config; anything missing or mismatched
    raises :class:`CheckpointError` (the caller falls back to random init).
    """
    ckpt_dir = Path(ckpt_dir)
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    schema = str(cfg_dict.get("schema", ""))
    if schema.startswith("auro.native_llm.model_config"):
        cfg = _growth_config_from_repo_schema(cfg_dict)
    else:
        cfg = GrowthConfig.from_dict(cfg_dict)
    z = np.load(ckpt_dir / "weights.npz")
    expected = _param_shapes(cfg)
    params: Dict[str, np.ndarray] = {}
    missing = [k for k in expected if k not in z.files]
    mismatched = [
        (k, tuple(np.shape(z[k])), shape)
        for k, shape in expected.items()
        if k in z.files and tuple(np.shape(z[k])) != tuple(shape)
    ]
    if missing or mismatched:
        raise CheckpointError(
            f"{ckpt_dir}: {len(missing)} missing keys, {len(mismatched)} shape "
            f"mismatches (e.g. {missing[:3]!r} {mismatched[:3]!r})"
        )
    for k in expected:
        params[k] = np.asarray(z[k], dtype=np.float32)
    return GrowthTransformer(cfg, params)


def _init_rung_zero(p0_cfg: GrowthConfig, init_from_156k: bool) -> GrowthTransformer:
    """Rung-0 init: real 156K checkpoint when present, else random init."""
    if init_from_156k:
        ckpt = find_156k_checkpoint()
        if ckpt is not None:
            try:
                model = load_156k_checkpoint(ckpt)
                print(f"[rung 0] initialized from Auro-156K checkpoint {ckpt} "
                      f"({model.param_count():,} params)")
                return model
            except CheckpointError as e:
                print(f"[rung 0] WARNING: 156K checkpoint at {ckpt} unusable "
                      f"({e}); falling back to random init")
    checked = ", ".join(str(d) for d in candidate_156k_dirs())
    print(f"[rung 0] no Auro-156K checkpoint found (checked: {checked}); "
          f"random init -- mechanics mode")
    return GrowthTransformer(p0_cfg)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_rung(model: GrowthTransformer, rng: np.random.Generator, *,
               steps: int, batch: int, lr: float,
               log_every: int = 50) -> List[float]:
    """Train one rung; returns the logged loss curve (every ``log_every`` steps)."""
    opt = Adam(model.params, lr=lr)
    curve: List[float] = []
    for step in range(steps):
        inputs, targets = synthetic_batch(rng, batch, model.cfg.seq, model.cfg.vocab)
        loss, grads = model.forward_train(inputs, targets)
        opt.step(grads)
        # The autograd graph is a web of reference cycles (Tensor <->
        # _backward closures). It is all collectable, but CPython's
        # automatic cyclic GC does not keep up with one ~0.5GB graph per
        # step at larger rungs, so RSS grows without bound until the OOM
        # killer intervenes (observed: 118MB -> ~3GB in 10 steps at rung 2).
        # An explicit collection per step keeps RSS flat; it is
        # numerically a no-op.
        del grads
        gc.collect()
        if step % log_every == 0 or step == steps - 1:
            curve.append(loss)
    return curve


def eval_loss(model: GrowthTransformer, rng: np.random.Generator, batches: int = 4,
              batch: int = 32) -> float:
    losses = []
    for _ in range(batches):
        inputs, targets = synthetic_batch(rng, batch, model.cfg.seq, model.cfg.vocab)
        logits = model.forward(inputs)
        losses.append(float(causal_lm_loss(logits, targets).data))
    return float(np.mean(losses))


# ----------------------------------------------------------------------------
# Career orchestration
# ----------------------------------------------------------------------------

def model_id_for(rung: int) -> str:
    # Honest id: a growth-rung checkpoint, not a family trained checkpoint.
    return f"Auro-156K-growth-r{rung}"


def _grow_one_step(model: GrowthTransformer, rng: np.random.Generator):
    """Plan and apply one growth step, verifying function preservation.

    Returns ``(grown_model, plan_name, achieved_params, max_abs_diff)``.
    """
    n_params = count_params(model.cfg)
    target = n_params * 2
    plan_name, new_cfg, achieved = plan_growth(model.cfg, target)
    print(f"[grow] plan={plan_name} target={target:,} achieved={achieved:,} "
          f"(ratio {achieved / target:.2f})", flush=True)
    # preservation probe on a fixed batch BEFORE growth
    probe_inputs, _ = synthetic_batch(rng, 8, model.cfg.seq, model.cfg.vocab)
    before = model.forward(probe_inputs).data
    new_params, new_cfg = apply_plan(model.params, model.cfg, plan_name)
    assert sum(v.size for v in new_params.values()) == count_params(new_cfg)
    grown = GrowthTransformer(new_cfg, new_params)
    after = grown.forward(probe_inputs).data
    pres_diff = float(np.max(np.abs(before - after)))
    print(f"[grow] preservation max|diff| = {pres_diff:.3e}", flush=True)
    assert pres_diff < 1e-4, f"growth broke function preservation: {pres_diff}"
    return grown, plan_name, achieved, pres_diff


def run_career(p0_cfg: GrowthConfig, *, num_growths: int = 2, steps_per_rung: int = 250,
               batch: int = 32, lr: float = 3e-4, seed: int = 0,
               out_dir: Path = Path("checkpoints"),
               resume: bool = False, init_from_156k: bool = True) -> List[Dict]:
    """Run the career ladder; returns per-rung summaries.

    With ``resume=True``, the latest *completed* rung in ``out_dir``
    (weights + config + signed receipt) is loaded and the ladder continues
    from there: ``num_growths`` counts *additional* growths, rung numbering
    continues, and each new rung's receipt chains back to the resumed rung.
    Runs compound instead of restarting. With no completed rung found,
    resume falls back to a fresh start (logged).
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)

    resumed_from: Optional[int] = None
    if resume:
        found = latest_completed_rung(out_dir)
        if found is None:
            print("[resume] no completed rung checkpoints found; starting fresh")
        else:
            k, ckpt_dir = found
            resumed_from = k
            model = load_rung(ckpt_dir)
            n_k = count_params(model.cfg)
            print(f"[resume] loaded rung-{k} ({model_id_for(k)}, {n_k:,} params) "
                  f"from {ckpt_dir}")
            # Grow FIRST, exactly as the loop's growth block would: the
            # resumed run continues the ladder identically to an
            # uninterrupted one -- rung k+1 is the grown model, trained next.
            model, growth_plan, _achieved, pres_diff = _grow_one_step(model, rng)
            parent_meta: Optional[Dict] = {
                "rung": k,
                "model_id": model_id_for(k),
                "params": n_k,
                "plan": growth_plan,
            }
            start_rung = k + 1
            print(f"[resume] continuing at rung-{start_rung} "
                  f"({count_params(model.cfg):,} params)")

    if resumed_from is None:
        model = _init_rung_zero(p0_cfg, init_from_156k)
        start_rung = 0
        parent_meta = None
        growth_plan = None
        pres_diff = None

    summaries: List[Dict] = []
    for rung in range(start_rung, start_rung + num_growths + 1):
        mid = model_id_for(rung)
        n_params = count_params(model.cfg)
        crit = default_criteria(rung, model.cfg)
        steps = min(steps_per_rung, crit.max_steps)
        print(f"[rung {rung}] {mid}: {n_params:,} params, cfg={model.cfg.to_dict()}",
              flush=True)
        t0 = time.time()
        curve = train_rung(model, rng, steps=steps, batch=batch, lr=lr)
        elapsed = time.time() - t0
        tokens_seen = steps * batch * model.cfg.seq
        print(f"[rung {rung}] loss {curve[0]:.4f} -> {curve[-1]:.4f} "
              f"({steps} steps, {elapsed:.1f}s)", flush=True)

        ckpt_dir = out_dir / f"rung-{rung}"
        rung_meta = {
            "model_id": mid,
            "rung": rung,
            "params": n_params,
            "parent": parent_meta,
            "growth_plan": growth_plan,
            "preservation_max_abs_diff": pres_diff,
            "steps": steps,
            "tokens_seen": tokens_seen,
            "data_provenance": DATA_PROVENANCE,
            "claim": "growth-mechanics checkpoint on synthetic data; not a capability claim",
        }
        saved = save_rung(model, ckpt_dir, rung_meta)
        receipt = finalize_rung_receipt(
            rung=rung, model_id=mid, config=model.cfg.to_dict(), params=n_params,
            parent=parent_meta, growth_plan=growth_plan,
            preservation_max_abs_diff=pres_diff, steps=steps, tokens_seen=tokens_seen,
            loss_first=curve[0], loss_last=curve[-1], loss_curve=curve,
            elapsed_s=elapsed, weights_sha256=saved["sha256"],
            data_provenance=DATA_PROVENANCE,
        )
        save_receipt(receipt, ckpt_dir / "receipt.json")
        print(f"[rung {rung}] checkpointed -> {ckpt_dir} (receipt {receipt['receipt_sha256'][:12]}...)",
              flush=True)
        summaries.append({"rung": rung, "model_id": mid, "params": n_params,
                          "loss_first": curve[0], "loss_last": curve[-1],
                          "ckpt": str(ckpt_dir)})

        if rung < start_rung + num_growths:
            # --- grow to the next rung ---
            model, growth_plan, _achieved, pres_diff = _grow_one_step(model, rng)
            parent_meta = {"rung": rung, "model_id": mid, "params": n_params,
                           "plan": growth_plan}
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="Auro exponential-growth career runner (synthetic data).")
    ap.add_argument("--growths", type=int, default=2)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest completed rung in --out "
                         "instead of restarting; --growths then counts "
                         "additional growths and each run compounds")
    ap.add_argument("--no-156k-init", action="store_true",
                    help="skip Auro-156K checkpoint discovery; rung 0 always "
                         "starts from random init")
    args = ap.parse_args()

    p0 = GrowthConfig(vocab=args.vocab, d=64, layers=2, heads=2, kv_heads=2,
                      head_dim=32, ffn=128, seq=64, seed=args.seed)
    print(f"rung-0 params: {count_params(p0):,}")
    summaries = run_career(p0, num_growths=args.growths, steps_per_rung=args.steps,
                           batch=args.batch, lr=args.lr, seed=args.seed,
                           out_dir=Path(args.out), resume=args.resume,
                           init_from_156k=not args.no_156k_init)
    print("\ncareer complete:")
    for s in summaries:
        print(f"  rung {s['rung']} {s['model_id']}: {s['params']:,} params, "
              f"loss {s['loss_first']:.4f} -> {s['loss_last']:.4f}")


if __name__ == "__main__":
    main()
