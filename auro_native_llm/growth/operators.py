"""Net2Net-style function-preserving growth operators for the Auro core.

Each operator maps ``(params, cfg) -> (new_params, new_cfg)`` such that the
grown model computes *exactly* the same function as the source model at
growth time (up to float rounding). Training then continues from a
strictly more expressive initialization.

Operators
---------
- :func:`grow_width`  -- double attention heads (GQA-aware), QK-norm gains,
  and the SwiGLU intermediate dim. Model dim ``d`` is fixed, so norms and
  embeddings are untouched.
- :func:`grow_depth`  -- double the layer count by appending copies of each
  block whose residual branches (W_o, W_d / expert W_d) are zero-initialized,
  making every new block an identity at growth time.
- :func:`grow_moe`    -- introduce MoE on dense layers (2 experts cloned from
  the FFN, router zero-init -> uniform routing == original FFN) or duplicate
  the experts of an existing MoE layer (router columns duplicated -> routing
  probabilities halve -> identical mixture).

The preservation proofs are in the docstrings; ``tests/test_auro_growth.py``
asserts them numerically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from auro_native_llm.growth.core import GrowthConfig, _FLOAT


def _dup_rows(w: np.ndarray) -> np.ndarray:
    """Duplicate each head-block of rows: [W; W] (axis=0)."""
    return np.concatenate([w, w], axis=0)


def _dup_cols(w: np.ndarray) -> np.ndarray:
    """Duplicate each head-block of cols: [W, W] (axis=1)."""
    return np.concatenate([w, w], axis=1)


def _dup_cols_half(w: np.ndarray) -> np.ndarray:
    """[W/2, W/2] along cols."""
    return np.concatenate([w / 2, w / 2], axis=1)


def _dup_rows_half(w: np.ndarray) -> np.ndarray:
    """[W/2; W/2] along rows."""
    return np.concatenate([w / 2, w / 2], axis=0)


# ----------------------------------------------------------------------------
# Pure config transforms (no weights needed -- used by the scheduler)
# ----------------------------------------------------------------------------

def width_config(cfg: GrowthConfig) -> GrowthConfig:
    return GrowthConfig(
        **{**cfg.to_dict(), "heads": cfg.heads * 2, "kv_heads": cfg.kv_heads * 2,
           "ffn": cfg.ffn * 2, "moe_layers": cfg.moe_layers}
    )


def depth_config(cfg: GrowthConfig) -> GrowthConfig:
    return GrowthConfig(
        **{**cfg.to_dict(), "layers": cfg.layers * 2,
           "moe_layers": tuple(sorted(set(cfg.moe_layers) | {cfg.layers + l for l in cfg.moe_layers}))}
    )


def moe_config(cfg: GrowthConfig, layers: Optional[List[int]] = None) -> GrowthConfig:
    targets = list(layers) if layers is not None else [l for l in range(cfg.layers) if l not in cfg.moe_layers]
    new_moe = set(cfg.moe_layers)
    new_experts = cfg.experts
    for l in targets:
        if l in cfg.moe_layers:
            new_experts = max(new_experts, 2 * cfg.experts)
        else:
            new_moe.add(l)
            new_experts = max(new_experts, 2)
    return GrowthConfig(
        **{**cfg.to_dict(), "moe_layers": tuple(sorted(new_moe)), "experts": new_experts}
    )


# ----------------------------------------------------------------------------
# Width growth
# ----------------------------------------------------------------------------

def grow_width(params: Dict[str, np.ndarray], cfg: GrowthConfig) -> Tuple[Dict[str, np.ndarray], GrowthConfig]:
    """Double heads, KV heads, and FFN width at fixed model dim.

    Preservation argument: new head ``h+i`` is an exact copy of head ``i``
    (identical q/k/v projections, QK-norm gains, RoPE geometry), so its
    attention output equals head ``i``'s. The concatenated output ``[O; O]``
    then passes through ``Wo' = [Wo/2; Wo/2]``, giving ``O@Wo`` exactly.
    Likewise each SwiGLU neuron is split in two with halved down-projection:
    ``silu(x@[Wg;Wg]) * (x@[Wu;Wu]) @ [Wd/2; Wd/2] == silu(x@Wg)*(x@Wu)@Wd``.
    """
    new = dict(params)
    for l in range(cfg.layers):
        # attention: duplicate per-head row blocks (q/k/v), duplicate QK-norm
        # gains, halve the output projection across the duplicated concat.
        new[f"b{l}.wq"] = _dup_cols(params[f"b{l}.wq"])
        new[f"b{l}.wk"] = _dup_cols(params[f"b{l}.wk"])
        new[f"b{l}.wv"] = _dup_cols(params[f"b{l}.wv"])
        new[f"b{l}.wo"] = _dup_rows_half(params[f"b{l}.wo"])
        new[f"b{l}.qnw"] = _dup_rows(params[f"b{l}.qnw"])
        new[f"b{l}.knw"] = _dup_rows(params[f"b{l}.knw"])
        # SwiGLU: neuron splitting
        if l in cfg.moe_layers:
            for e in range(cfg.experts):
                new[f"b{l}.e{e}.g"] = _dup_cols(params[f"b{l}.e{e}.g"])
                new[f"b{l}.e{e}.u"] = _dup_cols(params[f"b{l}.e{e}.u"])
                new[f"b{l}.e{e}.d"] = _dup_rows_half(params[f"b{l}.e{e}.d"])
        else:
            new[f"b{l}.fg"] = _dup_cols(params[f"b{l}.fg"])
            new[f"b{l}.fu"] = _dup_cols(params[f"b{l}.fu"])
            new[f"b{l}.fd"] = _dup_rows_half(params[f"b{l}.fd"])
    new_cfg = width_config(cfg)
    return new, new_cfg


# ----------------------------------------------------------------------------
# Depth growth
# ----------------------------------------------------------------------------

def _zero_residual_block(params: Dict[str, np.ndarray], cfg: GrowthConfig, layer: int) -> Dict[str, np.ndarray]:
    """Copy of ``layer``'s params with residual branches zeroed (identity block)."""
    out: Dict[str, np.ndarray] = {}
    for k, v in params.items():
        if not k.startswith(f"b{layer}."):
            continue
        nk = k.replace(f"b{layer}.", f"b{cfg.layers + layer}.", 1)
        if k.endswith(".wo") or k.endswith(".fd") or ".e" in k and k.endswith(".d"):
            out[nk] = np.zeros_like(v)
        else:
            out[nk] = v.copy()
    return out


def grow_depth(params: Dict[str, np.ndarray], cfg: GrowthConfig) -> Tuple[Dict[str, np.ndarray], GrowthConfig]:
    """Double the layer count by stacking identity-at-birth copies.

    Preservation argument: block ``l`` computes ``x + attn'(x) + ffn'(x)``.
    The appended copy zeroes ``Wo`` and ``Wd`` (and expert ``Wd``), so
    ``attn'(x) == 0`` and ``ffn'(x) == 0`` regardless of the other weights:
    the new block is the identity, and the grown stack computes the same
    function as the original.
    """
    new = dict(params)
    for l in range(cfg.layers):
        new.update(_zero_residual_block(params, cfg, l))
    new_cfg = depth_config(cfg)
    return new, new_cfg


# ----------------------------------------------------------------------------
# MoE growth
# ----------------------------------------------------------------------------

def grow_moe(params: Dict[str, np.ndarray], cfg: GrowthConfig,
             layers: Optional[List[int]] = None) -> Tuple[Dict[str, np.ndarray], GrowthConfig]:
    """Introduce MoE on dense layers, or duplicate experts on MoE layers.

    Dense -> MoE (2 experts): each expert is a clone of the layer's FFN and
    the router is zero-initialized, so routing is uniform and the mixture
    equals the original FFN output exactly. (If the config already carries
    E > 2 experts on other layers, the new layer gets E clones instead, so
    the per-layer expert count always matches ``cfg.experts``.)

    MoE E -> 2E: experts are duplicated and router *columns* are duplicated
    (router is (d, E) -> (d, 2E) == [R, R]), so each duplicated pair
    receives half the original probability mass:
    ``sum_i (p_i/2 * e_i + p_i/2 * e_i) == sum_i p_i * e_i``.
    """
    targets = list(layers) if layers is not None else [l for l in range(cfg.layers) if l not in cfg.moe_layers]
    new = dict(params)
    for l in targets:
        if l in cfg.moe_layers:
            e = cfg.experts
            for i in range(e):
                new[f"b{l}.e{e + i}.g"] = params[f"b{l}.e{i}.g"].copy()
                new[f"b{l}.e{e + i}.u"] = params[f"b{l}.e{i}.u"].copy()
                new[f"b{l}.e{e + i}.d"] = params[f"b{l}.e{i}.d"].copy()
            # Router is (d, E): duplicate COLUMNS -> (d, 2E) == [R, R].
            # softmax([l, l]) gives each duplicated pair p_i/2, so the
            # mixture is unchanged. Halving the *weights* would be wrong:
            # softmax is translation-invariant, not scale-invariant, so
            # [R/2, R/2] changes the distribution (temperature shift).
            new[f"b{l}.router"] = _dup_cols(params[f"b{l}.router"])
        else:
            n_new = max(cfg.experts, 2)
            for i in range(n_new):
                new[f"b{l}.e{i}.g"] = params[f"b{l}.fg"].copy()
                new[f"b{l}.e{i}.u"] = params[f"b{l}.fu"].copy()
                new[f"b{l}.e{i}.d"] = params[f"b{l}.fd"].copy()
            new[f"b{l}.router"] = np.zeros((cfg.d, n_new), dtype=_FLOAT)
            for k in (f"b{l}.fg", f"b{l}.fu", f"b{l}.fd"):
                del new[k]
    new_cfg = moe_config(cfg, targets)
    return new, new_cfg
