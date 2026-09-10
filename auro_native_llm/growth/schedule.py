"""Exponential milestone schedule for the Auro career ladder.

Rung ``n+1`` targets ``factor`` x the params of rung ``n`` (default 2x).
The 2B-lane targets are the user's published honest numbers
(total 4,251,025,408 / active 1,497,464,832 per token); they are the
*destination* of the schedule, not a claim that the ladder has been run.

``plan_growth`` picks the operator combination whose achieved size lands
closest to the ideal 2x target, because exact doubling is not always
reachable with function-preserving operators (embeddings don't grow).
Both ideal and achieved sizes are logged by the runner.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from auro_native_llm.growth.core import GrowthConfig, count_params
from auro_native_llm.growth import operators as ops

# User's published honest 2B-lane accounting (Auro-2B config, 2026-09-10).
LANE_2B_TOTAL = 4_251_025_408
LANE_2B_ACTIVE = 1_497_464_832

GROWTH_FACTOR = 2.0


def rung_targets(p0: int, target: int = LANE_2B_TOTAL,
                 factor: float = GROWTH_FACTOR) -> List[int]:
    """Ideal exponential rung sizes from ``p0`` up to ``target`` (inclusive).

    Monotonic and capped: each ideal doubling is clamped to ``target`` so
    the schedule never overshoots it, and the final rung lands exactly on
    ``target``. ``len(rung_targets(p0)) == num_rungs(p0) + 1`` always holds.
    """
    rungs = [int(p0)]
    while rungs[-1] < target:
        nxt = min(int(round(rungs[-1] * factor)), target)
        if nxt <= rungs[-1]:
            break
        rungs.append(nxt)
    return rungs


def num_rungs(p0: int, target: int = LANE_2B_TOTAL,
              factor: float = GROWTH_FACTOR) -> int:
    """Number of doublings needed to reach ``target`` from ``p0``.

    Zero when already at/past the target; ``len(rung_targets(p0))`` is
    always ``num_rungs(p0) + 1``.
    """
    if p0 >= target:
        return 0
    return int(math.ceil(math.log(target / p0, factor)))


@dataclass
class RungCriteria:
    """Done-criteria for one rung. Defaults are mechanics placeholders.

    A real curriculum replaces these with data-driven gates (eval suites,
    loss thresholds on held-out data). The runner enforces ``max_steps``
    and records whether ``loss_target`` was met; it never claims the
    threshold implies capability.
    """

    max_steps: int = 500
    tokens_per_step: int = 2048      # batch * seq; budget accounting
    loss_target: Optional[float] = None
    eval_every: int = 100
    min_steps: int = 50


def default_criteria(rung: int, cfg: GrowthConfig) -> RungCriteria:
    """Placeholder criteria per rung; the curriculum is the remaining work."""
    return RungCriteria(
        max_steps=500,
        tokens_per_step=32 * cfg.seq,
        loss_target=None,            # set by curriculum, not by mechanics
        eval_every=100,
        min_steps=50,
    )


# Operator combos considered by the planner: (name, config_fn, apply_fn).
def _combos():
    return {
        "width": (ops.width_config, ops.grow_width),
        "depth": (ops.depth_config, ops.grow_depth),
        "moe": (ops.moe_config, ops.grow_moe),
        "width+depth": (lambda c: ops.depth_config(ops.width_config(c)),
                        lambda p, c: ops.grow_depth(*ops.grow_width(p, c))),
        "width+moe": (lambda c: ops.moe_config(ops.width_config(c)),
                      lambda p, c: ops.grow_moe(*ops.grow_width(p, c))),
        "depth+moe": (lambda c: ops.moe_config(ops.depth_config(c)),
                      lambda p, c: ops.grow_moe(*ops.grow_depth(p, c))),
    }


def plan_growth(cfg: GrowthConfig, target_params: int) -> Tuple[str, GrowthConfig, int]:
    """Pick the operator combo whose size lands closest to ``target_params``.

    Returns ``(combo_name, new_cfg, achieved_params)``. Pure function of the
    config: no weights are materialized.
    """
    best = None
    for name, (cfg_fn, _apply) in _combos().items():
        try:
            new_cfg = cfg_fn(cfg)
        except Exception:
            continue
        achieved = count_params(new_cfg)
        score = abs(math.log(achieved / target_params))
        if best is None or score < best[0]:
            best = (score, name, new_cfg, achieved)
    assert best is not None
    _, name, new_cfg, achieved = best
    return name, new_cfg, achieved


def apply_plan(params: Dict, cfg: GrowthConfig, plan_name: str):
    """Apply a named plan from :func:`plan_growth` to real params."""
    return _combos()[plan_name][1](params, cfg)


def describe_ladder(p0: int, target: int = LANE_2B_TOTAL) -> str:
    rungs = rung_targets(p0, target)
    lines = [f"rung {i}: ~{p:,} params" for i, p in enumerate(rungs)]
    lines.append(f"{len(rungs) - 1} doublings from {p0:,} to {target:,}")
    return "\n".join(lines)
