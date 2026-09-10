"""Auro exponential-growth training mechanics ("career ladder").

Implements the *mechanics* of progressive growth for the Auro SpectralGPT
core (RMSNorm, RoPE, GQA, SwiGLU, QK-norm, MoE every other layer):
function-preserving Net2Net-style growth operators, an exponential
milestone schedule, a domain curriculum, sandboxed code execution for the
tool-use milestone, and a train -> checkpoint -> grow -> continue runner.

Everything here is verified on synthetic data. Nothing in this package is
a capability claim: growth mechanics do not imply intelligence gains.
See README.md for the honest framing.
"""

from auro_native_llm.growth.core import GrowthConfig, GrowthTransformer, count_params
from auro_native_llm.growth.schedule import rung_targets, plan_growth, LANE_2B_TOTAL
from auro_native_llm.growth.curriculum import (
    MILESTONES,
    milestones_for_rung,
    curriculum_summary,
    GENERATORS,
)
from auro_native_llm.growth.exec_sandbox import (
    ExecResult,
    ModuleTask,
    run_python,
    run_powershell,
    powershell_available,
    example_tasks,
)

__all__ = [
    "GrowthConfig", "GrowthTransformer", "count_params",
    "rung_targets", "plan_growth", "LANE_2B_TOTAL",
    "MILESTONES", "milestones_for_rung", "curriculum_summary", "GENERATORS",
    "ExecResult", "ModuleTask", "run_python", "run_powershell",
    "powershell_available", "example_tasks",
]
