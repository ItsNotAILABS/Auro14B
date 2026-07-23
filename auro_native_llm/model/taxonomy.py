"""Canonical AURO family taxonomy.

The taxonomy describes deployment and composition classes. It does not imply
that a trained checkpoint exists for every architecture target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List


class ModelClass(str, Enum):
    ATOMIC = "atomic"
    MICRO = "micro"
    CORE = "core"
    ORCHESTRATOR = "orchestrator"
    FRONTIER = "frontier"


@dataclass(frozen=True)
class ModelClassSpec:
    name: ModelClass
    min_parameters: int
    max_parameters_exclusive: int | None
    composition_role: str
    deployment_profile: str

    def contains(self, parameters: int) -> bool:
        if parameters < self.min_parameters:
            return False
        return self.max_parameters_exclusive is None or parameters < self.max_parameters_exclusive

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


MODEL_CLASSES: List[ModelClassSpec] = [
    ModelClassSpec(
        ModelClass.ATOMIC,
        1,
        1_000_000_000,
        "single-purpose intelligence units composed in colonies, councils, tools, and embedded agents",
        "browser, edge, mobile, CPU, WASM, embedded and high-multiplicity deployments",
    ),
    ModelClassSpec(
        ModelClass.MICRO,
        1_000_000_000,
        5_000_000_000,
        "standalone routers, tool users, coding agents, domain specialists and colony supervisors",
        "local workstation, compact GPU, private API and edge-server deployments",
    ),
    ModelClassSpec(
        ModelClass.CORE,
        5_000_000_000,
        10_000_000_000,
        "general reasoning, synthesis, planning and multi-domain execution",
        "workstation and server inference",
    ),
    ModelClassSpec(
        ModelClass.ORCHESTRATOR,
        10_000_000_000,
        30_000_000_000,
        "council coordination, long workflows and multi-model orchestration",
        "GPU server and distributed private runtime",
    ),
    ModelClassSpec(
        ModelClass.FRONTIER,
        30_000_000_000,
        None,
        "research-scale long-horizon intelligence and deep council supervision",
        "distributed training and inference architecture",
    ),
]


RELEASE_LADDER: Dict[str, Dict[str, object]] = {
    "Auro-156K": {
        "parameter_target": 156_000,
        "model_class": ModelClass.ATOMIC.value,
        "role": "reference atomic checkpoint and specialization seed",
        "release_policy": "downloadable only when weights, tokenizer, hash manifest and evaluation receipt are present",
    },
    "Auro-2B": {
        "parameter_target": 2_000_000_000,
        "model_class": ModelClass.MICRO.value,
        "role": "router, tool-use, spectral triage and private local assistant",
        "release_policy": "checkpoint-specific evidence required",
    },
    "Auro-4B": {
        "parameter_target": 4_000_000_000,
        "model_class": ModelClass.MICRO.value,
        "role": "coding, structured output, specialist planning and council supervision",
        "release_policy": "checkpoint-specific evidence required",
    },
    "Auro-8B": {
        "parameter_target": 8_000_000_000,
        "model_class": ModelClass.CORE.value,
        "role": "general reasoning, planning, critique and synthesis",
        "release_policy": "architecture target until promoted checkpoint evidence exists",
    },
    "Auro-14B": {
        "parameter_target": 14_000_000_000,
        "model_class": ModelClass.ORCHESTRATOR.value,
        "role": "multi-model orchestrator and council chair",
        "release_policy": "training lane; do not describe as a finished 14B checkpoint without promotion evidence",
    },
    "Auro-100B": {
        "parameter_target": 100_000_000_000,
        "model_class": ModelClass.FRONTIER.value,
        "role": "frontier research architecture",
        "release_policy": "architecture target only",
    },
}


def classify_parameter_count(parameters: int) -> ModelClass:
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    for spec in MODEL_CLASSES:
        if spec.contains(parameters):
            return spec.name
    raise AssertionError("taxonomy must cover every positive parameter count")


def release_ladder() -> Dict[str, Dict[str, object]]:
    return {model_id: dict(data) for model_id, data in RELEASE_LADDER.items()}
