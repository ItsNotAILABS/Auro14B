"""Canonical AURO capacity taxonomy and release ladder.

Capacity is one routing axis; capability is the other. A family name or target
never proves that an exact trained checkpoint exists.
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
        return parameters >= self.min_parameters and (self.max_parameters_exclusive is None or parameters < self.max_parameters_exclusive)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


MODEL_CLASSES: List[ModelClassSpec] = [
    ModelClassSpec(ModelClass.ATOMIC, 1, 1_000_000_000, "single-purpose intelligence units composed in colonies, triads, swarms, tools and embedded experts", "browser, phone, edge, CPU, WASM and high-multiplicity deployments"),
    ModelClassSpec(ModelClass.MICRO, 1_000_000_000, 5_000_000_000, "standalone routers, tool users, coding agents, domain specialists and atomic-swarm supervisors", "local workstation, compact GPU, private API and edge server"),
    ModelClassSpec(ModelClass.CORE, 5_000_000_000, 10_000_000_000, "general reasoning, synthesis, planning and multi-domain execution", "workstation and server inference"),
    ModelClassSpec(ModelClass.ORCHESTRATOR, 10_000_000_000, 30_000_000_000, "council coordination, long workflows and multi-model orchestration", "GPU server and distributed private runtime"),
    ModelClassSpec(ModelClass.FRONTIER, 30_000_000_000, None, "research-scale long-horizon intelligence and deep council supervision", "distributed training and inference architecture"),
]


RELEASE_LADDER: Dict[str, Dict[str, object]] = {
    "Auro-156K": {"parameter_target": 156_000, "model_class": "atomic", "role": "routing, classification, repair and specialization seed", "release_policy": "exact weights, tokenizer, evaluation and promotion receipt required"},
    "Auro-250M": {"parameter_target": 250_000_000, "model_class": "atomic", "role": "phone/WASM retrieval, transformation, triage and memory expert", "release_policy": "architecture lane until an exact checkpoint bundle passes mobile and swarm evaluation"},
    "Auro-500M": {"parameter_target": 500_000_000, "model_class": "atomic", "role": "edge worker and base for SENSUS, PRAXIS and VERBUM triad specialists", "release_policy": "base checkpoint plus separate specialization evidence required for each triad identity"},
    "Auro-2B": {"parameter_target": 2_000_000_000, "model_class": "micro", "role": "private parent model, tool router and supervisor of the three-500M triad", "release_policy": "checkpoint-specific evidence and triad compatibility receipt required"},
    "Auro-4B": {"parameter_target": 4_000_000_000, "model_class": "micro", "role": "coding, structured output, specialist planning and council supervision", "release_policy": "checkpoint-specific evidence required"},
    "Auro-8B": {"parameter_target": 8_000_000_000, "model_class": "core", "role": "general reasoning, planning, critique and synthesis", "release_policy": "architecture target until promoted checkpoint evidence exists"},
    "Auro-14B": {"parameter_target": 14_000_000_000, "model_class": "orchestrator", "role": "multi-model orchestrator and council chair", "release_policy": "training lane; exact promoted checkpoint evidence required"},
    "Auro-100B": {"parameter_target": 100_000_000_000, "model_class": "frontier", "role": "frontier research architecture", "release_policy": "architecture target only"},
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
