"""Canonical AURO capacity x capability taxonomy.

The taxonomy describes architecture, deployment, and composition classes. It
never treats a model name, config, or local directory as proof of a promoted
checkpoint.
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
    ModelClassSpec(ModelClass.ATOMIC, 1, 1_000_000_000, "single-purpose intelligence units composed in swarms, colonies, councils, tools, and embedded agents", "browser, phone, edge, CPU, WebGPU, WASM, embedded and high-multiplicity deployments"),
    ModelClassSpec(ModelClass.MICRO, 1_000_000_000, 5_000_000_000, "standalone routers, tool users, coding agents, domain specialists and atomic-swarm supervisors", "local workstation, compact GPU, private API and edge-server deployments"),
    ModelClassSpec(ModelClass.CORE, 5_000_000_000, 10_000_000_000, "general reasoning, synthesis, planning and multi-domain execution", "workstation and server inference"),
    ModelClassSpec(ModelClass.ORCHESTRATOR, 10_000_000_000, 30_000_000_000, "council coordination, long workflows and multi-model orchestration", "GPU server and distributed private runtime"),
    ModelClassSpec(ModelClass.FRONTIER, 30_000_000_000, None, "research-scale long-horizon intelligence and deep council supervision", "distributed training and inference architecture"),
]


RELEASE_LADDER: Dict[str, Dict[str, object]] = {
    "Auro-156K": {
        "parameter_target": 156_000,
        "model_class": ModelClass.ATOMIC.value,
        "role": "routing seed, classifier, JSON repair and high-multiplicity specialization cell",
        "deploy_profiles": ["wasm", "embedded", "high-multiplicity-swarm"],
        "release_policy": "downloadable only when exact weights, tokenizer, hash manifest and evaluation receipt are present",
    },
    "Auro-250M": {
        "parameter_target": 250_000_000,
        "model_class": ModelClass.ATOMIC.value,
        "role": "phone and browser expert for intent, retrieval filtering, structured transforms and memory consolidation",
        "deploy_profiles": ["phone", "browser-wasm", "cpu", "embedded-expert"],
        "release_policy": "architecture and training lane until exact promoted checkpoint evidence exists",
    },
    "Auro-500M": {
        "parameter_target": 500_000_000,
        "model_class": ModelClass.ATOMIC.value,
        "role": "edge worker and embedded specialist for planning, code, evidence, expansion and consensus",
        "deploy_profiles": ["phone-high-memory", "laptop", "edge-gpu", "embedded-expert"],
        "release_policy": "architecture and training lane until exact promoted checkpoint evidence exists",
    },
    "Auro-2B": {
        "parameter_target": 2_000_000_000,
        "model_class": ModelClass.MICRO.value,
        "role": "parent coordinator for three 500M specialists, atomic swarms, tools and private local assistance",
        "composition": {
            "specialist_triad": ["Auro-500M-SENSUS", "Auro-500M-PRAXIS", "Auro-500M-VERBUM"],
            "atomic_lanes": ["Auro-156K", "Auro-250M", "Auro-500M"],
        },
        "release_policy": "checkpoint-specific evidence required; local operator claims do not establish promotion",
    },
    "Auro-4B": {
        "parameter_target": 4_000_000_000,
        "active_parameters_per_token_estimate": 4_026_977_280,
        "stored_expert_capacity_estimate": 7_650_855_936,
        "model_class": ModelClass.MICRO.value,
        "role": "sparse top-2 eight-expert coding, structured-output, specialist-planning and council-supervision lane",
        "release_policy": "active and stored capacity are architecture estimates; exact checkpoint and routing evidence required",
    },
    "Auro-8B": {
        "parameter_target": 8_000_000_000,
        "model_class": ModelClass.CORE.value,
        "role": "general reasoning, planning, critique and synthesis",
        "release_policy": "architecture target until exact promoted checkpoint evidence exists",
    },
    "Auro-14B": {
        "parameter_target": 14_000_000_000,
        "model_class": ModelClass.ORCHESTRATOR.value,
        "role": "multi-model orchestrator and council chair",
        "release_policy": "training lane; not a finished 14B checkpoint unless exact promoted checkpoint evidence exists",
    },
    "Auro-100B": {
        "parameter_target": 100_000_000_000,
        "model_class": ModelClass.FRONTIER.value,
        "role": "frontier research architecture",
        "release_policy": "architecture target only",
    },
}


CANONICAL_RELEASE_ORDER = tuple(RELEASE_LADDER.keys())


def classify_parameter_count(parameters: int) -> ModelClass:
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    for spec in MODEL_CLASSES:
        if spec.contains(parameters):
            return spec.name
    raise AssertionError("taxonomy must cover every positive parameter count")


def release_ladder() -> Dict[str, Dict[str, object]]:
    return {model_id: dict(data) for model_id, data in RELEASE_LADDER.items()}
