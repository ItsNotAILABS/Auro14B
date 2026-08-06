"""AURO sub-2B atomic family and hierarchical expert-offload runtime.

This module extends, rather than replaces, the existing AURO family. 250M and
500M are dual-purpose lanes: independently deployable edge models and embedded
experts used by 2B+ parents. The runtime moves bounded task capsules instead of
repeating the full parent context to every expert.

Architecture contracts are not trained-checkpoint claims. A release still
requires exact weights, tokenizer, provenance, evaluations, hashes, launch
proof, promotion authorization, and rollback evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


SUB2B_CONTRACT_VERSION = "auro.sub2b.atomic.v1"


@dataclass(frozen=True)
class AtomicArchitecture:
    model_id: str
    parameter_target: int
    hidden_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    intermediate_size: int
    context_window_tokens_target: int
    vocab_size_target: int
    experts: int = 8
    top_k: int = 2
    moe_every: int = 2
    status: str = "architecture-target-not-trained-checkpoint"
    roles: tuple[str, ...] = ()
    deploy_profiles: tuple[str, ...] = ()

    @property
    def model_class(self) -> str:
        return "atomic"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["model_class"] = self.model_class
        row["contract_version"] = SUB2B_CONTRACT_VERSION
        row["claim_boundary"] = (
            "architecture and training lane only; no trained checkpoint or "
            "quality result is implied"
        )
        return row


ATOMIC_LADDER: dict[str, AtomicArchitecture] = {
    "Auro-156K": AtomicArchitecture(
        model_id="Auro-156K",
        parameter_target=156_000,
        hidden_size=64,
        layers=2,
        attention_heads=4,
        kv_heads=2,
        intermediate_size=64,
        context_window_tokens_target=1_024,
        vocab_size_target=1_024,
        roles=("routing_seed", "classifier", "json_repair", "tool_selection"),
        deploy_profiles=("wasm", "embedded", "high-multiplicity-swarm"),
    ),
    "Auro-250M": AtomicArchitecture(
        model_id="Auro-250M",
        parameter_target=250_000_000,
        hidden_size=768,
        layers=16,
        attention_heads=12,
        kv_heads=4,
        intermediate_size=2_048,
        context_window_tokens_target=4_096,
        vocab_size_target=64_000,
        roles=(
            "intent_extract",
            "retrieval_filter",
            "structured_transform",
            "code_triage",
            "memory_consolidation",
        ),
        deploy_profiles=("phone", "browser-wasm", "cpu", "embedded-expert"),
    ),
    "Auro-500M": AtomicArchitecture(
        model_id="Auro-500M",
        parameter_target=500_000_000,
        hidden_size=1_024,
        layers=24,
        attention_heads=16,
        kv_heads=4,
        intermediate_size=4_096,
        context_window_tokens_target=8_192,
        vocab_size_target=64_000,
        roles=(
            "tool_execution_plan",
            "code_patch",
            "evidence_review",
            "local_worker",
            "expert_consensus",
            "text_expansion",
        ),
        deploy_profiles=("phone-high-memory", "laptop", "edge-gpu", "embedded-expert"),
    ),
}


@dataclass(frozen=True)
class TaskCapsule:
    task_id: str
    parent_model_id: str
    expert_model_id: str
    role: str
    objective: str
    constraints: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    max_output_tokens: int
    created_at_ms: int
    capsule_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertObservation:
    task_id: str
    expert_model_id: str
    role: str
    answer: str
    confidence: float
    evidence: tuple[str, ...] = ()
    proposed_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CouncilResult:
    task_id: str
    parent_model_id: str
    observations: tuple[ExpertObservation, ...]
    consensus: str
    confidence: float
    disagreements: tuple[str, ...]
    dispatch_tokens: int
    naive_broadcast_tokens: int
    estimated_text_reduction: float
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["observations"] = [item.to_dict() for item in self.observations]
        return row


ExpertCallable = Callable[[TaskCapsule], ExpertObservation | Mapping[str, Any] | str]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def estimate_tokens(text: str) -> int:
    """Deterministic architecture metric, not a tokenizer benchmark."""
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _bounded(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 1)] + "…"


class HierarchicalAtomicCouncil:
    """Dispatch compressed capsules to 250M/500M experts and reconcile results.

    The parent retains the complete conversation/context. Children receive only
    the role-specific objective, explicit constraints, and content-addressed
    evidence references. This avoids broadcasting the entire prompt to every
    expert while preserving auditable delegation.
    """

    def __init__(self, parent_model_id: str = "Auro-2B"):
        self.parent_model_id = parent_model_id
        self._experts: dict[tuple[str, str], ExpertCallable] = {}

    def register(self, model_id: str, role: str, expert: ExpertCallable) -> None:
        if model_id not in ATOMIC_LADDER:
            raise ValueError(f"unknown atomic model: {model_id}")
        if role not in ATOMIC_LADDER[model_id].roles:
            raise ValueError(f"{model_id} does not declare role {role}")
        self._experts[(model_id, role)] = expert

    def build_capsule(
        self,
        *,
        task_id: str,
        model_id: str,
        role: str,
        objective: str,
        constraints: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
        max_output_tokens: int = 256,
    ) -> TaskCapsule:
        if model_id not in ATOMIC_LADDER:
            raise ValueError(f"unknown atomic model: {model_id}")
        if role not in ATOMIC_LADDER[model_id].roles:
            raise ValueError(f"role {role} is not declared by {model_id}")
        material = {
            "task_id": task_id,
            "parent_model_id": self.parent_model_id,
            "expert_model_id": model_id,
            "role": role,
            "objective": _bounded(objective, 1_200),
            "constraints": tuple(_bounded(item, 240) for item in constraints[:12]),
            "evidence_refs": tuple(str(item) for item in evidence_refs[:24]),
            "max_output_tokens": max(16, min(int(max_output_tokens), 2_048)),
            "created_at_ms": int(time.time() * 1_000),
        }
        return TaskCapsule(capsule_hash=_sha(material), **material)

    def run(
        self,
        *,
        task_id: str,
        full_parent_context: str,
        assignments: Iterable[Mapping[str, Any]],
    ) -> CouncilResult:
        observations: list[ExpertObservation] = []
        capsules: list[TaskCapsule] = []
        for assignment in assignments:
            model_id = str(assignment["model_id"])
            role = str(assignment["role"])
            key = (model_id, role)
            if key not in self._experts:
                raise ValueError(f"no registered expert for {model_id}:{role}")
            capsule = self.build_capsule(
                task_id=task_id,
                model_id=model_id,
                role=role,
                objective=str(assignment["objective"]),
                constraints=tuple(assignment.get("constraints", ())),
                evidence_refs=tuple(assignment.get("evidence_refs", ())),
                max_output_tokens=int(assignment.get("max_output_tokens", 256)),
            )
            capsules.append(capsule)
            started = time.perf_counter()
            raw = self._experts[key](capsule)
            latency_ms = round((time.perf_counter() - started) * 1_000, 3)
            observations.append(self._normalize(raw, capsule, latency_ms))

        consensus, disagreements, confidence = self._consensus(observations)
        dispatch_tokens = sum(estimate_tokens(json.dumps(item.to_dict(), sort_keys=True)) for item in capsules)
        naive_tokens = estimate_tokens(full_parent_context) * max(1, len(capsules))
        reduction = 0.0 if naive_tokens <= 0 else max(0.0, 1.0 - dispatch_tokens / naive_tokens)
        receipt_material = {
            "schema": "auro.atomic.council.receipt.v1",
            "task_id": task_id,
            "parent_model_id": self.parent_model_id,
            "capsules": [item.capsule_hash for item in capsules],
            "observations": [item.to_dict() for item in observations],
            "consensus": consensus,
            "confidence": confidence,
            "disagreements": disagreements,
            "dispatch_tokens": dispatch_tokens,
            "naive_broadcast_tokens": naive_tokens,
        }
        return CouncilResult(
            task_id=task_id,
            parent_model_id=self.parent_model_id,
            observations=tuple(observations),
            consensus=consensus,
            confidence=confidence,
            disagreements=tuple(disagreements),
            dispatch_tokens=dispatch_tokens,
            naive_broadcast_tokens=naive_tokens,
            estimated_text_reduction=round(reduction, 6),
            receipt_hash=_sha(receipt_material),
        )

    @staticmethod
    def _normalize(raw: ExpertObservation | Mapping[str, Any] | str, capsule: TaskCapsule, latency_ms: float) -> ExpertObservation:
        if isinstance(raw, ExpertObservation):
            return ExpertObservation(
                task_id=raw.task_id,
                expert_model_id=raw.expert_model_id,
                role=raw.role,
                answer=raw.answer,
                confidence=max(0.0, min(1.0, float(raw.confidence))),
                evidence=tuple(raw.evidence),
                proposed_tokens=raw.proposed_tokens or estimate_tokens(raw.answer),
                latency_ms=latency_ms,
            )
        if isinstance(raw, Mapping):
            answer = str(raw.get("answer", ""))
            return ExpertObservation(
                task_id=capsule.task_id,
                expert_model_id=capsule.expert_model_id,
                role=capsule.role,
                answer=answer,
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
                evidence=tuple(str(item) for item in raw.get("evidence", ())),
                proposed_tokens=estimate_tokens(answer),
                latency_ms=latency_ms,
            )
        answer = str(raw)
        return ExpertObservation(
            task_id=capsule.task_id,
            expert_model_id=capsule.expert_model_id,
            role=capsule.role,
            answer=answer,
            confidence=0.5,
            proposed_tokens=estimate_tokens(answer),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _consensus(observations: Sequence[ExpertObservation]) -> tuple[str, list[str], float]:
        if not observations:
            return "", ["no expert observations"], 0.0
        ranked = sorted(observations, key=lambda item: (-item.confidence, item.expert_model_id, item.role))
        selected = ranked[0]
        normalized = {" ".join(item.answer.lower().split()) for item in observations if item.answer.strip()}
        disagreements = [] if len(normalized) <= 1 else [
            f"{item.expert_model_id}:{item.role} produced a distinct candidate" for item in ranked[1:]
            if " ".join(item.answer.lower().split()) != " ".join(selected.answer.lower().split())
        ]
        confidence = sum(item.confidence for item in observations) / len(observations)
        if disagreements:
            confidence *= 0.9
        return selected.answer, disagreements, round(max(0.0, min(1.0, confidence)), 6)


def sub2b_manifest() -> dict[str, Any]:
    return {
        "schema": "auro.sub2b.family.v1",
        "contract_version": SUB2B_CONTRACT_VERSION,
        "parent_lane": "Auro-2B",
        "lanes": [ATOMIC_LADDER[key].to_dict() for key in ("Auro-156K", "Auro-250M", "Auro-500M")],
        "execution_model": "bounded task capsules -> atomic experts -> parent consensus",
        "checkpoint_release_required": True,
    }
