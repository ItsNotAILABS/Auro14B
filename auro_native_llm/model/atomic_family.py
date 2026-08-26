"""Canonical AURO sub-2B family and bounded expert-offload contracts.

The 156K, 250M, and 500M lanes are independently deployable atomic models and
embeddable experts under Auro-2B and larger parents. Named specialist variants
are routing identities until exact checkpoint or adapter evidence proves that
they are separately trained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


SUB2B_CONTRACT_VERSION = "auro.sub2b.atomic.v2.1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def estimate_tokens(text: str) -> int:
    """Stable transport estimate, not an exact tokenizer measurement."""
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _bounded(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 1)] + "..."


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
    head_dim: int = 0
    status: str = "architecture-target-not-trained-checkpoint"
    roles: tuple[str, ...] = ()
    deploy_profiles: tuple[str, ...] = ()

    @property
    def model_class(self) -> str:
        return "atomic"

    @property
    def resolved_head_dim(self) -> int:
        return int(self.head_dim or max(1, self.hidden_size // self.attention_heads))

    @property
    def moe_layers(self) -> int:
        return len(range(1, self.layers, max(1, self.moe_every)))

    def parameter_accounting(self) -> dict[str, int]:
        """Estimate active-per-token and stored weights from declared geometry.

        Tied input/output embeddings are counted once. A SwiGLU expert is
        approximated as three d_model x d_ff matrices. This is architecture
        arithmetic and never substitutes for serialized-checkpoint inspection.
        """
        d = int(self.hidden_size)
        kv_width = int(self.kv_heads * self.resolved_head_dim)
        attention = 2 * d * d + 2 * d * kv_width
        ffn_expert = 3 * d * int(self.intermediate_size)
        moe_layers = self.moe_layers
        dense_layers = int(self.layers) - moe_layers
        embeddings = int(self.vocab_size_target) * d
        norms = (2 * int(self.layers) + 1) * d
        routers = moe_layers * d * int(self.experts)
        shared = embeddings + int(self.layers) * attention + dense_layers * ffn_expert + norms + routers
        active = shared + moe_layers * int(self.top_k) * ffn_expert
        stored = shared + moe_layers * int(self.experts) * ffn_expert
        return {
            "active_parameters_per_token_estimate": int(active),
            "stored_parameters_estimate": int(stored),
            "shared_parameters_estimate": int(shared),
            "moe_layers": int(moe_layers),
            "parameter_target_delta": int(active - self.parameter_target),
        }

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["head_dim"] = self.resolved_head_dim
        row.update(self.parameter_accounting())
        row["model_class"] = self.model_class
        row["contract_version"] = SUB2B_CONTRACT_VERSION
        row["claim_boundary"] = (
            "architecture arithmetic only; exact weights, tokenizer custody, "
            "training provenance, evaluation, serving, and promotion evidence are required"
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
        head_dim=16,
        intermediate_size=64,
        context_window_tokens_target=1_024,
        vocab_size_target=1_024,
        roles=("routing_seed", "classifier", "json_repair", "tool_selection", "style_guard"),
        deploy_profiles=("wasm", "embedded", "high-multiplicity-swarm"),
    ),
    "Auro-250M": AtomicArchitecture(
        model_id="Auro-250M",
        parameter_target=250_000_000,
        hidden_size=768,
        layers=16,
        attention_heads=12,
        kv_heads=4,
        head_dim=64,
        intermediate_size=2_048,
        context_window_tokens_target=4_096,
        vocab_size_target=64_000,
        roles=(
            "intent_extract",
            "retrieval_filter",
            "structured_transform",
            "code_triage",
            "memory_consolidation",
            "semantic_outline",
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
        head_dim=64,
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
            "creative_branch",
        ),
        deploy_profiles=("phone-high-memory", "laptop", "edge-gpu", "embedded-expert"),
    ),
}


@dataclass(frozen=True)
class AtomicVariant:
    variant_id: str
    base_model_id: str
    role: str
    capabilities: tuple[str, ...]
    adapter_required_for_distinct_checkpoint_claim: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AURO_500M_TRIAD: tuple[AtomicVariant, ...] = (
    AtomicVariant(
        variant_id="Auro-500M-SENSUS",
        base_model_id="Auro-500M",
        role="evidence_and_perception",
        capabilities=("research", "retrieval", "fact_check", "context", "risk"),
    ),
    AtomicVariant(
        variant_id="Auro-500M-PRAXIS",
        base_model_id="Auro-500M",
        role="code_and_execution",
        capabilities=("code", "tool", "build", "debug", "workflow"),
    ),
    AtomicVariant(
        variant_id="Auro-500M-VERBUM",
        base_model_id="Auro-500M",
        role="language_and_expression",
        capabilities=("writing", "creative", "explanation", "synthesis", "conversation"),
    ),
)


def architecture_for(model_id: str) -> AtomicArchitecture:
    base = next(
        (item.base_model_id for item in AURO_500M_TRIAD if item.variant_id == model_id),
        model_id,
    )
    try:
        return ATOMIC_LADDER[base]
    except KeyError as exc:
        raise ValueError(f"unknown AURO atomic model: {model_id}") from exc


def atomic_config_overrides(model_id: str) -> dict[str, Any]:
    arch = architecture_for(model_id)
    return {
        "model_id": model_id,
        "tier": "atomic",
        "parameter_target": arch.parameter_target,
        "hidden_dim": arch.hidden_size,
        "num_layers": arch.layers,
        "num_heads": arch.attention_heads,
        "num_kv_heads": arch.kv_heads,
        "head_dim": arch.resolved_head_dim,
        "ffn_dim": arch.intermediate_size,
        "vocab_size": arch.vocab_size_target,
        "max_seq_len": arch.context_window_tokens_target,
        "use_moe": True,
        "num_experts": arch.experts,
        "top_k_experts": arch.top_k,
        "moe_every": arch.moe_every,
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


class HierarchicalAtomicCouncil:
    """Dispatch bounded capsules to atomic experts and reconcile observations."""

    def __init__(self, parent_model_id: str = "Auro-2B"):
        self.parent_model_id = parent_model_id
        self._experts: dict[tuple[str, str], ExpertCallable] = {}

    def register(self, model_id: str, role: str, expert: ExpertCallable) -> None:
        architecture_for(model_id)
        if role not in architecture_for(model_id).roles:
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
        architecture = architecture_for(model_id)
        if role not in architecture.roles:
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
        dispatch_tokens = sum(
            estimate_tokens(json.dumps(item.to_dict(), sort_keys=True)) for item in capsules
        )
        naive_tokens = estimate_tokens(full_parent_context) * max(1, len(capsules))
        reduction = 0.0 if naive_tokens <= 0 else max(0.0, 1.0 - dispatch_tokens / naive_tokens)
        receipt_material = {
            "schema": "auro.atomic.council.receipt.v2",
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
    def _normalize(
        raw: ExpertObservation | Mapping[str, Any] | str,
        capsule: TaskCapsule,
        latency_ms: float,
    ) -> ExpertObservation:
        if isinstance(raw, ExpertObservation):
            answer = raw.answer
            confidence = raw.confidence
            evidence = tuple(raw.evidence)
        elif isinstance(raw, Mapping):
            answer = str(raw.get("answer", ""))
            confidence = float(raw.get("confidence", 0.5))
            evidence = tuple(str(item) for item in raw.get("evidence", ()))
        else:
            answer = str(raw)
            confidence = 0.5
            evidence = ()
        return ExpertObservation(
            task_id=capsule.task_id,
            expert_model_id=capsule.expert_model_id,
            role=capsule.role,
            answer=answer,
            confidence=max(0.0, min(1.0, float(confidence))),
            evidence=evidence,
            proposed_tokens=estimate_tokens(answer),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _consensus(
        observations: Sequence[ExpertObservation],
    ) -> tuple[str, list[str], float]:
        if not observations:
            return "", ["no expert observations"], 0.0
        ranked = sorted(
            observations,
            key=lambda item: (-item.confidence, item.expert_model_id, item.role),
        )
        selected = ranked[0]
        selected_normal = " ".join(selected.answer.lower().split())
        disagreements = [
            f"{item.expert_model_id}:{item.role} produced a distinct candidate"
            for item in ranked[1:]
            if " ".join(item.answer.lower().split()) != selected_normal
        ]
        confidence = sum(item.confidence for item in observations) / len(observations)
        if disagreements:
            confidence *= 0.9
        return selected.answer, disagreements, round(max(0.0, min(1.0, confidence)), 6)


def sub2b_manifest(extra_variants: Iterable[AtomicVariant] = ()) -> dict[str, Any]:
    variants = (*AURO_500M_TRIAD, *tuple(extra_variants))
    return {
        "schema": "auro.sub2b.family.v2",
        "contract_version": SUB2B_CONTRACT_VERSION,
        "parent_lane": "Auro-2B",
        "lanes": [ATOMIC_LADDER[key].to_dict() for key in ("Auro-156K", "Auro-250M", "Auro-500M")],
        "triad": [item.to_dict() for item in variants],
        "execution_model": (
            "2B parent -> three 500M specialists -> topic-scoped 250M/156K swarms "
            "-> triad consensus -> 2B synthesis"
        ),
        "checkpoint_release_required": True,
        "parameter_accounting": (
            "report each loaded checkpoint independently; do not add agent instances "
            "to one model's parameter count"
        ),
    }
