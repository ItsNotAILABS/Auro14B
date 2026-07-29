"""Canonical AURO sub-2B atomic family contracts.

The 250M and 500M lanes extend the existing 156K atomic strategy. They are
simultaneously deployable edge checkpoints and embeddable experts under Auro-2B
and larger parents. Architecture declarations never stand in for trained
checkpoint evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

SUB2B_CONTRACT_VERSION = "auro.sub2b.atomic.v2"


@dataclass(frozen=True)
class AtomicArchitecture:
    model_id: str
    parameter_target: int
    hidden_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
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

    @property
    def moe_layers(self) -> int:
        return len(range(1, self.layers, max(1, self.moe_every)))

    def parameter_accounting(self) -> dict[str, int]:
        """Estimate active-per-token and stored weights from declared geometry.

        This is architecture arithmetic, not serialized-checkpoint inspection.
        Tied input/output embeddings are counted once. A SwiGLU expert is
        approximated as three d_model x d_ff matrices.
        """
        d = int(self.hidden_size)
        kv_width = int(self.kv_heads * self.head_dim)
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
        }

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(self.parameter_accounting())
        row["model_class"] = self.model_class
        row["contract_version"] = SUB2B_CONTRACT_VERSION
        row["claim_boundary"] = (
            "architecture arithmetic only; exact weights, tokenizer custody, "
            "training provenance, evaluation and promotion evidence are required"
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
        layers=18,
        attention_heads=12,
        kv_heads=4,
        head_dim=64,
        intermediate_size=2_560,
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
        intermediate_size=3_584,
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
        "Auro-500M-SENSUS",
        "Auro-500M",
        "evidence_and_perception",
        ("research", "retrieval", "fact_check", "context", "risk"),
    ),
    AtomicVariant(
        "Auro-500M-PRAXIS",
        "Auro-500M",
        "code_and_execution",
        ("code", "tool", "build", "debug", "workflow"),
    ),
    AtomicVariant(
        "Auro-500M-VERBUM",
        "Auro-500M",
        "language_and_expression",
        ("writing", "creative", "explanation", "synthesis", "conversation"),
    ),
)


def architecture_for(model_id: str) -> AtomicArchitecture:
    base = next((item.base_model_id for item in AURO_500M_TRIAD if item.variant_id == model_id), model_id)
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
        "head_dim": arch.head_dim,
        "ffn_dim": arch.intermediate_size,
        "vocab_size": arch.vocab_size_target,
        "max_seq_len": arch.context_window_tokens_target,
        "use_moe": True,
        "num_experts": arch.experts,
        "top_k_experts": arch.top_k,
        "moe_every": arch.moe_every,
    }


def sub2b_manifest(extra_variants: Iterable[AtomicVariant] = ()) -> dict[str, Any]:
    variants = (*AURO_500M_TRIAD, *tuple(extra_variants))
    return {
        "schema": "auro.sub2b.family.v2",
        "contract_version": SUB2B_CONTRACT_VERSION,
        "parent_lane": "Auro-2B",
        "lanes": [ATOMIC_LADDER[key].to_dict() for key in ("Auro-156K", "Auro-250M", "Auro-500M")],
        "triad": [item.to_dict() for item in variants],
        "execution_model": "2B parent -> three 500M specialists -> topic-scoped 250M/156K swarms -> triad consensus -> 2B synthesis",
        "checkpoint_release_required": True,
        "parameter_accounting": "do not add agent instances to one model's parameter count; report each loaded checkpoint independently",
    }
