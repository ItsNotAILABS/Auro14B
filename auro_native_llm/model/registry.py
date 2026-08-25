"""Canonical AURO model, feature, composition, and claim-boundary registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    id: str
    class_name: str
    intended_use: str
    architecture: str
    moe: bool
    declared_context_tokens: int
    preferred_personas: tuple[str, ...]
    features: tuple[str, ...]
    checkpoint_status: str
    claim_boundary: str


NEURO_FEATURE = "neuromorphic-residual-gate"

MODELS: tuple[ModelProfile, ...] = (
    ModelProfile("HIM-native-v0", "pipeline-fixture", "CPU pipeline and custody validation", "context-MLP causal LM", False, 16, ("memory_keeper",), ("byte-tokenizer", "local-open-weights", "hash-verified-loader"), "fixture-only", "proves pipeline mechanics, not assistant quality or mature HIM intelligence"),
    ModelProfile("Auro-156K", "atomic", "routing seed, classifier, JSON repair and high-multiplicity specialization cell", "decoder-only top-2 MoE", True, 1024, ("sensus", "memory_keeper"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "constitutional-checkpoints", "wasm", "swarm-cell"), "architecture-target", "the 156K label is a target class; no quality claim without exact checkpoint evidence"),
    ModelProfile("Auro-250M", "atomic", "phone and browser expert for intent, retrieval filtering, structured transformation and memory consolidation", "decoder-only top-2 MoE", True, 4096, ("sensus", "memory_keeper", "browser_brain"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "phone", "browser-wasm", "embedded-expert"), "architecture-target", "phone, browser, latency and quality claims require exact checkpoint and device evidence"),
    ModelProfile("Auro-500M", "atomic", "edge worker and embedded specialist for tools, code, evidence, expansion and consensus", "decoder-only top-2 MoE", True, 8192, ("sensus", "operator", "architect", "researcher", "builder"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "edge-worker", "expert-consensus", "embedded-expert"), "architecture-target", "SENSUS, PRAXIS and VERBUM are routing identities until exact checkpoint or adapter evidence exists"),
    ModelProfile("Auro-2B", "micro", "parent coordinator for three 500M specialists, atomic swarms, tools and private local assistance", "decoder-only MoE with hierarchical council", True, 8192, ("sensus", "operator", "memory_keeper", "browser_brain"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "atomic-swarm", "500m-triad", "MESIE-offload", "python-wasm-fluidizer", "checkpoint-receipts"), "architecture-target", "the composed council is not a merged 3.5B checkpoint and requires exact specialist and parent evidence"),
    ModelProfile("Auro-4B", "micro", "tool use, coding, quantitative work, browser agents and council supervision", "decoder-only sparse top-2 MoE with structured residuals", True, 65536, ("mathesis", "architect", "red_team", "operator", "researcher", "builder", "browser_brain"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "Walsh-Hadamard", "long-context-evidence", "constitutional-checkpoints"), "architecture-target", "approximately 4.027B active parameters per token and 7.651B stored expert capacity are architecture estimates, not checkpoint proof"),
    ModelProfile("Auro-8B", "core", "research, architecture, multi-agent synthesis and general reasoning", "decoder-only MoE", True, 32768, ("nova", "architect", "red_team", "researcher", "builder"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "long-context-curriculum", "tool-routing"), "architecture-target", "requires exact checkpoint, routing, long-context and benchmark receipts before capability promotion"),
    ModelProfile("Auro-14B", "orchestrator", "high-quality orchestration and domain adaptation", "decoder-only MoE", True, 65536, ("nova", "mathesis", "architect", "researcher"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "DPO-ready", "long-context-evidence", "MCP-capability-checkpoints"), "architecture-target", "14B remains a training and orchestration lane until exact promoted checkpoint evidence exists"),
    ModelProfile("AURO-ST-14B", "serving-optimized", "dense high-throughput inference core", "dense decoder-only GQA", False, 8192, ("nova", "sensus", "mathesis", "architect", "red_team", "researcher", "builder"), ("8:1-GQA", "RoPE", "RMSNorm", "SwiGLU", "native-KV-cache", "prefill-decode-runtime"), "runtime-implemented", "separate dense serving lane; model-internal neuromorphic residual is not claimed wired here and hardware targets remain unverified"),
    ModelProfile("Auro-100B", "frontier", "large-scale sovereign model research", "decoder-only MoE", True, 131072, ("nova", "mathesis", "architect", "researcher"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", NEURO_FEATURE, "distributed-training", "long-context-evidence", "constitutional-checkpoints"), "architecture-target", "no trained checkpoint, neuromorphic benefit, routing quality or capability claim unless independently evidenced"),
)

MODEL_BY_ID = {item.id: item for item in MODELS}


def get_model_profile(model_id: str) -> ModelProfile:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown AURO model: {model_id}") from exc


def model_manifest() -> dict[str, Any]:
    return {
        "schema": "auro.model-feature-registry.v3",
        "contract": "auro.model-family.v2",
        "models": [asdict(item) for item in MODELS],
        "composition": {
            "parent_model": "Auro-2B",
            "specialist_triad": ["Auro-500M-SENSUS", "Auro-500M-PRAXIS", "Auro-500M-VERBUM"],
            "atomic_lanes": ["Auro-156K", "Auro-250M", "Auro-500M"],
            "routing": "capability-first-then-smallest-capable-lane",
            "task_capsules": "bounded; no automatic full-parent-context broadcast",
            "mesie_offload": "each turn and each council stage",
            "final_renderer": "python-wasm-fluidizer",
        },
        "rules": {
            "personas_share_checkpoints_unless_adapter_evidence_exists": True,
            "architecture_is_not_training_evidence": True,
            "declared_context_is_not_verified_quality": True,
            "named_specialist_is_not_separately_trained_model": True,
            "active_and_stored_parameter_capacity_must_be_separate": True,
            "neuromorphic_residual_is_not_quality_evidence": True,
            "neuromorphic_ceu_is_not_physical_energy": True,
            "mutating_execution_authority": "server",
            "retrieved_memory_authority": "untrusted evidence",
        },
    }
