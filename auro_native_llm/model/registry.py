"""Canonical AURO model, feature, and claim-boundary registry."""
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


MODELS: tuple[ModelProfile, ...] = (
    ModelProfile("HIM-native-v0", "pipeline-fixture", "CPU pipeline and custody validation", "context-MLP causal LM", False, 16, ("memory_keeper",), ("byte-tokenizer", "local-open-weights", "hash-verified-loader"), "fixture-only", "proves pipeline mechanics, not assistant quality or mature HIM intelligence"),
    ModelProfile("Auro-156K", "micro", "architecture and routing experiments", "decoder-only MoE", True, 1024, ("sensus", "memory_keeper"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", "constitutional-checkpoints"), "architecture-target", "no trained quality claim without exact checkpoint evidence"),
    ModelProfile("Auro-2B", "edge", "local assistants, memory, lightweight browser planning", "decoder-only MoE", True, 8192, ("sensus", "operator", "memory_keeper", "browser_brain"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", "long-context-curriculum", "checkpoint-receipts"), "architecture-target", "declared context and routing are not quality claims"),
    ModelProfile("Auro-4B", "specialist", "tool use, coding, quantitative work, browser agents", "decoder-only MoE with structured residuals", True, 65536, ("mathesis", "architect", "red_team", "operator", "researcher", "builder", "browser_brain"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", "Walsh-Hadamard", "long-context-evidence", "constitutional-checkpoints"), "architecture-target", "active compute and stored expert capacity must be reported separately"),
    ModelProfile("Auro-8B", "general", "research, architecture, multi-agent synthesis", "decoder-only MoE", True, 32768, ("nova", "architect", "red_team", "researcher", "builder"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", "long-context-curriculum", "tool-routing"), "architecture-target", "requires exact checkpoint and benchmark receipts"),
    ModelProfile("Auro-14B", "general-high", "high-quality orchestration and domain adaptation", "decoder-only MoE", True, 65536, ("nova", "mathesis", "architect", "researcher"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", "DPO-ready", "long-context-evidence", "MCP-capability-checkpoints"), "architecture-target", "alignment and hallucination resistance are evaluation claims, not architecture facts"),
    ModelProfile("AURO-ST-14B", "serving-optimized", "dense high-throughput inference core", "dense decoder-only GQA", False, 8192, ("nova", "sensus", "mathesis", "architect", "red_team", "researcher", "builder"), ("8:1-GQA", "RoPE", "RMSNorm", "SwiGLU", "native-KV-cache", "prefill-decode-runtime"), "runtime-implemented", "hardware performance targets remain unverified without exact checkpoint and H100 telemetry"),
    ModelProfile("Auro-100B", "frontier", "large-scale sovereign model research", "decoder-only MoE", True, 131072, ("nova", "mathesis", "architect", "researcher"), ("GQA", "RoPE", "RMSNorm", "SwiGLU", "MoE", "distributed-training", "long-context-evidence", "constitutional-checkpoints"), "architecture-target", "no trained checkpoint or capability claim unless independently evidenced"),
)

MODEL_BY_ID = {item.id: item for item in MODELS}


def get_model_profile(model_id: str) -> ModelProfile:
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown AURO model: {model_id}") from exc


def model_manifest() -> dict[str, Any]:
    return {
        "schema": "auro.model-feature-registry.v1",
        "models": [asdict(item) for item in MODELS],
        "rules": {
            "personas_share_checkpoints": True,
            "architecture_is_not_training_evidence": True,
            "declared_context_is_not_verified_quality": True,
            "mutating_execution_authority": "server",
            "retrieved_memory_authority": "untrusted evidence",
        },
    }
