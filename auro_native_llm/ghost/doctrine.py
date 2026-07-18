"""GHOST training doctrine blocks for specialization / absorb."""

from __future__ import annotations

from typing import List

from auro_native_llm.ghost.pillars import GHOST_DOCTRINE

EXTRA_BLOCKS: List[str] = [
    "GHOST Ghost Agents are lightweight autonomous embodiments of computation. "
    "They receive TaskSpec, activate engines on an internal bus, chain results, "
    "and deliver structured output — not heavy supervisory layers.",
    "GHOST Hybrid Architecture: MESIE/Ghost layer handles precise fast deterministic "
    "numerical/signal work (filtering, frequency features, shadow execution, streaming). "
    "LLM used only for higher-level reasoning, planning, NL, or justified escalation.",
    "GHOST Benefits: faster/cheaper skip LLM most steps; more reliable/auditable "
    "deterministic math; reduced hallucinations; edge low-latency; binary envelope fit.",
    "GHOST MESIE Virtual Processor: prompts are measurable work calls (bytes, NOVA cycles, "
    "entropy, spectral buckets, coherence, routing, receipts, hash-chain state).",
    "GHOST MONDAY-style supervision: intent compilation → provenance → risk/policy gating "
    "Class0..Class5 → plan → Autonomeon execution boundary (local models, tools, ghosts, "
    "MESIE) → validation → receipt engine. Haunt Detector for orphaned behaviors.",
    "GHOST Persona separation: persona never influences permissions or evidence requirements.",
    "GHOST Use cases: HFT/trading audit trails, sensor processing, edge swarms, coding agents, "
    "scientific spectral workflows, regulated environments needing determinism.",
    "GHOST Philosophy: counterpoint to pure monolithic LLM scaling — hybrid determinism, "
    "agentic lightness, verifiability, local sovereignty.",
]


def all_ghost_training_blocks() -> List[str]:
    return list(GHOST_DOCTRINE) + list(EXTRA_BLOCKS)
