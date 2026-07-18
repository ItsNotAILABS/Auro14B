"""GHOST — grounded, hardened, open, scalable, traceable execution for Auro.

Pillars (ItsNotAILabs / LoomMultiAI):
  G — Grounded Intelligence
  H — Hardened Operation
  O — Open and Auditable Architecture
  S — Scalable, Local-First Infrastructure
  T — Traceable Receipts and Chain of Custody

Hybrid: MESIE Ghost Node (deterministic spectral/math) + LLM only on escalation.
"""

from auro_native_llm.ghost.pillars import GhostPillar, ClaimKind, label_claim
from auro_native_llm.ghost.receipts import GhostReceiptChain, GhostReceipt
from auro_native_llm.ghost.policy import RiskClass, PolicyGate, PolicyDecision
from auro_native_llm.ghost.agent import GhostAgentRuntime, GhostTask, GhostOutcome
from auro_native_llm.ghost.node import MesieGhostNode
from auro_native_llm.ghost.supervisor import GhostSupervisor, run_ghost_intent

__all__ = [
    "GhostPillar",
    "ClaimKind",
    "label_claim",
    "GhostReceiptChain",
    "GhostReceipt",
    "RiskClass",
    "PolicyGate",
    "PolicyDecision",
    "GhostAgentRuntime",
    "GhostTask",
    "GhostOutcome",
    "MesieGhostNode",
    "GhostSupervisor",
    "run_ghost_intent",
]
