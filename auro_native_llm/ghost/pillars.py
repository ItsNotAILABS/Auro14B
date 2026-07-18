"""GHOST pillars — claim grounding and doctrine constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class GhostPillar(str, Enum):
    GROUNDED = "grounded_intelligence"
    HARDENED = "hardened_operation"
    OPEN = "open_auditable"
    SCALABLE = "scalable_local_first"
    TRACEABLE = "traceable_receipts"


class ClaimKind(str, Enum):
    """Grounded Intelligence: every material claim has a kind."""

    OPERATOR_FACT = "operator_fact"  # human-supplied fact
    RETRIEVED_EVIDENCE = "retrieved_evidence"  # corpus / tool evidence
    SYSTEM_OBSERVATION = "system_observation"  # runtime metrics
    MODEL_INFERENCE = "model_inference"  # LLM or generative guess
    UNRESOLVED = "unresolved_assumption"  # explicit uncertainty
    DETERMINISTIC_MESIE = "deterministic_mesie"  # spectral/math engine result


@dataclass
class LabeledClaim:
    text: str
    kind: ClaimKind
    evidence_ref: Optional[str] = None
    uncertainty: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind.value,
            "evidence_ref": self.evidence_ref,
            "uncertainty": self.uncertainty,
        }


def label_claim(
    text: str,
    kind: ClaimKind | str,
    *,
    evidence_ref: Optional[str] = None,
    uncertainty: Optional[str] = None,
) -> LabeledClaim:
    k = ClaimKind(kind) if isinstance(kind, str) else kind
    if k in (ClaimKind.RETRIEVED_EVIDENCE, ClaimKind.DETERMINISTIC_MESIE) and not evidence_ref:
        evidence_ref = "missing-evidence-ref"
    if k == ClaimKind.UNRESOLVED and not uncertainty:
        uncertainty = "unverified"
    if k == ClaimKind.MODEL_INFERENCE and not uncertainty:
        uncertainty = "model-inferred-not-attested"
    return LabeledClaim(text=text, kind=k, evidence_ref=evidence_ref, uncertainty=uncertainty)


GHOST_DOCTRINE: List[str] = [
    "GHOST G — Grounded Intelligence: distinguish operator facts, retrieved evidence, "
    "system observations, model inferences, unresolved assumptions. Every material claim "
    "gets evidence_ref or explicit uncertainty.",
    "GHOST H — Hardened Operation: least-privilege, explicit scopes, isolated permissions, "
    "resource/time limits, policy gates, human approval for irreversible actions. "
    "Persona layers cannot weaken controls.",
    "GHOST O — Open and Auditable: emit inspectable records (intent, policy, evidence, "
    "tool I/O, validation, hashes) — not opaque assurances. Support replay and attestation.",
    "GHOST S — Scalable Local-First: prefer local workstations, model clusters, federated "
    "agents, offline/degraded modes. Remote services only behind explicit gateways.",
    "GHOST T — Traceable Receipts: linked receipt chain answers who initiated, what policy, "
    "which tools/models, what changed, how validated, can it be replayed/reversed.",
    "GHOST Hybrid: MESIE Ghost Node does deterministic spectral/math work first; "
    "LLM escalates only for language/planning/strategy when justified by cleaned signals.",
    "GHOST Ghost Agents: lightweight embodiments of computation — receive TaskSpec, "
    "activate engines on internal bus, chain results, deliver structured output.",
    "GHOST Anti-haunt: no unexplained/orphaned behaviors — Haunt Detector flags unaccountable acts.",
    "GHOST Sovereignty: edge deployment, minimal external deps, operator oversight, "
    "counterpoint to pure monolithic LLM scaling.",
    "GHOST Flow: Human intent → Supervisor → task envelope + risk class + policy → "
    "Ghost Agents/MESIE (+ LLM if escalate) → validation → receipt chain → auditable result.",
]


def pillars_health() -> Dict[str, Any]:
    return {
        "schema": "auro.ghost.pillars.v1",
        "pillars": [p.value for p in GhostPillar],
        "claim_kinds": [c.value for c in ClaimKind],
        "doctrine_n": len(GHOST_DOCTRINE),
        "ecosystem": ["ItsNotAILabs", "LoomMultiAI", "MESIE", "NOVA"],
    }
