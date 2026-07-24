"""Canonical separation between model lanes and NOVA agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NovaAgentSpec:
    id: str
    name: str
    role: str
    purpose: str
    capabilities: tuple[str, ...]
    artifact_types: tuple[str, ...] = ()
    requires_approval: bool = False

    def public(self) -> dict[str, Any]:
        return asdict(self)


NOVA_AGENT_FAMILY = (
    NovaAgentSpec("nova.sensus", "NOVA Sensus", "analysis", "Extract intent, constraints, evidence, and ambiguity.", ("memory.rank_text", "brain.state"), ("analysis",)),
    NovaAgentSpec("nova.mathesis", "NOVA Mathesis", "logic", "Check quantities, contradictions, proofs, and falsifiability.", ("compute.matmul", "brain.cycle"), ("verification",)),
    NovaAgentSpec("nova.architect", "NOVA Architect", "architecture", "Design coherent systems, interfaces, acceptance gates, and migration boundaries.", ("skill.reason", "skill.build"), ("specification", "architecture")),
    NovaAgentSpec("nova.hermes", "NOVA Hermes", "browser-worker", "Coordinate governed browser, web research, and external-information tasks.", ("browser.task.enqueue", "browser.task.status", "browser.tasks.list", "skill.research"), ("research", "source-map")),
    NovaAgentSpec("nova.forge", "NOVA Forge", "builder", "Build tested code and artifact bundles through bounded execution lanes.", ("build.create_session", "build.write_file", "build.run", "build.manifest", "office.create_bundle"), ("code", "bundle"), True),
    NovaAgentSpec("nova.engineer", "NOVA Engineer", "engine", "Exercise MESIE, compute, brain, and runtime engines and preserve receipts.", ("compute.engines", "compute.matmul", "brain.cycle", "brain.migration_status"), ("engine-report",)),
    NovaAgentSpec("nova.auditor", "NOVA Auditor", "critic", "Audit evidence, actions, receipts, safety boundaries, and unsupported claims.", ("brain.operator_snapshot", "wallet.verify_ledger", "build.manifest"), ("audit", "receipt-review")),
    NovaAgentSpec("nova.publisher", "NOVA Publisher", "publication", "Turn verified work into MD, CSV, DOCX, XLSX, PDF, and manifests.", ("office.create_bundle",), ("md", "csv", "docx", "xlsx", "pdf", "manifest"), True),
)


def model_agent_taxonomy(models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return the public contract that prevents model/agent identity inflation."""
    return {
        "schema": "nexus.model-agent-taxonomy.v1",
        "definitions": {
            "model": "A weight-bearing token generator or explicit inference endpoint.",
            "nova_agent": "A governed role using one explicit model lane, tools, context, policies, and receipts.",
            "tool": "A bounded capability contract. A tool is neither a model nor an agent.",
            "engine": "A deterministic or learned computation subsystem invoked through a capability boundary.",
        },
        "accounting_laws": [
            "Agent count never contributes to parameter count.",
            "A NOVA agent must report the model lane used for each task.",
            "Weights, checkpoint identity, and provider are properties of models, not agents.",
            "Tools and engines must emit separate execution receipts.",
            "No silent fallback may change model identity.",
        ],
        "models": list(models or []),
        "nova_agents": [x.public() for x in NOVA_AGENT_FAMILY],
    }
