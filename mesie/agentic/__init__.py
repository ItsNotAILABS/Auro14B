"""Agentic embodiment system — ghost agents that perform computational work.

Each GhostAgent is a lightweight coroutine-friendly unit that can:
- Receive a task specification
- Activate one or more engines via the internal bus
- Chain results across multiple processing steps
- Report completion with structured output

This enables MESIE to do *actual work* — not just match spectra, but
autonomously run analysis pipelines, generate reports, search corpora,
and reason over findings as independent "ghosts in the machine."
"""

from mesie.agentic.ghost import (
    AgentState,
    GhostAgent,
    GhostConfig,
    GhostResult,
    TaskSpec,
)
from mesie.agentic.spawner import AgentSpawner, SpawnerConfig
from mesie.agentic.network import AgentNetwork, NetworkTopology

__all__ = [
    "AgentNetwork",
    "AgentSpawner",
    "AgentState",
    "GhostAgent",
    "GhostConfig",
    "GhostResult",
    "NetworkTopology",
    "SpawnerConfig",
    "TaskSpec",
]
