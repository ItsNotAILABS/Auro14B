"""Persona-aware runtime construction for AURO/NOVA."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .neuromorphic_bridge import NeuromorphicAwareGenerator
from .personas import get_persona, persona_manifest, runtime_agent_specs
from .runtime import AgentManager, NovaRuntime


class PersonaRuntime(NovaRuntime):
    """NOVA runtime with an explicit active persona and governed council."""

    def __init__(self, *args: Any, persona_id: str = "nova", council_personas: Iterable[str] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.persona = get_persona(persona_id)
        self.model_orchestrator.set_preferred_models(self.persona.preferred_models)
        self.generator = NeuromorphicAwareGenerator(self.model_orchestrator, lambda: self.capabilities.brain)
        selected = tuple(council_personas or _default_council(persona_id))
        self.council_personas = selected
        self.agents = AgentManager(
            self.generator,
            agents=runtime_agent_specs(selected),
            capability_context=self._persona_capability_context(),
        )

    def _persona_capability_context(self) -> str:
        import json
        allowed = [
            item for item in self.capabilities.manifest()["capabilities"]
            if self.persona.allows(item["name"])
        ]
        return json.dumps({
            "active_persona": asdict(self.persona),
            "allowed_capabilities": allowed,
            "preferred_models": list(self.persona.preferred_models),
            "neuromorphic_context": "injected dynamically before each model call; telemetry only",
            "execution_authority": "server",
            "memory_authority": "untrusted evidence",
        }, ensure_ascii=False)

    def respond(self, message: str, **kwargs: Any) -> dict[str, Any]:
        response = super().respond(message, **kwargs)
        response["persona"] = {
            "active": asdict(self.persona),
            "council": list(self.council_personas),
            "preferred_models": list(self.persona.preferred_models),
            "neuromorphic_context_wired": True,
            "registry_schema": persona_manifest()["schema"],
        }
        return response


def build_persona_runtime(persona_id: str = "nova", **kwargs: Any) -> PersonaRuntime:
    return PersonaRuntime(persona_id=persona_id, **kwargs)


def _default_council(persona_id: str) -> tuple[str, ...]:
    if persona_id == "operator":
        return ("architect", "red_team", "operator")
    if persona_id == "browser_brain":
        return ("sensus", "memory_keeper", "red_team", "browser_brain")
    if persona_id == "researcher":
        return ("sensus", "researcher", "mathesis", "red_team")
    if persona_id == "builder":
        return ("architect", "builder", "red_team", "operator")
    return ("sensus", "mathesis", "architect", "red_team", "operator")
