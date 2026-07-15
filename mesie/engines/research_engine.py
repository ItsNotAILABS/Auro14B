"""Research Engine — exposes the Research Agent on the internal bus."""

from __future__ import annotations

from typing import Dict, List, Optional

from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.research.agent import ResearchAgent, ResearchConfig


class ResearchEngine(Engine):
    """Engine that integrates the Research Agent with the internal bus.

    Handles research requests, thesis pipelines, and planner queries
    routed via the InternalBus message system.

    Supported actions:
        - research: Run a research workflow
        - thesis: Run thesis-mode research
        - plan: Generate a research plan without executing
        - templates: List available planner templates
    """

    name = "research"
    capabilities = ["research", "thesis", "plan", "templates"]

    def __init__(self, agent: Optional[ResearchAgent] = None) -> None:
        self._agent = agent or ResearchAgent()

    @property
    def agent(self) -> ResearchAgent:
        return self._agent

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        action = message.action
        payload = message.payload

        if action == "research":
            question = payload.get("question", "")
            template = payload.get("template")
            result = self._agent.research(question, template=template)
            return EngineResponse(
                ok=True,
                engine=self.name,
                action=action,
                data=result.to_dict(),
            )

        elif action == "thesis":
            title = payload.get("title", "")
            hypothesis = payload.get("hypothesis", "")
            result = self._agent.thesis(title=title, hypothesis=hypothesis)
            return EngineResponse(
                ok=True,
                engine=self.name,
                action=action,
                data=result.to_dict(),
            )

        elif action == "plan":
            question = payload.get("question", "")
            template = payload.get("template")
            graph = self._agent._planner.plan(question, template=template)
            return EngineResponse(
                ok=True,
                engine=self.name,
                action=action,
                data={
                    "tasks": [
                        {"name": t.name, "tool": t.tool, "action": t.action}
                        for t in graph.tasks
                    ]
                },
            )

        elif action == "templates":
            return EngineResponse(
                ok=True,
                engine=self.name,
                action=action,
                data={"templates": self._agent._planner.templates},
            )

        return None
