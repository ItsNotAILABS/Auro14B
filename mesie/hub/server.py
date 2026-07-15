"""Research Hub server — central API for labs, tools, and the research agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mesie.hub.connectors import ConnectorRegistry
from mesie.hub.schema import HubSchema
from mesie.hub.session import HubSession, SessionManager
from mesie.labs.base_lab import BaseLab, LabRegistry, build_default_lab_registry
from mesie.research.agent import ResearchAgent, ResearchConfig, ResearchResult


@dataclass
class HubConfig:
    """Configuration for the Research Hub.

    Attributes:
        host: Server bind address.
        port: Server port.
        max_sessions: Maximum concurrent sessions.
        enable_cors: Enable CORS for web clients.
        research_config: Configuration for the built-in research agent.
    """

    host: str = "0.0.0.0"
    port: int = 8765
    max_sessions: int = 100
    enable_cors: bool = True
    research_config: Optional[ResearchConfig] = None


class ResearchHub:
    """Central hub that exposes all MESIE labs, tools, and research capabilities.

    The ResearchHub acts as a unified gateway. External clients (notebooks,
    CLI, web UIs) connect through connectors and can:
    - Execute lab operations across any domain
    - Run research agent workflows
    - Discover available tools and schemas
    - Maintain session state

    Args:
        config: Hub configuration.
        lab_registry: Optional pre-built lab registry.
        research_agent: Optional pre-configured research agent.

    Example:
        >>> hub = ResearchHub()
        >>> session = hub.create_session(user="scientist")
        >>> result = hub.run_lab("chemistry", "molecular_fingerprint", smiles="CCO")
        >>> research = hub.research("What are the properties of graphene?")
    """

    def __init__(
        self,
        config: Optional[HubConfig] = None,
        lab_registry: Optional[LabRegistry] = None,
        research_agent: Optional[ResearchAgent] = None,
    ) -> None:
        self._config = config or HubConfig()
        self._labs = lab_registry or build_default_lab_registry()
        self._agent = research_agent or ResearchAgent(
            config=self._config.research_config
        )
        self._sessions = SessionManager(max_sessions=self._config.max_sessions)
        self._schema = HubSchema()
        self._connectors = ConnectorRegistry()
        self._started_at: Optional[float] = None

    @property
    def config(self) -> HubConfig:
        return self._config

    @property
    def labs(self) -> LabRegistry:
        return self._labs

    @property
    def agent(self) -> ResearchAgent:
        return self._agent

    @property
    def schema(self) -> HubSchema:
        return self._schema

    @property
    def connectors(self) -> ConnectorRegistry:
        return self._connectors

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, user: Optional[str] = None) -> HubSession:
        """Create a new client session."""
        return self._sessions.create(user=user)

    def get_session(self, session_id: str) -> Optional[HubSession]:
        """Retrieve an active session."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return self._sessions.list_active()

    # ------------------------------------------------------------------
    # Lab operations
    # ------------------------------------------------------------------

    def run_lab(
        self, domain: str, operation: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute a lab operation.

        Args:
            domain: Lab domain (e.g., 'chemistry', 'physics').
            operation: Operation name within that lab.
            session_id: Optional session to track the action.
            **kwargs: Operation parameters.

        Returns:
            Lab result as a dictionary.
        """
        lab = self._labs.get(domain)
        if lab is None:
            return {"status": "error", "error": f"Lab '{domain}' not found"}

        result = lab.run(operation, **kwargs)

        if session_id:
            session = self._sessions.get(session_id)
            if session:
                session.record_action(
                    f"lab.{domain}.{operation}", kwargs, result.to_dict()
                )

        return result.to_dict()

    def list_labs(self) -> List[Dict[str, Any]]:
        """List all available labs and their capabilities."""
        return self._labs.list_labs()

    # ------------------------------------------------------------------
    # Research agent
    # ------------------------------------------------------------------

    def research(
        self,
        question: str,
        *,
        template: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a research workflow.

        Args:
            question: Research question.
            template: Optional planner template.
            session_id: Optional session to track.

        Returns:
            Research result dictionary.
        """
        result = self._agent.research(question, template=template, context=kwargs)

        if session_id:
            session = self._sessions.get(session_id)
            if session:
                session.record_action(
                    "research", {"question": question}, result.to_dict()
                )

        return result.to_dict()

    def thesis(
        self,
        title: str,
        hypothesis: str,
        *,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run thesis-mode research.

        Args:
            title: Thesis title.
            hypothesis: Hypothesis to test.
            session_id: Optional session.

        Returns:
            Thesis result dictionary.
        """
        result = self._agent.thesis(title=title, hypothesis=hypothesis, **kwargs)

        if session_id:
            session = self._sessions.get(session_id)
            if session:
                session.record_action(
                    "thesis", {"title": title, "hypothesis": hypothesis},
                    result.to_dict(),
                )

        return result.to_dict()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Hub metadata and status."""
        return {
            "name": "MESIE Research Hub",
            "version": "1.0.0",
            "labs": self._labs.domains,
            "active_sessions": self._sessions.active_count,
            "connectors": self._connectors.names,
            "uptime_seconds": (
                time.time() - self._started_at if self._started_at else 0.0
            ),
        }

    def start(self) -> None:
        """Mark hub as started (for embedded/non-HTTP use)."""
        self._started_at = time.time()

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a generic request (protocol-agnostic dispatch).

        Routes to labs, research, or info based on request type.

        Args:
            request: Request dict with 'type' and relevant params.

        Returns:
            Response dictionary.
        """
        req_type = request.get("type", "")
        params = request.get("params", {})
        session_id = request.get("session_id")

        if req_type == "lab":
            domain = params.pop("domain", "")
            operation = params.pop("operation", "")
            return self.run_lab(domain, operation, session_id=session_id, **params)
        elif req_type == "research":
            question = params.pop("question", "")
            return self.research(question, session_id=session_id, **params)
        elif req_type == "thesis":
            title = params.pop("title", "")
            hypothesis = params.pop("hypothesis", "")
            return self.thesis(title, hypothesis, session_id=session_id, **params)
        elif req_type == "info":
            return self.info()
        elif req_type == "list_labs":
            return {"labs": self.list_labs()}
        elif req_type == "list_sessions":
            return {"sessions": self.list_sessions()}
        else:
            return {"status": "error", "error": f"Unknown request type: {req_type}"}
