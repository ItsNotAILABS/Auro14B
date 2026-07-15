"""UniversalLabSDK — unified interface to all MESIE labs, research agent, and data sources.

Provides a single high-level API for multi-domain research.

Example:
    >>> from mesie.sdk.universal_lab_sdk import UniversalLabSDK
    >>> sdk = UniversalLabSDK()
    >>> sdk.research("What is the spectral signature of steel?")
    >>> sdk.lab("chemistry").run("molecular_fingerprint", smiles="CCO")
    >>> sdk.lab("physics").run("constants", name="c")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mesie.data_sources.registry import DataSourceRegistry, build_default_sources
from mesie.hub.server import ResearchHub, HubConfig
from mesie.labs.base_lab import BaseLab, LabRegistry, build_default_lab_registry, LabResult
from mesie.research.agent import ResearchAgent, ResearchConfig, ResearchResult
from mesie.research.thesis_engine import ThesisEngine, ThesisResult


class UniversalLabSDK:
    """Unified SDK wrapping all research labs, agents, and data sources.

    One interface for everything: spectral analysis, chemistry, physics,
    biology, earth science, research orchestration, and external data.

    Args:
        research_config: Optional research agent configuration.
        hub_config: Optional hub server configuration.

    Example:
        >>> sdk = UniversalLabSDK()
        >>> # Run a research workflow
        >>> result = sdk.research("Properties of graphene oxide")
        >>> # Use a specific lab
        >>> physics = sdk.lab("physics")
        >>> physics.run("blackbody", temperature=5778)
        >>> # Access data sources
        >>> sdk.data_source("arxiv").search("quantum computing")
    """

    def __init__(
        self,
        research_config: Optional[ResearchConfig] = None,
        hub_config: Optional[HubConfig] = None,
    ) -> None:
        self._lab_registry = build_default_lab_registry()
        self._data_sources = build_default_sources()
        self._agent = ResearchAgent(config=research_config)
        self._thesis_engine = ThesisEngine()
        self._hub = ResearchHub(
            config=hub_config,
            lab_registry=self._lab_registry,
            research_agent=self._agent,
        )
        self._hub.start()

    # ------------------------------------------------------------------
    # Labs
    # ------------------------------------------------------------------

    def lab(self, domain: str) -> BaseLab:
        """Get a lab by domain name.

        Args:
            domain: Lab domain ('spectral', 'chemistry', 'physics', 'bio', 'earth').

        Returns:
            The lab instance.

        Raises:
            KeyError: If domain not found.
        """
        lab_instance = self._lab_registry.get(domain)
        if lab_instance is None:
            available = self._lab_registry.domains
            raise KeyError(
                f"Lab '{domain}' not found. Available: {available}"
            )
        return lab_instance

    def list_labs(self) -> List[Dict[str, Any]]:
        """List all available labs and their capabilities."""
        return self._lab_registry.list_labs()

    def run_lab(self, domain: str, operation: str, **kwargs: Any) -> LabResult:
        """Convenience: run a lab operation directly.

        Args:
            domain: Lab domain.
            operation: Operation name.
            **kwargs: Operation parameters.

        Returns:
            Lab operation result.
        """
        return self.lab(domain).run(operation, **kwargs)

    # ------------------------------------------------------------------
    # Research Agent
    # ------------------------------------------------------------------

    def research(
        self,
        question: str,
        *,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        """Run an autonomous research workflow.

        Args:
            question: The research question to investigate.
            template: Optional planner template.
            context: Additional context.

        Returns:
            Research result with task outcomes and report.
        """
        return self._agent.research(question, template=template, context=context)

    def thesis(
        self,
        title: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> ResearchResult:
        """Run thesis-mode research (hypothesis → conclusion).

        Args:
            title: Thesis title.
            hypothesis: Hypothesis to test.

        Returns:
            Research result with thesis outcome.
        """
        return self._agent.thesis(title=title, hypothesis=hypothesis, **kwargs)

    # ------------------------------------------------------------------
    # Data Sources
    # ------------------------------------------------------------------

    def data_source(self, name: str) -> Any:
        """Get a data source by name.

        Args:
            name: Source name ('arxiv', 'pubchem', 'usgs').

        Returns:
            The data source connector.

        Raises:
            KeyError: If source not found.
        """
        source = self._data_sources.get(name)
        if source is None:
            available = self._data_sources.names
            raise KeyError(
                f"Data source '{name}' not found. Available: {available}"
            )
        return source

    def list_data_sources(self) -> List[Dict[str, str]]:
        """List all available data sources."""
        return self._data_sources.list_sources()

    # ------------------------------------------------------------------
    # Hub access
    # ------------------------------------------------------------------

    @property
    def hub(self) -> ResearchHub:
        """Access the underlying ResearchHub instance."""
        return self._hub

    def info(self) -> Dict[str, Any]:
        """SDK overview: labs, sources, and status."""
        return {
            "sdk": "UniversalLabSDK",
            "labs": self._lab_registry.domains,
            "data_sources": self._data_sources.names,
            "hub": self._hub.info(),
            "research_history": len(self._agent.history),
        }
