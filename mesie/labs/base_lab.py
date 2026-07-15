"""Base class and registry for domain-specific lab environments."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class LabConfig:
    """Configuration for a lab environment.

    Attributes:
        name: Human-readable lab name.
        domain: Scientific domain (e.g., 'chemistry', 'physics').
        capabilities: List of operations this lab supports.
        max_concurrent: Maximum parallel operations.
    """

    name: str = ""
    domain: str = ""
    capabilities: List[str] = field(default_factory=list)
    max_concurrent: int = 4


@dataclass
class LabResult:
    """Result from a lab operation.

    Attributes:
        lab: Name of the lab that produced this result.
        operation: The operation that was performed.
        data: Result data payload.
        status: "success" or "error".
        error: Error message if status is "error".
        duration_seconds: Time taken.
    """

    lab: str
    operation: str
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lab": self.lab,
            "operation": self.operation,
            "data": self.data,
            "status": self.status,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


class BaseLab(ABC):
    """Abstract base class for all domain-specific labs.

    Subclasses implement domain-specific tools and operations while
    sharing a common interface for discovery and execution.

    Args:
        config: Lab configuration.
    """

    def __init__(self, config: Optional[LabConfig] = None) -> None:
        self._config = config or self._default_config()

    @abstractmethod
    def _default_config(self) -> LabConfig:
        """Return default configuration for this lab."""

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def domain(self) -> str:
        return self._config.domain

    @property
    def capabilities(self) -> List[str]:
        return self._config.capabilities

    @abstractmethod
    def run(self, operation: str, **kwargs: Any) -> LabResult:
        """Execute a lab operation.

        Args:
            operation: Name of the operation to perform.
            **kwargs: Operation-specific parameters.

        Returns:
            A LabResult with the operation outcome.
        """

    def supports(self, operation: str) -> bool:
        """Check if this lab supports the given operation."""
        return operation in self._config.capabilities

    def info(self) -> Dict[str, Any]:
        """Return metadata about this lab."""
        return {
            "name": self._config.name,
            "domain": self._config.domain,
            "capabilities": self._config.capabilities,
        }


class LabRegistry:
    """Registry of available lab environments.

    Provides discovery and access to all registered labs.
    """

    def __init__(self) -> None:
        self._labs: Dict[str, BaseLab] = {}

    def register(self, lab: BaseLab) -> None:
        """Register a lab instance."""
        self._labs[lab.domain] = lab

    def get(self, domain: str) -> Optional[BaseLab]:
        """Get lab by domain name."""
        return self._labs.get(domain)

    def list_labs(self) -> List[Dict[str, Any]]:
        """List all registered labs with their capabilities."""
        return [lab.info() for lab in self._labs.values()]

    def all_capabilities(self) -> Dict[str, List[str]]:
        """Map of domain → capabilities for all labs."""
        return {domain: lab.capabilities for domain, lab in self._labs.items()}

    @property
    def domains(self) -> List[str]:
        return list(self._labs.keys())


def build_default_lab_registry() -> LabRegistry:
    """Create a registry with all built-in labs."""
    from mesie.labs.spectral_lab import SpectralLab
    from mesie.labs.chemistry_lab import ChemistryLab
    from mesie.labs.physics_lab import PhysicsLab
    from mesie.labs.bio_lab import BioLab
    from mesie.labs.earth_lab import EarthLab

    registry = LabRegistry()
    registry.register(SpectralLab())
    registry.register(ChemistryLab())
    registry.register(PhysicsLab())
    registry.register(BioLab())
    registry.register(EarthLab())
    return registry
