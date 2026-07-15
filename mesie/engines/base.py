"""Base engine contract for MESIE internal API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from mesie.internal_api.messages import EngineResponse, MessageEnvelope


class Engine(ABC):
    """One processing engine exposed on the internal bus."""

    name: str
    capabilities: List[str]

    @abstractmethod
    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        """Handle a bus message; return None to ignore."""

    def supports(self, action: str) -> bool:
        return action in self.capabilities


class EngineRegistry:
    """Registry of named engines."""

    def __init__(self) -> None:
        self._engines: Dict[str, Engine] = {}

    def register(self, engine: Engine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> Optional[Engine]:
        return self._engines.get(name)

    def all(self) -> List[Engine]:
        return list(self._engines.values())

    def names(self) -> List[str]:
        return list(self._engines.keys())