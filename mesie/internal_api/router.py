"""Route internal API calls to registered engines."""

from __future__ import annotations

from typing import Dict, Optional

from mesie.internal_api.bus import InternalBus
from mesie.internal_api.messages import EngineResponse, MessageEnvelope, MessageTopic
from mesie.engines.base import EngineRegistry


class InternalRouter:
    """Facade: resolves engine name → bus request."""

    def __init__(
        self,
        bus: Optional[InternalBus] = None,
        registry: Optional[EngineRegistry] = None,
    ) -> None:
        self.bus = bus or InternalBus()
        if registry is None:
            from mesie.engines.registry import build_default_registry

            registry = build_default_registry(self.bus)
        self.registry = registry
        self._wire_registry()

    def _wire_registry(self) -> None:
        for engine in self.registry.all():
            self.bus.register_engine(engine.name, engine.handle)

    def call(
        self,
        engine_name: str,
        action: str,
        payload: Optional[Dict] = None,
        *,
        source: str = "router",
    ) -> EngineResponse:
        return self.bus.request(source, engine_name, action, payload or {})

    def broadcast(self, action: str, payload: Optional[Dict] = None, *, source: str = "router") -> list:
        msg = MessageEnvelope(
            topic=MessageTopic.BROADCAST,
            source=source,
            target="*",
            action=action,
            payload=payload or {},
        )
        return self.bus.publish(msg)