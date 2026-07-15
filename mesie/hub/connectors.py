"""Connectors — adapters for external tools (Jupyter, CLI, WebSocket)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class BaseConnector(ABC):
    """Abstract base class for hub connectors.

    Connectors allow external tools and environments to interact
    with the MESIE research hub.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Connector name."""

    @property
    @abstractmethod
    def protocol(self) -> str:
        """Communication protocol (http, ws, stdio, etc.)."""

    @abstractmethod
    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message/request through this connector."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the connector is active."""


class JupyterConnector(BaseConnector):
    """Connector for Jupyter notebook integration.

    Enables MESIE hub interaction from Jupyter cells via
    direct Python object passing.
    """

    def __init__(self, hub_callback: Optional[Callable[..., Any]] = None) -> None:
        self._callback = hub_callback
        self._connected = True

    @property
    def name(self) -> str:
        return "jupyter"

    @property
    def protocol(self) -> str:
        return "direct"

    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if self._callback:
            return self._callback(message)
        return {"status": "ok", "note": "No hub callback registered"}

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False


class CLIConnector(BaseConnector):
    """Connector for command-line interface access."""

    def __init__(self) -> None:
        self._connected = True

    @property
    def name(self) -> str:
        return "cli"

    @property
    def protocol(self) -> str:
        return "stdio"

    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        action = message.get("action", "")
        params = message.get("params", {})
        return {
            "status": "dispatched",
            "action": action,
            "params": params,
        }

    def is_connected(self) -> bool:
        return self._connected


class WebSocketConnector(BaseConnector):
    """Connector for WebSocket-based real-time communication."""

    def __init__(self, url: str = "ws://localhost:8765") -> None:
        self._url = url
        self._connected = False
        self._message_queue: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "websocket"

    @property
    def protocol(self) -> str:
        return "ws"

    @property
    def url(self) -> str:
        return self._url

    def connect(self) -> None:
        """Establish WebSocket connection (placeholder for async impl)."""
        self._connected = True

    def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not self._connected:
            return {"status": "error", "error": "Not connected"}
        self._message_queue.append(message)
        return {"status": "queued", "queue_size": len(self._message_queue)}

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        self._connected = False
        self._message_queue.clear()


class ConnectorRegistry:
    """Registry of available connectors."""

    def __init__(self) -> None:
        self._connectors: Dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
        self._connectors[connector.name] = connector

    def get(self, name: str) -> Optional[BaseConnector]:
        return self._connectors.get(name)

    def list_connectors(self) -> List[Dict[str, str]]:
        return [
            {"name": c.name, "protocol": c.protocol, "connected": str(c.is_connected())}
            for c in self._connectors.values()
        ]

    @property
    def names(self) -> List[str]:
        return list(self._connectors.keys())
