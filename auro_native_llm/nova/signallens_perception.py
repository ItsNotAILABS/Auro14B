"""NOVA perception adapter for a deployed SignalLens Worker through NEXUS Relay.

No sensor simulation or direct SignalLens bypass is provided. The adapter is
ready only when an operator supplies the deployed Relay endpoint and credential.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib import error, request


@dataclass(frozen=True)
class SignalLensRelayConfig:
    relay_base_url: str
    api_key: str
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "SignalLensRelayConfig":
        return cls(
            relay_base_url=os.environ.get("NEXUS_RELAY_BASE_URL", ""),
            api_key=os.environ.get("NEXUS_RELAY_API_KEY", ""),
            timeout_seconds=float(os.environ.get("NEXUS_RELAY_TIMEOUT_SECONDS", "30")),
        )

    @property
    def configured(self) -> bool:
        return self.relay_base_url.startswith("https://") and bool(self.api_key)


class SignalLensRelayPerception:
    def __init__(
        self,
        config: SignalLensRelayConfig | None = None,
        transport: Callable[[request.Request, float], bytes] | None = None,
    ) -> None:
        self.config = config or SignalLensRelayConfig.from_environment()
        self.transport = transport or self._urlopen

    @staticmethod
    def _urlopen(req: request.Request, timeout: float) -> bytes:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()

    def health(self) -> dict[str, Any]:
        return {
            "schema": "nova.signallens-relay.health.v1",
            "configured": self.config.configured,
            "relay_base_url": self.config.relay_base_url if self.config.configured else "",
            "direct_signallens_access": False,
            "simulation_fallback": False,
            "cross_repository_deployment_verified": False,
            "claim_boundary": "configuration proves wiring only; deployed SignalLens Worker health requires a successful signed Relay response",
        }

    def perceive(self, source_url: str, *, session_id: str, approval_receipt: Mapping[str, Any]) -> dict[str, Any]:
        if not self.config.configured:
            raise RuntimeError("NEXUS Relay perception is not configured")
        if not session_id:
            raise PermissionError("durable NOVA session_id required")
        if not approval_receipt.get("authorized") or len(str(approval_receipt.get("action_sha256") or "")) != 64:
            raise PermissionError("consumed action-bound approval receipt required")
        endpoint = self.config.relay_base_url.rstrip("/") + "/v1/read"
        body = json.dumps({
            "url": source_url,
            "consumer": "nova-signallens-perception",
            "session_id": session_id,
            "approval": dict(approval_receipt),
        }, sort_keys=True).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "authorization": f"Bearer {self.config.api_key}",
                "content-type": "application/json",
                "accept": "application/json",
            },
        )
        try:
            raw = self.transport(req, self.config.timeout_seconds)
        except error.HTTPError as exc:
            raise RuntimeError(f"NEXUS Relay rejected perception request: HTTP {exc.code}") from exc
        payload = json.loads(raw.decode("utf-8"))
        receipt = dict(payload.get("receipt") or {})
        content = str(payload.get("text") or payload.get("content") or "")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if receipt.get("content_sha256") != expected:
            raise ValueError("SignalLens Relay content hash mismatch")
        if not receipt.get("receipt_sha256") or not receipt.get("request_id"):
            raise ValueError("SignalLens Relay custody receipt is incomplete")
        return {
            "schema": "nova.signallens-relay.perception.v1",
            "session_id": session_id,
            "source_url": source_url,
            "content": content,
            "intelligence": dict(payload.get("intelligence") or {}),
            "relay_receipt": receipt,
            "cross_repository_deployment_verified": True,
            "simulation": False,
        }
