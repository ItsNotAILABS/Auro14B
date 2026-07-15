"""Third-party AI connector — OpenAI-style tools, webhooks, generic REST agents."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mesie.polyglot.contract import (
    AISVectorMessage,
    AISVectorResponse,
    PolyglotAction,
    RuntimeId,
    SUITE_NAME,
    record_to_dict,
)


@dataclass
class ThirdPartyTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class ThirdPartyConfig:
    webhook_url: str = ""
    api_key: str = ""
    timeout_s: float = 15.0
    user_agent: str = f"{SUITE_NAME}/1.0"


class ThirdPartyAIConnector:
    """Expose MESIE spectral ops as tools for external AI agents (OpenAI, Anthropic, custom)."""

    def __init__(self, suite_dispatch: Callable[[AISVectorMessage], AISVectorResponse]) -> None:
        self._dispatch = suite_dispatch
        self.config = ThirdPartyConfig()
        self._tools = self._build_tools()

    def _build_tools(self) -> List[ThirdPartyTool]:
        def _validate(args: dict) -> dict:
            msg = AISVectorMessage(
                action=PolyglotAction.VALIDATE,
                runtime=RuntimeId.PYTHON,
                record=args.get("record"),
            )
            return self._dispatch(msg).to_dict()

        def _match(args: dict) -> dict:
            msg = AISVectorMessage(
                action=PolyglotAction.MATCH,
                runtime=RuntimeId.PYTHON,
                record_a=args.get("reference"),
                record_b=args.get("candidate"),
            )
            return self._dispatch(msg).to_dict()

        def _embed(args: dict) -> dict:
            msg = AISVectorMessage(
                action=PolyglotAction.EMBED,
                runtime=RuntimeId.PYTHON,
                record=args.get("record"),
            )
            return self._dispatch(msg).to_dict()

        return [
            ThirdPartyTool(
                "mesie_validate_spectrum",
                "Validate a spectral JSON record against MESIE schema.",
                {"type": "object", "properties": {"record": {"type": "object"}}},
                _validate,
            ),
            ThirdPartyTool(
                "mesie_match_spectra",
                "Compare two spectral records and return composite similarity.",
                {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "object"},
                        "candidate": {"type": "object"},
                    },
                },
                _match,
            ),
            ThirdPartyTool(
                "mesie_embed_spectrum",
                "Vectorize a spectral record for ANN retrieval.",
                {"type": "object", "properties": {"record": {"type": "object"}}},
                _embed,
            ),
        ]

    def openai_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools
        ]

    def invoke_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        for t in self._tools:
            if t.name == name:
                return t.handler(arguments)
        return {"ok": False, "error": f"unknown tool: {name}"}

    def post_webhook(self, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.webhook_url:
            return {"ok": False, "error": "webhook_url not configured"}
        body = json.dumps({"event": event, "suite": SUITE_NAME, "payload": payload}).encode("utf-8")
        req = urllib.request.Request(
            self.config.webhook_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
                **({"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}),
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return {"ok": True, "status": resp.status, "body": raw, "latency_ms": (time.perf_counter() - t0) * 1000}
        except urllib.error.URLError as exc:
            return {"ok": False, "error": str(exc), "latency_ms": (time.perf_counter() - t0) * 1000}