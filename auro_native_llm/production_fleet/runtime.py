"""Usable text-generation and internal-agent plane for the Auro/NOVA fleet.

The runtime deliberately separates model parameters from agent count: agents
share a checkpoint endpoint and never inflate the reported parameter total.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen


Generator = Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ModelEndpoint:
    id: str
    base_url: str
    model: str
    parameter_count: int | None = None
    role: str = "general"
    api_key_env: str | None = None

    @classmethod
    def from_env(cls) -> "ModelEndpoint":
        return cls(
            id=os.getenv("AURO_ENDPOINT_ID", "medina-native-8b"),
            base_url=os.getenv("AURO_BASE_URL", "http://127.0.0.1:8088/v1"),
            model=os.getenv("AURO_MODEL", "medina-native-8b"),
            parameter_count=_optional_int(os.getenv("AURO_PARAMETER_COUNT")),
            role="orchestrator",
            api_key_env=os.getenv("AURO_API_KEY_ENV") or None,
        )


@dataclass(frozen=True)
class AgentSpec:
    id: str
    role: str
    instruction: str
    capabilities: tuple[str, ...]


@dataclass
class AgentResult:
    agent_id: str
    role: str
    summary: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


DEFAULT_AGENTS = (
    AgentSpec("sensus", "analysis", "Extract intent, constraints, evidence, and ambiguity.", ("read", "analyze")),
    AgentSpec("mathesis", "logic", "Check logic, quantities, contradictions, and falsifiability.", ("calculate", "verify")),
    AgentSpec("architect", "architecture", "Design the smallest coherent system and interfaces.", ("plan", "design")),
    AgentSpec("red_team", "critic", "Find unsupported claims, unsafe actions, and likely failures.", ("review", "deny")),
    AgentSpec("operator", "execution", "Convert approved decisions into bounded executable actions.", ("capsula", "matdaemon")),
)


class OpenAICompatibleGenerator:
    def __init__(self, endpoint: ModelEndpoint, timeout: float = 120.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def __call__(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        body = {"model": self.endpoint.model, "messages": messages, **options}
        headers = {"content-type": "application/json"}
        if self.endpoint.api_key_env and os.getenv(self.endpoint.api_key_env):
            headers["authorization"] = "Bearer " + os.environ[self.endpoint.api_key_env]
        req = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode())
        text = payload["choices"][0]["message"]["content"]
        return {"text": text, "usage": payload.get("usage", {}), "raw_model": payload.get("model")}


class AgentManager:
    def __init__(self, generator: Generator, agents: Iterable[AgentSpec] = DEFAULT_AGENTS):
        self.generator = generator
        self.agents = {agent.id: agent for agent in agents}

    def run(self, objective: str, agent_ids: Iterable[str] | None = None) -> list[AgentResult]:
        selected = list(agent_ids or self.agents)
        results: list[AgentResult] = []
        shared = ""
        for agent_id in selected:
            agent = self.agents[agent_id]
            t0 = time.perf_counter()
            prompt = _agent_prompt(agent, objective, shared)
            response = self.generator(
                [{"role": "system", "content": prompt}, {"role": "user", "content": objective}],
                {"temperature": 0.2, "max_tokens": 700},
            )
            parsed = _parse_object(response["text"])
            result = AgentResult(
                agent_id=agent.id,
                role=agent.role,
                summary=str(parsed.get("summary", response["text"])).strip(),
                confidence=_clamp(parsed.get("confidence", 0.5)),
                evidence=_strings(parsed.get("evidence", [])),
                proposed_actions=_actions(parsed.get("proposed_actions", [])),
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
            results.append(result)
            shared += f"\n{agent.id}: {result.summary[:1200]}"
        return results


class NovaRuntime:
    """Interpret → deliberate → verify → answer, with action proposals gated."""

    def __init__(self, endpoint: ModelEndpoint | None = None, generator: Generator | None = None):
        self.endpoint = endpoint or ModelEndpoint.from_env()
        self.generator = generator or OpenAICompatibleGenerator(self.endpoint)
        self.agents = AgentManager(self.generator)

    def respond(self, message: str, *, execute: bool = False) -> dict[str, Any]:
        started = time.time()
        council = self.agents.run(message)
        council_json = json.dumps([asdict(x) for x in council], ensure_ascii=False)
        synthesis = self.generator(
            [
                {"role": "system", "content": _synthesis_prompt(execute)},
                {"role": "user", "content": f"OBJECTIVE:\n{message}\n\nCOUNCIL:\n{council_json}"},
            ],
            {"temperature": 0.25, "max_tokens": 1400},
        )
        answer = _parse_object(synthesis["text"])
        actions = _actions(answer.get("actions", []))
        approved = actions if execute else []
        return {
            "schema": "nova.production.response.v1",
            "answer": str(answer.get("answer", synthesis["text"])).strip(),
            "reasoning_summary": _strings(answer.get("reasoning_summary", [])),
            "confidence": _clamp(answer.get("confidence", 0.5)),
            "agents": [asdict(x) for x in council],
            "proposed_actions": actions,
            "approved_actions": approved,
            "model": {
                "endpoint_id": self.endpoint.id,
                "model": self.endpoint.model,
                "parameter_count": self.endpoint.parameter_count,
                "parameter_count_verified": self.endpoint.parameter_count is not None,
                "agent_count_is_not_parameter_count": True,
            },
            "elapsed_ms": round((time.time() - started) * 1000, 3),
        }


def _agent_prompt(agent: AgentSpec, objective: str, shared: str) -> str:
    return f"""You are NOVA internal agent {agent.id} ({agent.role}).
{agent.instruction}
Return JSON only: {{"summary": string, "confidence": 0..1,
"evidence": [string], "proposed_actions": [{{"tool": string, "arguments": object, "reason": string}}]}}.
Use concise conclusions; do not reveal private chain-of-thought. Never claim an action ran.
Prior council context: {shared or 'none'}"""


def _synthesis_prompt(execute: bool) -> str:
    posture = "Actions may be approved for a separate executor." if execute else "Do not approve or claim execution."
    return f"""You are NOVA, governing a council of internal model-backed agents.
Resolve disagreement, distinguish evidence from inference, and answer the user directly.
{posture}
Return JSON only: {{"answer": string, "reasoning_summary": [string],
"confidence": 0..1, "actions": [{{"tool": "matdaemon|capsula", "arguments": object, "reason": string}}]}}.
Reasoning summaries are conclusions and checks, not hidden chain-of-thought."""


def _parse_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"summary": text}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else {"summary": text}
            except json.JSONDecodeError:
                pass
        return {"summary": text, "answer": text}


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _strings(value: Any) -> list[str]:
    return [str(x) for x in value] if isinstance(value, list) else []


def _actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict) and x.get("tool") in {"matdaemon", "capsula"}]

