"""Model-explicit NOVA agent family runtime with capability receipts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Callable, Iterable

from .registry import NOVA_AGENT_FAMILY, NovaAgentSpec

Generator = Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]]


@dataclass
class NovaTaskResult:
    agent_id: str
    objective: str
    model_lane: dict[str, Any]
    answer: str
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    executions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    latency_ms: float = 0.0
    receipt_hash: str = ""

    def seal(self) -> "NovaTaskResult":
        material = asdict(self)
        material["receipt_hash"] = ""
        self.receipt_hash = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        return self

    def public(self) -> dict[str, Any]:
        return asdict(self)


class NovaAgentFamily:
    """Run named NOVA agents over one explicit model lane and capability plane."""

    def __init__(
        self,
        generator: Generator,
        model_lane: dict[str, Any],
        capabilities: Any,
        agents: Iterable[NovaAgentSpec] = NOVA_AGENT_FAMILY,
    ) -> None:
        self.generator = generator
        self.model_lane = dict(model_lane)
        self.capabilities = capabilities
        self.agents = {agent.id: agent for agent in agents}
        self.history: list[dict[str, Any]] = []

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "nova.agent-family.v1",
            "model_lane": self.model_lane,
            "agent_count": len(self.agents),
            "parameter_accounting": "model lane only; agents add zero parameters",
            "agents": [agent.public() for agent in self.agents.values()],
        }

    def run(
        self,
        agent_id: str,
        objective: str,
        *,
        execute: bool = False,
        approved_capabilities: Iterable[str] = (),
    ) -> NovaTaskResult:
        if agent_id not in self.agents:
            raise ValueError(f"unknown NOVA agent: {agent_id}")
        agent = self.agents[agent_id]
        approved = set(approved_capabilities)
        started = time.perf_counter()
        response = self.generator(
            [
                {"role": "system", "content": self._prompt(agent, execute)},
                {"role": "user", "content": objective},
            ],
            {"temperature": 0.2, "max_tokens": 1200},
        )
        parsed = _parse(response.get("text", ""))
        actions = _actions(parsed.get("actions", []))
        executions: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        if execute:
            for action in actions:
                name = action["name"]
                if name not in agent.capabilities:
                    executions.append({"ok": False, "denied": True, "capability": name, "reason": "outside agent contract"})
                    continue
                call_approved = name in approved
                result = self.capabilities.call(name, action.get("arguments", {}), approved=call_approved)
                executions.append(result)
                output = result.get("output") if isinstance(result, dict) else None
                if isinstance(output, dict) and (output.get("files") or output.get("manifest")):
                    artifacts.append(output)
        result = NovaTaskResult(
            agent_id=agent.id,
            objective=objective,
            model_lane=self.model_lane,
            answer=str(parsed.get("answer") or parsed.get("summary") or response.get("text", "")).strip(),
            proposed_actions=actions,
            executions=executions,
            artifacts=artifacts,
            ok=bool(str(parsed.get("answer") or parsed.get("summary") or response.get("text", "")).strip()),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        ).seal()
        self.history.append(result.public())
        return result

    def run_team(self, objective: str, agent_ids: Iterable[str]) -> dict[str, Any]:
        results = [self.run(agent_id, objective) for agent_id in agent_ids]
        body = {
            "schema": "nova.agent-team.v1",
            "objective": objective,
            "model_lane": self.model_lane,
            "results": [x.public() for x in results],
            "ok": all(x.ok for x in results),
        }
        body["receipt_hash"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        return body

    def _prompt(self, agent: NovaAgentSpec, execute: bool) -> str:
        return (
            f"You are {agent.name}, a NOVA agent. Role: {agent.role}. Purpose: {agent.purpose}\n"
            f"You are not a model. The explicit model lane is {json.dumps(self.model_lane, sort_keys=True)}.\n"
            f"Allowed capabilities: {json.dumps(list(agent.capabilities))}.\n"
            f"Execution enabled: {execute}. Never claim execution unless a returned receipt proves it.\n"
            "Return JSON only: {\"answer\": string, \"actions\": [{\"name\": string, \"arguments\": object, \"reason\": string}]}."
        )


def _parse(text: str) -> dict[str, Any]:
    clean = str(text or "").strip()
    try:
        value = json.loads(clean)
        return value if isinstance(value, dict) else {"answer": clean, "actions": []}
    except Exception:
        start, end = clean.find("{"), clean.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(clean[start : end + 1])
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    return {"answer": clean, "actions": []}


def _actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        out.append({
            "name": item["name"],
            "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
            "reason": str(item.get("reason", "")),
        })
    return out
