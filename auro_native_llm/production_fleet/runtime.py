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

from .model_orchestrator import ModelLane, MultiModelOrchestrator


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


class NativeOpenWeightGenerator:
    """Run a repository-native open-weight checkpoint with no provider call."""
    def __init__(self, checkpoint: str):
        from auro_native_llm.open_weights import OpenHIM
        self.model = OpenHIM.load(checkpoint)
        self.checkpoint = checkpoint

    def __call__(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        prompt = "\n".join(f"<{m.get('role','user')}> {m.get('content','')}" for m in messages) + "\n<assistant>"
        prompt_tokens = len(self.model.tokenizer.encode(prompt))
        text = self.model.generate(prompt, max_new_tokens=int(options.get("max_tokens", 256)),
                                   temperature=float(options.get("temperature", .4)), top_k=16)
        completion = text[len(prompt):] if text.startswith(prompt) else text
        completion_tokens = len(self.model.tokenizer.encode(completion))
        return {"text": completion, "usage": {"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,
                "total_tokens":prompt_tokens+completion_tokens}, "raw_model":"HIM-native-v0",
                "provider":"repository-native-open-weights"}


class AgentManager:
    def __init__(self, generator: Generator, agents: Iterable[AgentSpec] = DEFAULT_AGENTS, capability_context: str = ""):
        self.generator = generator
        self.agents = {agent.id: agent for agent in agents}
        self.capability_context = capability_context

    def run(self, objective: str, agent_ids: Iterable[str] | None = None) -> list[AgentResult]:
        selected = list(agent_ids or self.agents)
        results: list[AgentResult] = []
        shared = ""
        for agent_id in selected:
            agent = self.agents[agent_id]
            t0 = time.perf_counter()
            prompt = _agent_prompt(agent, objective, shared, self.capability_context)
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

    def __init__(self, endpoint: ModelEndpoint | None = None, generator: Generator | None = None, sdk=None):
        native_checkpoint = os.getenv("AURO_NATIVE_CHECKPOINT", "").strip()
        if endpoint is None and native_checkpoint:
            from auro_native_llm.open_weights import OpenHIM
            native = OpenHIM.load(native_checkpoint)
            endpoint = ModelEndpoint("him-native-v0", "local://open-weights", "HIM-native-v0",
                                     native.num_parameters, "orchestrator", None)
        self.endpoint = endpoint or ModelEndpoint.from_env()
        base_generator = generator or (NativeOpenWeightGenerator(native_checkpoint) if native_checkpoint else OpenAICompatibleGenerator(self.endpoint))
        self.model_orchestrator = _build_model_orchestrator(self.endpoint, base_generator, native_checkpoint)
        self.generator = self.model_orchestrator
        if sdk is None:
            from .organ_sdk import AuroOrganSDK
            sdk = AuroOrganSDK()
        self.sdk = sdk
        from .capabilities import NativeCapabilities
        self.capabilities = NativeCapabilities(sdk)
        capability_context = json.dumps({"organs":sdk.manifest(),"native_capabilities":self.capabilities.manifest(),
                                         "brain":self.capabilities.brain.snapshot()},ensure_ascii=False)
        self.agents = AgentManager(self.generator, capability_context=capability_context)

    def respond(self, message: str, *, execute: bool = False) -> dict[str, Any]:
        started = time.time()
        self.model_orchestrator.drain_traces()
        brain_cycle = self.capabilities.brain.cycle(message, importance=.7 if execute else .5, execute_requested=execute)
        council = self.agents.run(message)
        council_json = json.dumps([asdict(x) for x in council], ensure_ascii=False)
        synthesis = self.generator(
            [
                {"role": "system", "content": _synthesis_prompt(execute, self.sdk.action_contract())},
                {"role": "user", "content": f"OBJECTIVE:\n{message}\n\nCOUNCIL:\n{council_json}"},
            ],
            {"temperature": 0.25, "max_tokens": 1400},
        )
        answer = _parse_object(synthesis["text"])
        actions = _actions(answer.get("actions", []))
        approved = actions if execute else []
        executions = []
        if execute:
            for action in approved:
                try:
                    executions.append(self.sdk.execute(action))
                except Exception as exc:
                    executions.append({"tool":action.get("tool"),"ok":False,"error":str(exc)[:500]})
        routing_traces = self.model_orchestrator.drain_traces()
        response = {
            "schema": "nova.production.response.v1",
            "answer": str(answer.get("answer", synthesis["text"])).strip(),
            "reasoning_summary": _strings(answer.get("reasoning_summary", [])),
            "confidence": _clamp(answer.get("confidence", 0.5)),
            "agents": [asdict(x) for x in council],
            "proposed_actions": actions,
            "approved_actions": approved,
            "executions": executions,
            "organ_sdk": self.sdk.manifest(),
            "native_capabilities": self.capabilities.manifest(),
            "brain": {"cycle":asdict(brain_cycle),"snapshot":self.capabilities.brain.snapshot()},
            "model_fleet": self.model_orchestrator.manifest(),
            "routing_traces": routing_traces,
            "models_used": sorted({a["lane_id"] for trace in routing_traces for a in trace["attempts"] if a["ok"]}),
            "model": {
                "endpoint_id": self.endpoint.id,
                "model": self.endpoint.model,
                "parameter_count": self.endpoint.parameter_count,
                "parameter_count_verified": self.endpoint.parameter_count is not None,
                "agent_count_is_not_parameter_count": True,
            },
            "elapsed_ms": round((time.time() - started) * 1000, 3),
        }
        response["receipt"] = asdict(self.capabilities.ledger.record("model_response",self.endpoint.model,True,response,{"agent_count":len(council)}))
        return response


def _agent_prompt(agent: AgentSpec, objective: str, shared: str, capability_context: str = "") -> str:
    return f"""You are NOVA internal agent {agent.id} ({agent.role}).
{agent.instruction}
Return JSON only: {{"summary": string, "confidence": 0..1,
"evidence": [string], "proposed_actions": [{{"tool": string, "arguments": object, "reason": string}}]}}.
Use concise conclusions; do not reveal private chain-of-thought. Never claim an action ran.
Only propose actions matching this exact capability contract: {capability_context or 'no tools available'}
Prior council context: {shared or 'none'}"""


def _synthesis_prompt(execute: bool, action_contract: dict[str,Any] | None = None) -> str:
    posture = "Actions may be approved for a separate executor." if execute else "Do not approve or claim execution."
    return f"""You are NOVA, governing a council of internal model-backed agents.
Resolve disagreement, distinguish evidence from inference, and answer the user directly.
{posture}
Return JSON only: {{"answer": string, "reasoning_summary": [string],
"confidence": 0..1, "actions": [{{"tool": "matdaemon|capsula", "arguments": object, "reason": string}}]}}.
Reasoning summaries are conclusions and checks, not hidden chain-of-thought.
Every action must match this exact contract: {json.dumps(action_contract or {},ensure_ascii=False)}"""


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


def _build_model_orchestrator(endpoint: ModelEndpoint, base_generator: Generator, native_checkpoint: str = "") -> MultiModelOrchestrator:
    """Build the real fleet from the primary lane plus AURO_MODEL_FLEET_JSON.

    Registry entries are explicit endpoints; no model is invented or downloaded.
    """
    native=bool(native_checkpoint)
    lanes=[ModelLane(endpoint.id,endpoint.model,endpoint.role,
        "repository-native-open-weights" if native else "openai-compatible-explicit",
        base_generator,endpoint.parameter_count,("general","code","math","research","tool"),0,native,True,
        os.getenv("AURO_NATIVE_CHECKPOINT_SHA256") or None)]
    raw=os.getenv("AURO_MODEL_FLEET_JSON","").strip()
    if raw:
        registry=json.loads(raw)
        if not isinstance(registry,list): raise ValueError("AURO_MODEL_FLEET_JSON must be a JSON array")
        for item in registry:
            if not isinstance(item,dict): raise ValueError("model fleet entries must be objects")
            model_endpoint=ModelEndpoint(
                id=str(item["id"]),base_url=str(item["base_url"]),model=str(item["model"]),
                parameter_count=_optional_int(str(item.get("parameter_count") or "")),
                role=str(item.get("role","general")),api_key_env=str(item.get("api_key_env") or "") or None)
            capabilities=tuple(str(x) for x in item.get("capabilities",["general"]))
            lanes.append(ModelLane(model_endpoint.id,model_endpoint.model,model_endpoint.role,
                str(item.get("provider","openai-compatible-explicit")),OpenAICompatibleGenerator(model_endpoint),
                model_endpoint.parameter_count,capabilities,int(item.get("priority",100)),
                bool(item.get("local",False)),bool(item.get("enabled",True)),item.get("checkpoint_hash")))
    ids=[x.id for x in lanes]
    if len(ids)!=len(set(ids)): raise ValueError("model lane ids must be unique")
    return MultiModelOrchestrator(lanes,allow_hosted_fallback=os.getenv("AURO_ALLOW_HOSTED_FALLBACK","0")=="1")
