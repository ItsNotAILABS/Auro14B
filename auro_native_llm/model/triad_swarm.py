"""Auro-2B + three Auro-500M specialists + dynamic atomic swarms.

Every turn is processed by MESIE, split into bounded task capsules, executed by
three 500M specialist identities and topic-scoped 250M/156K agents, reconciled
by a triad consensus round, synthesized by the 2B parent, and rendered through
the pure-Python fluidizer. Model instances and parameter counts remain
separate; the runtime never relabels a swarm as one larger checkpoint.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
import uuid
from typing import Any, Callable, Mapping, Protocol, Sequence

from .atomic_family import ATOMIC_LADDER, AURO_500M_TRIAD, architecture_for
from .fluidizer import FluidizedResult, fluidize_report


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def estimate_tokens(text: str) -> int:
    """Stable transport estimate; exact checkpoint tokenizer counts are separate."""
    return 0 if not text else max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    parameter_target: int
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    adapter_id: str | None = None
    adapter_sha256: str | None = None
    measured_parameters: int | None = None
    provider: str = "repository-native"

    @property
    def checkpoint_verified(self) -> bool:
        return bool(self.checkpoint_id and self.checkpoint_sha256 and len(self.checkpoint_sha256) == 64)

    @property
    def distinct_specialization_verified(self) -> bool:
        return self.checkpoint_verified and bool(
            (self.adapter_id and self.adapter_sha256 and len(self.adapter_sha256) == 64)
            or self.checkpoint_id == self.model_id
        )

    def public(self) -> dict[str, Any]:
        row = asdict(self)
        row["checkpoint_verified"] = self.checkpoint_verified
        row["distinct_specialization_verified"] = self.distinct_specialization_verified
        row["agent_count_is_not_parameter_count"] = True
        return row


class ModelCallable(Protocol):
    def __call__(self, messages: list[dict[str, str]], options: dict[str, Any]) -> Mapping[str, Any] | str: ...


@dataclass
class ModelExecutor:
    identity: ModelIdentity
    generate: ModelCallable = field(repr=False)

    def invoke(self, system: str, user: str, *, max_tokens: int, temperature: float = 0.2) -> dict[str, Any]:
        started = time.perf_counter()
        raw = self.generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            {"max_tokens": int(max_tokens), "temperature": float(temperature)},
        )
        if isinstance(raw, Mapping):
            text = str(raw.get("text") or raw.get("answer") or raw.get("content") or "")
            usage = dict(raw.get("usage") or {})
        else:
            text, usage = str(raw), {}
        return {
            "text": text,
            "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "identity": self.identity.public(),
        }

    @classmethod
    def from_auro_language_model(cls, model: Any, identity: ModelIdentity | None = None) -> "ModelExecutor":
        ident = identity or ModelIdentity(
            model_id=str(model.model_id),
            parameter_target=int(model.config.parameter_target),
            measured_parameters=int(model.num_params),
            provider="auro-mesie-native",
        )

        def call(messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
            prompt = "\n".join(str(item.get("content", "")) for item in messages)
            result = model.generate(
                prompt,
                max_new_tokens=int(options.get("max_tokens", 256)),
                temperature=float(options.get("temperature", 0.2)),
                top_k=32,
                top_p=0.92,
            )
            return {"text": result.text, "usage": {"completion_tokens": len(result.token_ids)}, "raw": result.to_dict()}

        return cls(ident, call)


class MesieAdapter(Protocol):
    def analyze(self, text: str, model_id: str) -> dict[str, Any]: ...


class RepositoryMesieAdapter:
    """Runs the existing MESIE compute plane and returns a bounded receipt."""

    def __init__(self) -> None:
        from auro_native_llm.mesie_compute import get_compute_plane

        self.plane = get_compute_plane()

    def analyze(self, text: str, model_id: str) -> dict[str, Any]:
        from auro_native_llm.mesie_compute import profile_from_lane

        try:
            arch = architecture_for(model_id)
            architecture = {
                "runtime_d_model": min(arch.hidden_size, 768),
                "runtime_layers": min(arch.layers, 12),
                "runtime_heads": min(arch.attention_heads, 12),
                "kv_heads": min(arch.kv_heads, 4),
                "context_window_tokens_target": arch.context_window_tokens_target,
                "vocab_size_target": arch.vocab_size_target,
            }
            profile = profile_from_lane(model_id, arch.parameter_target, "atomic", architecture)
        except ValueError:
            from auro_native_llm.family import get_lane

            lane = get_lane(model_id) or get_lane("Auro-2B")
            if lane is None:
                raise RuntimeError(f"no MESIE profile for {model_id}")
            profile = profile_from_lane(lane.model_id, lane.parameter_target, lane.tier.value, lane.architecture.to_dict())
        result = self.plane.forward(text, profile)
        payload = result.to_dict()
        payload.pop("hidden_shape", None)
        payload["embedding_sha256"] = _sha(result.embedding)
        payload["model_id"] = model_id
        payload["receipt_sha256"] = _sha(payload)
        return payload


@dataclass(frozen=True)
class AtomicTask:
    task_id: str
    specialist_id: str
    model_id: str
    role: str
    objective: str
    constraints: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    capsule_sha256: str

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicReport:
    task: AtomicTask
    answer: str
    confidence: float
    evidence: tuple[str, ...]
    model_executed: bool
    model_identity: Mapping[str, Any] | None
    mesie_receipt: Mapping[str, Any]
    agent_receipt: Mapping[str, Any]
    latency_ms: float

    def public(self) -> dict[str, Any]:
        row = asdict(self)
        row["task"] = self.task.public()
        return row


@dataclass(frozen=True)
class SpecialistReport:
    specialist_id: str
    role: str
    analysis: str
    draft: str
    recommendations: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    contract_valid: bool
    atomic_reports: tuple[AtomicReport, ...]
    mesie_receipt: Mapping[str, Any]
    model_identity: Mapping[str, Any]
    latency_ms: float

    def public(self) -> dict[str, Any]:
        row = asdict(self)
        row["atomic_reports"] = [item.public() for item in self.atomic_reports]
        return row


@dataclass(frozen=True)
class ConsensusVote:
    specialist_id: str
    consensus: str
    confidence: float
    disagreements: tuple[str, ...]
    evidence: tuple[str, ...]
    contract_valid: bool
    latency_ms: float

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriadTurnResult:
    turn_id: str
    text: str
    structured_answer: Mapping[str, Any]
    specialist_reports: tuple[SpecialistReport, ...]
    consensus_votes: tuple[ConsensusVote, ...]
    atomic_agent_count: int
    model_backed_atomic_count: int
    mesie_receipts: tuple[Mapping[str, Any], ...]
    fluidizer: Mapping[str, Any]
    estimated_dispatch_tokens: int
    estimated_naive_broadcast_tokens: int
    estimated_text_reduction: float
    runtime_receipt_sha256: str
    promotion_ready: bool
    blockers: tuple[str, ...]
    schema: str = "auro.2b_triad_swarm.turn.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "specialist_reports": [item.public() for item in self.specialist_reports],
            "consensus_votes": [item.public() for item in self.consensus_votes],
        }


class TopicSwarmPlanner:
    """Selects topic-specific 250M/156K workers without broadcasting context."""

    KEYWORDS = {
        "code": ("code", "python", "javascript", "typescript", "debug", "test", "repo", "function", "api"),
        "research": ("research", "source", "evidence", "compare", "paper", "benchmark", "latest"),
        "execution": ("run", "build", "deploy", "tool", "worker", "sandbox", "browser", "file"),
        "creative": ("write", "creative", "story", "voice", "brand", "design", "conversation", "explain"),
        "memory": ("remember", "memory", "history", "context", "continuity", "previous"),
    }

    def classify(self, message: str) -> tuple[str, ...]:
        lower = message.lower()
        scores = {topic: sum(term in lower for term in terms) for topic, terms in self.KEYWORDS.items()}
        ranked = [topic for topic, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score]
        return tuple(ranked[:3] or ["general"])

    def plan(self, message: str, specialist_id: str, *, max_agents: int = 4) -> tuple[AtomicTask, ...]:
        topics = self.classify(message)
        roles: list[tuple[str, str]]
        if specialist_id.endswith("SENSUS"):
            roles = [("Auro-250M", "retrieval_filter"), ("Auro-250M", "intent_extract"), ("Auro-156K", "classifier")]
            if "memory" in topics:
                roles.append(("Auro-250M", "memory_consolidation"))
        elif specialist_id.endswith("PRAXIS"):
            roles = [("Auro-250M", "code_triage"), ("Auro-156K", "tool_selection"), ("Auro-156K", "json_repair")]
            if "execution" in topics or "code" in topics:
                roles.append(("Auro-250M", "structured_transform"))
        else:
            roles = [("Auro-250M", "semantic_outline"), ("Auro-156K", "style_guard"), ("Auro-250M", "structured_transform")]
            if "creative" in topics:
                roles.append(("Auro-156K", "classifier"))
        tasks: list[AtomicTask] = []
        for model_id, role in roles[: max(1, int(max_agents))]:
            material = {
                "specialist_id": specialist_id,
                "model_id": model_id,
                "role": role,
                "objective": f"Handle the {role} portion of this turn: {message[:1200]}",
                "constraints": ("Return conclusions, not hidden chain-of-thought", "Do not claim unexecuted actions"),
                "evidence_refs": tuple(f"topic:{topic}" for topic in topics),
            }
            tasks.append(
                AtomicTask(
                    task_id="atom_" + uuid.uuid4().hex[:12],
                    capsule_sha256=_sha(material),
                    **material,
                )
            )
        return tuple(tasks)


class Auro2BTriadSwarm:
    """Hierarchical generation runtime with explicit model and evidence custody."""

    def __init__(
        self,
        *,
        main_2b: ModelExecutor,
        specialists: Sequence[ModelExecutor],
        atomic_executors: Mapping[tuple[str, str], ModelExecutor] | None = None,
        mesie: MesieAdapter | None = None,
        planner: TopicSwarmPlanner | None = None,
        max_workers: int = 12,
    ) -> None:
        if len(specialists) != 3:
            raise ValueError("Auro-2B triad requires exactly three 500M specialists")
        expected = {item.variant_id for item in AURO_500M_TRIAD}
        found = {item.identity.model_id for item in specialists}
        if found != expected:
            raise ValueError(f"triad identities must be {sorted(expected)}; received {sorted(found)}")
        if main_2b.identity.parameter_target != 2_000_000_000:
            raise ValueError("main executor must declare the Auro-2B parameter target")
        self.main = main_2b
        self.specialists = tuple(specialists)
        self.atomic_executors = dict(atomic_executors or {})
        self.mesie = mesie or RepositoryMesieAdapter()
        self.planner = planner or TopicSwarmPlanner()
        self.max_workers = max(3, min(int(max_workers), 32))

    def run_turn(self, message: str, *, full_parent_context: str | None = None) -> TriadTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message is required")
        turn_id = "turn_" + uuid.uuid4().hex[:16]
        full_context = full_parent_context or message
        mesie_receipts: list[Mapping[str, Any]] = [self.mesie.analyze(message, "Auro-2B")]

        reports: list[SpecialistReport] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self._run_specialist, specialist, message): specialist for specialist in self.specialists}
            for future in as_completed(futures):
                reports.append(future.result())
        reports.sort(key=lambda item: item.specialist_id)
        for report in reports:
            mesie_receipts.append(report.mesie_receipt)
            mesie_receipts.extend(item.mesie_receipt for item in report.atomic_reports)

        votes: list[ConsensusVote] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self._consensus_vote, specialist, message, reports): specialist for specialist in self.specialists}
            for future in as_completed(futures):
                votes.append(future.result())
        votes.sort(key=lambda item: item.specialist_id)

        triad_consensus = self._reconcile_votes(votes)
        structured = self._main_synthesis(message, reports, votes, triad_consensus, mesie_receipts[0])
        fluid: FluidizedResult = fluidize_report(structured, voice="conversational")
        mesie_receipts.append(self.mesie.analyze(fluid.text, "Auro-2B"))

        atomic_count = sum(len(report.atomic_reports) for report in reports)
        model_backed = sum(item.model_executed for report in reports for item in report.atomic_reports)
        dispatch_payloads = [json.dumps(item.task.public(), sort_keys=True) for report in reports for item in report.atomic_reports]
        dispatch_tokens = sum(estimate_tokens(value) for value in dispatch_payloads)
        naive_calls = atomic_count + 6
        naive_tokens = estimate_tokens(full_context) * max(1, naive_calls)
        reduction = 0.0 if naive_tokens == 0 else max(0.0, 1.0 - dispatch_tokens / naive_tokens)

        blockers: list[str] = []
        if not self.main.identity.checkpoint_verified:
            blockers.append("Auro-2B exact checkpoint identity is not verified")
        for specialist in self.specialists:
            if not specialist.identity.distinct_specialization_verified:
                blockers.append(f"{specialist.identity.model_id} lacks distinct checkpoint or adapter evidence")
        if model_backed < atomic_count:
            blockers.append("one or more atomic agents used MESIE compute without an exact atomic checkpoint")
        if any(not report.contract_valid for report in reports):
            blockers.append("one or more specialist JSON contracts failed")
        if any(not vote.contract_valid for vote in votes):
            blockers.append("one or more triad consensus contracts failed")

        receipt_material = {
            "schema": "auro.2b_triad_swarm.turn.v1",
            "turn_id": turn_id,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "main": self.main.identity.public(),
            "specialists": [item.identity.public() for item in self.specialists],
            "reports": [item.public() for item in reports],
            "votes": [item.public() for item in votes],
            "structured_answer": structured,
            "fluidizer": fluid.to_dict(),
            "mesie_receipts": list(mesie_receipts),
            "estimated_dispatch_tokens": dispatch_tokens,
            "estimated_naive_broadcast_tokens": naive_tokens,
            "estimated_text_reduction": reduction,
        }
        return TriadTurnResult(
            turn_id=turn_id,
            text=fluid.text,
            structured_answer=structured,
            specialist_reports=tuple(reports),
            consensus_votes=tuple(votes),
            atomic_agent_count=atomic_count,
            model_backed_atomic_count=model_backed,
            mesie_receipts=tuple(mesie_receipts),
            fluidizer=fluid.to_dict(),
            estimated_dispatch_tokens=dispatch_tokens,
            estimated_naive_broadcast_tokens=naive_tokens,
            estimated_text_reduction=round(reduction, 6),
            runtime_receipt_sha256=_sha(receipt_material),
            promotion_ready=not blockers,
            blockers=tuple(dict.fromkeys(blockers)),
        )

    def _run_specialist(self, specialist: ModelExecutor, message: str) -> SpecialistReport:
        started = time.perf_counter()
        mesie_receipt = self.mesie.analyze(message, specialist.identity.model_id)
        tasks = self.planner.plan(message, specialist.identity.model_id)
        atomics: list[AtomicReport] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(tasks)))) as pool:
            futures = [pool.submit(self._run_atomic, task) for task in tasks]
            for future in as_completed(futures):
                atomics.append(future.result())
        atomics.sort(key=lambda item: item.task.task_id)
        variant = next(item for item in AURO_500M_TRIAD if item.variant_id == specialist.identity.model_id)
        response = specialist.invoke(
            f"You are {variant.variant_id}, AURO's {variant.role} specialist. Use bounded atomic reports and MESIE receipts. Return JSON only with analysis, draft, recommendations, evidence and confidence. Conclusions only; never expose hidden chain-of-thought or claim an action ran.",
            json.dumps({"objective": message, "capabilities": variant.capabilities, "mesie": mesie_receipt, "atomic_reports": [item.public() for item in atomics]}, ensure_ascii=False),
            max_tokens=900,
            temperature=0.2,
        )
        parsed, valid = _parse_json_object(response["text"])
        analysis = str(parsed.get("analysis") or parsed.get("summary") or response["text"]).strip()
        draft = str(parsed.get("draft") or parsed.get("answer") or analysis).strip()
        return SpecialistReport(
            specialist_id=specialist.identity.model_id,
            role=variant.role,
            analysis=analysis,
            draft=draft,
            recommendations=tuple(_string_list(parsed.get("recommendations"))),
            evidence=tuple(_string_list(parsed.get("evidence"))),
            confidence=_clamp(parsed.get("confidence", 0.5)),
            contract_valid=valid,
            atomic_reports=tuple(atomics),
            mesie_receipt=mesie_receipt,
            model_identity=specialist.identity.public(),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _run_atomic(self, task: AtomicTask) -> AtomicReport:
        started = time.perf_counter()
        mesie_receipt = self.mesie.analyze(task.objective, task.model_id)
        executor = self.atomic_executors.get((task.model_id, task.role)) or self.atomic_executors.get((task.model_id, "*"))
        answer = _mesie_summary(task, mesie_receipt)
        confidence = 0.45
        evidence = tuple(task.evidence_refs)
        model_identity: Mapping[str, Any] | None = None
        model_executed = False
        if executor is not None:
            response = executor.invoke(
                "You are a bounded AURO atomic expert. Return JSON only with answer, confidence and evidence. Do not claim unexecuted actions and do not reveal hidden chain-of-thought.",
                json.dumps({"task": task.public(), "mesie": mesie_receipt}, ensure_ascii=False),
                max_tokens=320,
                temperature=0.15,
            )
            parsed, _ = _parse_json_object(response["text"])
            answer = str(parsed.get("answer") or parsed.get("summary") or response["text"]).strip()
            confidence = _clamp(parsed.get("confidence", 0.5))
            evidence = tuple(_string_list(parsed.get("evidence"))) or evidence
            model_identity = executor.identity.public()
            model_executed = True

        agent_receipt = self._ghost_receipt(task, mesie_receipt, answer, model_executed)
        return AtomicReport(
            task=task,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            model_executed=model_executed,
            model_identity=model_identity,
            mesie_receipt=mesie_receipt,
            agent_receipt=agent_receipt,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @staticmethod
    def _ghost_receipt(task: AtomicTask, mesie_receipt: Mapping[str, Any], answer: str, model_executed: bool) -> dict[str, Any]:
        payload = {
            "schema": "auro.atomic_agent.execution.v1",
            "task_id": task.task_id,
            "model_id": task.model_id,
            "role": task.role,
            "capsule_sha256": task.capsule_sha256,
            "mesie_receipt_sha256": mesie_receipt.get("receipt_sha256"),
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "model_executed": model_executed,
            "execution_mode": "checkpoint+mesie" if model_executed else "mesie-compute-only",
        }
        payload["receipt_sha256"] = _sha(payload)
        return payload

    def _consensus_vote(self, specialist: ModelExecutor, message: str, reports: Sequence[SpecialistReport]) -> ConsensusVote:
        started = time.perf_counter()
        response = specialist.invoke(
            "You are one member of a three-model AURO consensus. Review all specialist reports. Return JSON only with consensus, disagreements, evidence, and confidence. Conclusions only; no hidden chain-of-thought.",
            json.dumps({"objective": message, "reports": [item.public() for item in reports]}, ensure_ascii=False),
            max_tokens=600,
            temperature=0.15,
        )
        parsed, valid = _parse_json_object(response["text"])
        return ConsensusVote(
            specialist_id=specialist.identity.model_id,
            consensus=str(parsed.get("consensus") or parsed.get("answer") or response["text"]).strip(),
            confidence=_clamp(parsed.get("confidence", 0.5)),
            disagreements=tuple(_string_list(parsed.get("disagreements"))),
            evidence=tuple(_string_list(parsed.get("evidence"))),
            contract_valid=valid,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @staticmethod
    def _reconcile_votes(votes: Sequence[ConsensusVote]) -> dict[str, Any]:
        ranked = sorted(votes, key=lambda item: (-item.confidence, item.specialist_id))
        leader = ranked[0] if ranked else None
        return {
            "consensus": leader.consensus if leader else "",
            "confidence": round(sum(item.confidence for item in votes) / max(1, len(votes)), 6),
            "leader": leader.specialist_id if leader else None,
            "disagreements": list(dict.fromkeys(value for item in votes for value in item.disagreements)),
            "evidence": list(dict.fromkeys(value for item in votes for value in item.evidence)),
            "vote_receipt_sha256": _sha([item.public() for item in votes]),
        }

    def _main_synthesis(self, message: str, reports: Sequence[SpecialistReport], votes: Sequence[ConsensusVote], triad_consensus: Mapping[str, Any], mesie_ingress: Mapping[str, Any]) -> dict[str, Any]:
        response = self.main.invoke(
            "You are Auro-2B, parent of a three-model 500M specialist triad and dynamic atomic swarms. Synthesize the evidence into JSON only: answer, reasoning_summary, key_points, caveats, next_steps, citations, confidence. Do not expose hidden chain-of-thought. Do not invent sources or claim actions ran.",
            json.dumps({"objective": message, "mesie_ingress": mesie_ingress, "specialist_reports": [item.public() for item in reports], "consensus_votes": [item.public() for item in votes], "triad_consensus": triad_consensus}, ensure_ascii=False),
            max_tokens=1_400,
            temperature=0.2,
        )
        parsed, valid = _parse_json_object(response["text"])
        if not valid:
            parsed = {"answer": response["text"].strip(), "reasoning_summary": ["The parent model did not return the requested JSON contract."], "caveats": ["Structured synthesis contract failed."], "confidence": 0.4}
        parsed["contract_valid"] = valid
        parsed["model_identity"] = self.main.identity.public()
        parsed["triad_consensus_receipt"] = triad_consensus.get("vote_receipt_sha256")
        return parsed


def _parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    raw = text.strip()
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, True
    return {"answer": raw}, False


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _mesie_summary(task: AtomicTask, receipt: Mapping[str, Any]) -> str:
    metrics = receipt.get("spectral_metrics") or {}
    return f"MESIE-only atomic observation for {task.role}: backend={receipt.get('backend')}; entropy={metrics.get('spectral_entropy')}; centroid={metrics.get('spectral_centroid')}."
