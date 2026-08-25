"""Executable Auro-2B council with three 500M specialists and atomic swarms.

The parent, specialist, and atomic model identities remain separate. The
runtime never adds their parameter counts together and calls the result one
checkpoint. It moves bounded task capsules, invokes MESIE at every stage,
preserves disagreement, performs two synthesis rounds, and applies a
Python/Pyodide-compatible conversational renderer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Callable, Mapping, Protocol, Sequence

from .atomic_family import AURO_500M_TRIAD, architecture_for, estimate_tokens
from .fluidizer import FluidizedResult, fluidize_report


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _parse_object(text: str) -> tuple[dict[str, Any], bool]:
    raw = str(text or "").strip()
    candidates = [raw]
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.find("{") : raw.rfind("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value, True
    return {"answer": raw}, False


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
        return bool(
            self.checkpoint_id
            and self.checkpoint_sha256
            and len(self.checkpoint_sha256) == 64
        )

    @property
    def adapter_verified(self) -> bool:
        return bool(self.adapter_id and self.adapter_sha256 and len(self.adapter_sha256) == 64)

    @property
    def specialization_token(self) -> str | None:
        if self.adapter_verified:
            return "adapter:" + str(self.adapter_sha256)
        if self.checkpoint_verified:
            return "checkpoint:" + str(self.checkpoint_sha256)
        return None

    def public(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(
            {
                "checkpoint_verified": self.checkpoint_verified,
                "adapter_verified": self.adapter_verified,
                "agent_count_is_not_parameter_count": True,
            }
        )
        return row


class ModelCallable(Protocol):
    def __call__(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> Mapping[str, Any] | str: ...


@dataclass
class ModelExecutor:
    identity: ModelIdentity
    generate: ModelCallable = field(repr=False)

    def invoke(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        raw = self.generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            {
                "max_tokens": max(16, int(max_tokens)),
                "temperature": float(temperature),
            },
        )
        if isinstance(raw, Mapping):
            text = str(raw.get("text") or raw.get("answer") or raw.get("content") or "")
            usage = dict(raw.get("usage") or {})
        else:
            text = str(raw)
            usage = {}
        return {
            "text": text,
            "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            "identity": self.identity.public(),
        }

    @classmethod
    def from_auro_language_model(
        cls,
        model: Any,
        identity: ModelIdentity | None = None,
    ) -> "ModelExecutor":
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
            return {
                "text": result.text,
                "usage": {"completion_tokens": len(result.token_ids)},
                "raw": result.to_dict(),
            }

        return cls(ident, call)


class MesieAdapter(Protocol):
    def analyze(self, text: str, model_id: str) -> dict[str, Any]: ...


class RepositoryMesieAdapter:
    """Run the repository MESIE plane and emit a bounded stage receipt."""

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
            profile = profile_from_lane(
                model_id,
                arch.parameter_target,
                "atomic",
                architecture,
            )
        except ValueError:
            from auro_native_llm.family import get_lane

            lane = get_lane(model_id) or get_lane("Auro-2B")
            if lane is None:
                raise RuntimeError(f"no MESIE profile for {model_id}")
            profile = profile_from_lane(
                lane.model_id,
                lane.parameter_target,
                lane.tier.value,
                lane.architecture.to_dict(),
            )
        result = self.plane.forward(text, profile)
        payload = result.to_dict()
        payload.pop("hidden_shape", None)
        payload["embedding_sha256"] = _sha(result.embedding)
        payload["model_id"] = model_id
        payload["input_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
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
    max_output_tokens: int
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
    contract_valid: bool
    model_identity: Mapping[str, Any] | None
    mesie_receipt: Mapping[str, Any]
    execution_receipt_sha256: str
    latency_ms: float

    def public(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "task": self.task.public(),
        }


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
        return {
            **asdict(self),
            "atomic_reports": [item.public() for item in self.atomic_reports],
        }


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
class CouncilTurnResult:
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
    runtime_receipt: Mapping[str, Any]
    evidence_class: str
    release_evidence_ready: bool
    blockers: tuple[str, ...]
    schema: str = "auro.2b-council.turn.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "specialist_reports": [item.public() for item in self.specialist_reports],
            "consensus_votes": [item.public() for item in self.consensus_votes],
        }


class TopicSwarmPlanner:
    """Select topic-specific 250M and 156K workers without full-context broadcast."""

    KEYWORDS = {
        "code": ("code", "python", "javascript", "typescript", "debug", "test", "repo", "api"),
        "research": ("research", "source", "evidence", "compare", "paper", "benchmark", "latest"),
        "execution": ("run", "build", "deploy", "tool", "worker", "sandbox", "browser", "file"),
        "creative": ("write", "creative", "story", "voice", "brand", "design", "conversation"),
        "memory": ("remember", "memory", "history", "context", "continuity", "previous"),
    }

    def classify(self, message: str) -> tuple[str, ...]:
        lower = message.lower()
        scores = {
            topic: sum(term in lower for term in terms)
            for topic, terms in self.KEYWORDS.items()
        }
        ranked = [
            topic
            for topic, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if score
        ]
        return tuple(ranked[:3] or ["general"])

    def plan(
        self,
        message: str,
        specialist_id: str,
        *,
        max_agents: int = 4,
    ) -> tuple[AtomicTask, ...]:
        topics = self.classify(message)
        if specialist_id.endswith("SENSUS"):
            roles = [
                ("Auro-250M", "retrieval_filter"),
                ("Auro-250M", "intent_extract"),
                ("Auro-156K", "classifier"),
            ]
            if "memory" in topics:
                roles.append(("Auro-250M", "memory_consolidation"))
        elif specialist_id.endswith("PRAXIS"):
            roles = [
                ("Auro-250M", "code_triage"),
                ("Auro-156K", "tool_selection"),
                ("Auro-156K", "json_repair"),
            ]
            if "execution" in topics or "code" in topics:
                roles.append(("Auro-250M", "structured_transform"))
        else:
            roles = [
                ("Auro-250M", "semantic_outline"),
                ("Auro-156K", "style_guard"),
                ("Auro-250M", "structured_transform"),
            ]
            if "creative" in topics:
                roles.append(("Auro-156K", "classifier"))

        tasks: list[AtomicTask] = []
        for model_id, role in roles[: max(1, int(max_agents))]:
            material = {
                "specialist_id": specialist_id,
                "model_id": model_id,
                "role": role,
                "objective": f"Handle the {role} portion of this turn: {message[:1200]}",
                "constraints": (
                    "Return conclusions, not hidden chain-of-thought",
                    "Do not claim unexecuted actions",
                    "Preserve uncertainty and evidence references",
                ),
                "evidence_refs": tuple(f"topic:{topic}" for topic in topics),
                "max_output_tokens": 256,
            }
            tasks.append(
                AtomicTask(
                    task_id="atom_" + uuid.uuid4().hex[:12],
                    capsule_sha256=_sha(material),
                    **material,
                )
            )
        return tuple(tasks)


class Auro2BCouncilRuntime:
    """Hierarchical generation runtime with explicit model and evidence custody."""

    def __init__(
        self,
        *,
        main_2b: ModelExecutor,
        specialists: Sequence[ModelExecutor],
        atomic_executors: Mapping[str, ModelExecutor] | None = None,
        mesie: MesieAdapter | None = None,
        planner: TopicSwarmPlanner | None = None,
        max_workers: int = 12,
        signing_key: str | bytes | None = None,
        signer_id: str = "auro-council-local",
    ) -> None:
        expected = {item.variant_id for item in AURO_500M_TRIAD}
        found = {item.identity.model_id for item in specialists}
        if len(specialists) != 3 or found != expected:
            raise ValueError(
                f"Auro-2B council requires exactly {sorted(expected)}; received {sorted(found)}"
            )
        if main_2b.identity.model_id != "Auro-2B":
            raise ValueError("main executor must identify as Auro-2B")
        if main_2b.identity.parameter_target != 2_000_000_000:
            raise ValueError("main executor must declare the Auro-2B parameter target")
        self.main = main_2b
        self.specialists = tuple(specialists)
        self.atomic_executors = dict(atomic_executors or {})
        self.mesie = mesie or RepositoryMesieAdapter()
        self.planner = planner or TopicSwarmPlanner()
        self.max_workers = max(3, min(int(max_workers), 32))
        raw_key = signing_key or b""
        self.signing_key = raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key
        self.signer_id = signer_id

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "auro.2b-council.manifest.v1",
            "main": self.main.identity.public(),
            "specialists": [item.identity.public() for item in self.specialists],
            "atomic_executors": {
                key: value.identity.public() for key, value in sorted(self.atomic_executors.items())
            },
            "mesie_every_stage": True,
            "full_parent_context_broadcast_to_atomic": False,
            "conversational_renderer": "python-wasm-fluidizer",
            "receipt_signing_configured": bool(self.signing_key),
            "parameter_accounting": "each checkpoint reported independently",
        }

    def run_turn(
        self,
        message: str,
        *,
        full_parent_context: str | None = None,
    ) -> CouncilTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message is required")
        turn_id = "turn_" + uuid.uuid4().hex[:16]
        parent_context = str(full_parent_context or message)
        mesie_receipts: list[Mapping[str, Any]] = [self.mesie.analyze(message, "Auro-2B")]

        reports: list[SpecialistReport] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._run_specialist, specialist, message): specialist
                for specialist in self.specialists
            }
            for future in as_completed(futures):
                reports.append(future.result())
        reports.sort(key=lambda item: item.specialist_id)
        for report in reports:
            mesie_receipts.append(report.mesie_receipt)
            mesie_receipts.extend(item.mesie_receipt for item in report.atomic_reports)

        votes: list[ConsensusVote] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._consensus_vote, specialist, message, reports): specialist
                for specialist in self.specialists
            }
            for future in as_completed(futures):
                votes.append(future.result())
        votes.sort(key=lambda item: item.specialist_id)

        triad_consensus = self._reconcile_votes(votes)
        structured, main_contract_valid = self._main_synthesis(
            message,
            parent_context,
            reports,
            votes,
            triad_consensus,
            mesie_receipts[0],
        )
        fluid: FluidizedResult = fluidize_report(structured, voice="conversational")
        mesie_receipts.append(self.mesie.analyze(fluid.text, "Auro-2B"))

        atomic_reports = [item for report in reports for item in report.atomic_reports]
        atomic_count = len(atomic_reports)
        model_backed = sum(1 for item in atomic_reports if item.model_executed)
        dispatch_tokens = sum(
            estimate_tokens(json.dumps(item.task.public(), sort_keys=True))
            for item in atomic_reports
        )
        naive_calls = atomic_count + 6
        naive_tokens = estimate_tokens(parent_context) * max(1, naive_calls)
        reduction = 0.0 if naive_tokens == 0 else max(0.0, 1.0 - dispatch_tokens / naive_tokens)

        blockers = self._blockers(
            reports=reports,
            votes=votes,
            atomic_reports=atomic_reports,
            main_contract_valid=main_contract_valid,
        )
        receipt_material = {
            "schema": "auro.2b-council.receipt.v1",
            "turn_id": turn_id,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "main_identity": self.main.identity.public(),
            "specialist_identities": [item.identity.public() for item in self.specialists],
            "specialist_report_hashes": [_sha(item.public()) for item in reports],
            "consensus_vote_hashes": [_sha(item.public()) for item in votes],
            "structured_answer_sha256": _sha(structured),
            "fluidized_output_sha256": fluid.output_sha256,
            "mesie_receipts": [str(item.get("receipt_sha256", "")) for item in mesie_receipts],
            "atomic_agent_count": atomic_count,
            "model_backed_atomic_count": model_backed,
            "estimated_dispatch_tokens": dispatch_tokens,
            "estimated_naive_broadcast_tokens": naive_tokens,
            "estimated_text_reduction": round(reduction, 6),
            "blockers": blockers,
        }
        receipt_hash = _sha(receipt_material)
        signature = None
        if self.signing_key:
            signature = hmac.new(self.signing_key, receipt_hash.encode("ascii"), hashlib.sha256).hexdigest()
        runtime_receipt = {
            **receipt_material,
            "receipt_sha256": receipt_hash,
            "signature": signature,
            "signer_id": self.signer_id if signature else None,
            "custody": "local-signed" if signature else "local-unsigned",
        }

        evidence_class = (
            "E4-signed-receipt"
            if signature and not blockers
            else "E3-validated-output"
            if main_contract_valid
            and all(item.contract_valid for item in reports)
            and all(item.contract_valid for item in votes)
            else "E2-execution-log"
        )
        release_ready = bool(signature and not blockers)
        return CouncilTurnResult(
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
            runtime_receipt=runtime_receipt,
            evidence_class=evidence_class,
            release_evidence_ready=release_ready,
            blockers=tuple(blockers),
        )

    def _run_specialist(
        self,
        specialist: ModelExecutor,
        message: str,
    ) -> SpecialistReport:
        started = time.perf_counter()
        tasks = self.planner.plan(message, specialist.identity.model_id)
        atomic_reports: list[AtomicReport] = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_workers)) as pool:
            futures = {pool.submit(self._run_atomic, task): task for task in tasks}
            for future in as_completed(futures):
                atomic_reports.append(future.result())
        atomic_reports.sort(key=lambda item: item.task.task_id)
        mesie_receipt = self.mesie.analyze(message, specialist.identity.model_id)
        atomic_payload = [
            {
                "model_id": item.task.model_id,
                "role": item.task.role,
                "answer": item.answer,
                "confidence": item.confidence,
                "evidence": list(item.evidence),
                "model_executed": item.model_executed,
                "mesie": item.mesie_receipt.get("spectral_metrics", {}),
            }
            for item in atomic_reports
        ]
        system = (
            f"You are {specialist.identity.model_id}. Produce a bounded specialist report. "
            "Use the atomic reports as evidence, preserve uncertainty, expose no hidden chain-of-thought, "
            "and never claim that a tool or training run executed. Return JSON only with keys: "
            "analysis, draft, recommendations, evidence, confidence."
        )
        user = json.dumps(
            {
                "objective": message,
                "atomic_reports": atomic_payload,
                "mesie": mesie_receipt.get("spectral_metrics", {}),
            },
            ensure_ascii=False,
        )
        invoked = specialist.invoke(system, user, max_tokens=700, temperature=0.2)
        parsed, parsed_json = _parse_object(invoked["text"])
        analysis = str(parsed.get("analysis") or "").strip()
        draft = str(parsed.get("draft") or parsed.get("answer") or invoked["text"]).strip()
        contract_valid = bool(parsed_json and analysis and draft)
        return SpecialistReport(
            specialist_id=specialist.identity.model_id,
            role=next(
                item.role for item in AURO_500M_TRIAD if item.variant_id == specialist.identity.model_id
            ),
            analysis=analysis or "Specialist contract did not provide a separate analysis summary.",
            draft=draft,
            recommendations=_strings(parsed.get("recommendations", [])),
            evidence=_strings(parsed.get("evidence", [])),
            confidence=_clamp(parsed.get("confidence", 0.5)),
            contract_valid=contract_valid,
            atomic_reports=tuple(atomic_reports),
            mesie_receipt=mesie_receipt,
            model_identity=specialist.identity.public(),
            latency_ms=round((time.perf_counter() - started) * 1_000, 3),
        )

    def _run_atomic(self, task: AtomicTask) -> AtomicReport:
        started = time.perf_counter()
        mesie_receipt = self.mesie.analyze(task.objective, task.model_id)
        executor = self.atomic_executors.get(task.model_id)
        if executor is None:
            receipt = {
                "schema": "auro.atomic.execution.v1",
                "task_id": task.task_id,
                "status": "mesie-only",
                "model_executed": False,
                "mesie_receipt_sha256": mesie_receipt.get("receipt_sha256"),
            }
            return AtomicReport(
                task=task,
                answer="",
                confidence=0.0,
                evidence=task.evidence_refs,
                model_executed=False,
                contract_valid=True,
                model_identity=None,
                mesie_receipt=mesie_receipt,
                execution_receipt_sha256=_sha(receipt),
                latency_ms=round((time.perf_counter() - started) * 1_000, 3),
            )

        system = (
            f"You are {task.model_id} acting only as {task.role}. Return JSON only with keys "
            "answer, confidence, evidence. Return conclusions, not hidden chain-of-thought. "
            "Do not claim unexecuted actions."
        )
        invoked = executor.invoke(
            system,
            json.dumps(task.public(), ensure_ascii=False),
            max_tokens=task.max_output_tokens,
            temperature=0.1,
        )
        parsed, parsed_json = _parse_object(invoked["text"])
        answer = str(parsed.get("answer") or invoked["text"]).strip()
        contract_valid = bool(parsed_json and answer)
        receipt = {
            "schema": "auro.atomic.execution.v1",
            "task_id": task.task_id,
            "status": "completed" if contract_valid else "contract-invalid",
            "model_executed": True,
            "model_identity": executor.identity.public(),
            "output_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "mesie_receipt_sha256": mesie_receipt.get("receipt_sha256"),
        }
        return AtomicReport(
            task=task,
            answer=answer,
            confidence=_clamp(parsed.get("confidence", 0.5)),
            evidence=_strings(parsed.get("evidence", [])),
            model_executed=True,
            contract_valid=contract_valid,
            model_identity=executor.identity.public(),
            mesie_receipt=mesie_receipt,
            execution_receipt_sha256=_sha(receipt),
            latency_ms=round((time.perf_counter() - started) * 1_000, 3),
        )

    def _consensus_vote(
        self,
        specialist: ModelExecutor,
        message: str,
        reports: Sequence[SpecialistReport],
    ) -> ConsensusVote:
        started = time.perf_counter()
        compact = [
            {
                "specialist_id": item.specialist_id,
                "analysis": item.analysis,
                "draft": item.draft,
                "recommendations": list(item.recommendations),
                "evidence": list(item.evidence),
                "confidence": item.confidence,
            }
            for item in reports
        ]
        system = (
            "Review all three specialist reports. Preserve material disagreement and uncertainty. "
            "Return JSON only with keys consensus, confidence, disagreements, evidence. "
            "Do not claim execution or hidden reasoning."
        )
        invoked = specialist.invoke(
            system,
            json.dumps({"objective": message, "reports": compact}, ensure_ascii=False),
            max_tokens=500,
            temperature=0.1,
        )
        parsed, parsed_json = _parse_object(invoked["text"])
        consensus = str(parsed.get("consensus") or parsed.get("answer") or invoked["text"]).strip()
        return ConsensusVote(
            specialist_id=specialist.identity.model_id,
            consensus=consensus,
            confidence=_clamp(parsed.get("confidence", 0.5)),
            disagreements=_strings(parsed.get("disagreements", [])),
            evidence=_strings(parsed.get("evidence", [])),
            contract_valid=bool(parsed_json and consensus),
            latency_ms=round((time.perf_counter() - started) * 1_000, 3),
        )

    @staticmethod
    def _reconcile_votes(votes: Sequence[ConsensusVote]) -> str:
        if not votes:
            return ""
        ranked = sorted(votes, key=lambda item: (-item.confidence, item.specialist_id))
        return ranked[0].consensus

    def _main_synthesis(
        self,
        message: str,
        parent_context: str,
        reports: Sequence[SpecialistReport],
        votes: Sequence[ConsensusVote],
        triad_consensus: str,
        ingress_mesie: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        compact_reports = [
            {
                "specialist_id": item.specialist_id,
                "analysis": item.analysis,
                "draft": item.draft,
                "recommendations": list(item.recommendations),
                "evidence": list(item.evidence),
                "confidence": item.confidence,
            }
            for item in reports
        ]
        compact_votes = [item.public() for item in votes]
        system = (
            "You are Auro-2B, the final parent synthesizer. Resolve the council without hiding "
            "meaningful disagreement. Return JSON only with keys answer, key_points, recommendations, "
            "caveats, confidence, citations. Do not claim actions, training, deployment, memory, or "
            "checkpoint quality without evidence in the supplied reports."
        )
        user = json.dumps(
            {
                "objective": message,
                "bounded_parent_context": parent_context[-12_000:],
                "triad_consensus": triad_consensus,
                "specialist_reports": compact_reports,
                "consensus_votes": compact_votes,
                "mesie_ingress": ingress_mesie.get("spectral_metrics", {}),
            },
            ensure_ascii=False,
        )
        invoked = self.main.invoke(system, user, max_tokens=1_200, temperature=0.2)
        parsed, parsed_json = _parse_object(invoked["text"])
        answer = str(parsed.get("answer") or invoked["text"]).strip()
        if not answer:
            fallback = max(reports, key=lambda item: item.confidence).draft if reports else ""
            answer = fallback or "The council did not produce a usable answer."
        structured = {
            "answer": answer,
            "key_points": list(_strings(parsed.get("key_points", []))),
            "recommendations": list(_strings(parsed.get("recommendations", []))),
            "caveats": list(_strings(parsed.get("caveats", []))),
            "confidence": _clamp(parsed.get("confidence", 0.0), 0.0),
            "citations": list(_strings(parsed.get("citations", []))),
            "triad_consensus": triad_consensus,
            "main_model": self.main.identity.public(),
        }
        return structured, bool(parsed_json and isinstance(parsed.get("answer"), str) and answer)

    def _blockers(
        self,
        *,
        reports: Sequence[SpecialistReport],
        votes: Sequence[ConsensusVote],
        atomic_reports: Sequence[AtomicReport],
        main_contract_valid: bool,
    ) -> list[str]:
        blockers: list[str] = []
        if not self.main.identity.checkpoint_verified:
            blockers.append("Auro-2B exact checkpoint custody is not verified")
        specialization_tokens = [item.identity.specialization_token for item in self.specialists]
        if any(token is None for token in specialization_tokens):
            blockers.append("one or more 500M specialists lack checkpoint or adapter evidence")
        elif len(set(specialization_tokens)) != 3:
            blockers.append("500M specialist identities do not have three distinct checkpoint or adapter proofs")
        if not all(item.contract_valid for item in reports):
            blockers.append("one or more specialist report contracts failed")
        if not all(item.contract_valid for item in votes):
            blockers.append("one or more consensus vote contracts failed")
        if not main_contract_valid:
            blockers.append("Auro-2B final synthesis contract failed")
        if not atomic_reports or any(not item.model_executed for item in atomic_reports):
            blockers.append("one or more atomic tasks ran MESIE-only without an atomic checkpoint executor")
        if any(item.model_executed and not item.contract_valid for item in atomic_reports):
            blockers.append("one or more atomic model response contracts failed")
        atomic_identities = {
            str(item.model_identity.get("model_id")): item.model_identity
            for item in atomic_reports
            if item.model_identity
        }
        for model_id in ("Auro-156K", "Auro-250M"):
            identity = atomic_identities.get(model_id)
            if not identity or not identity.get("checkpoint_verified"):
                blockers.append(f"{model_id} checkpoint custody is not verified")
        if not self.signing_key:
            blockers.append("council runtime receipt signing is not configured")
        return list(dict.fromkeys(blockers))
