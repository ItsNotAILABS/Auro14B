"""User-facing AURO conversation and NEXUS tool-use release gate."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ConversationCase:
    case_id: str
    messages: tuple[Mapping[str, str], ...]
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    minimum_characters: int = 24
    require_nexus_evidence: bool = False


@dataclass(frozen=True)
class ConversationResult:
    case_id: str
    passed: bool
    score: float
    response_sha256: str
    response_excerpt: str
    checks: Mapping[str, bool]
    latency_ms: float


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate_case(
    case: ConversationCase,
    generator: Callable[[Sequence[Mapping[str, str]]], str],
    nexus_trace: Mapping[str, Any] | None = None,
) -> ConversationResult:
    started = time.perf_counter()
    response = str(generator(case.messages))
    latency_ms = (time.perf_counter() - started) * 1000.0
    normalized = _normalized(response)
    checks = {
        "minimum_length": len(response.strip()) >= case.minimum_characters,
        "required_terms": all(term.lower() in normalized for term in case.required_terms),
        "forbidden_terms": all(term.lower() not in normalized for term in case.forbidden_terms),
        "nexus_evidence": (not case.require_nexus_evidence) or bool(
            nexus_trace
            and nexus_trace.get("tool") in {"nexus", "nexus_relay"}
            and nexus_trace.get("receipt_sha256")
            and nexus_trace.get("content_sha256")
        ),
    }
    score = sum(bool(value) for value in checks.values()) / len(checks)
    return ConversationResult(
        case_id=case.case_id,
        passed=all(checks.values()),
        score=round(score, 4),
        response_sha256=_sha(response),
        response_excerpt=response[:500],
        checks=checks,
        latency_ms=round(latency_ms, 3),
    )


def run_conversation_gate(
    cases: Sequence[ConversationCase],
    generator: Callable[[Sequence[Mapping[str, str]]], str],
    nexus_traces: Mapping[str, Mapping[str, Any]] | None = None,
    minimum_pass_rate: float = 1.0,
) -> dict[str, Any]:
    traces = dict(nexus_traces or {})
    results = [evaluate_case(case, generator, traces.get(case.case_id)) for case in cases]
    passed = sum(result.passed for result in results)
    pass_rate = passed / max(1, len(results))
    payload = {
        "schema": "auro.user-conversation-gate.v1",
        "generated_at_unix": int(time.time()),
        "cases": len(results),
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "minimum_pass_rate": minimum_pass_rate,
        "gate_passed": bool(results) and pass_rate >= minimum_pass_rate,
        "results": [asdict(result) for result in results],
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return payload


def default_release_cases() -> tuple[ConversationCase, ...]:
    return (
        ConversationCase(
            "identity-continuity",
            (
                {"role": "system", "content": "You are AURO. Preserve the user's stated project identity."},
                {"role": "user", "content": "Relay is the product lane; AURO is the model family. Explain the boundary."},
            ),
            required_terms=("relay", "auro", "model"),
            forbidden_terms=("i don't know", "cannot remember"),
        ),
        ConversationCase(
            "nexus-grounded-answer",
            (
                {"role": "user", "content": "Use NEXUS evidence to summarize the retrieved source and identify its provenance."},
            ),
            required_terms=("source", "receipt"),
            require_nexus_evidence=True,
        ),
        ConversationCase(
            "failure-recovery",
            (
                {"role": "user", "content": "A tool call failed. State the limitation and give the safest next action without inventing results."},
            ),
            required_terms=("failed",),
            forbidden_terms=("successfully completed", "confirmed result"),
        ),
    )
