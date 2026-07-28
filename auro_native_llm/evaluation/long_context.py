"""Long-context and MoE evaluation substrate for AURO.

The module is runtime-agnostic: exact model runners provide token losses,
retrieval answers, and router assignments. AURO turns those observations into
machine-verifiable curriculum, quality, balance, regression, and promotion
receipts. Synthetic smoke inputs prove mechanics only and never count as model
quality evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA = "auro.long_context_evidence.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    context_length: int
    token_budget: int
    long_sample_ratio: float
    learning_rate_scale: float = 1.0
    minimum_pass_rate: float = 0.0

    def validate(self, previous: Optional["CurriculumStage"] = None) -> None:
        if self.context_length <= 0 or self.token_budget <= 0:
            raise ValueError("context_length and token_budget must be positive")
        if not 0.0 <= self.long_sample_ratio <= 1.0:
            raise ValueError("long_sample_ratio must be within [0, 1]")
        if self.learning_rate_scale <= 0:
            raise ValueError("learning_rate_scale must be positive")
        if previous and self.context_length <= previous.context_length:
            raise ValueError("curriculum context lengths must strictly increase")


@dataclass(frozen=True)
class LongContextCurriculum:
    model_id: str
    target_context: int
    stages: Sequence[CurriculumStage]
    seed: int = 42
    schema: str = "auro.long_context_curriculum.v1"

    def validate(self) -> None:
        if not self.stages:
            raise ValueError("curriculum requires at least one stage")
        previous = None
        for stage in self.stages:
            stage.validate(previous)
            previous = stage
        if self.stages[-1].context_length != self.target_context:
            raise ValueError("final curriculum stage must equal target_context")

    def stage_for_tokens(self, consumed_tokens: int) -> CurriculumStage:
        self.validate()
        consumed = max(0, int(consumed_tokens))
        cursor = 0
        for stage in self.stages:
            cursor += stage.token_budget
            if consumed < cursor:
                return stage
        return self.stages[-1]

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {**asdict(self), "stages": [asdict(stage) for stage in self.stages], "curriculum_sha256": _hash(asdict(self))}


def geometric_curriculum(model_id: str, base_context: int, target_context: int, total_tokens: int, stages: int = 4) -> LongContextCurriculum:
    if target_context < base_context or target_context % base_context:
        raise ValueError("target_context must be an integer multiple of base_context")
    lengths = []
    value = base_context
    while value < target_context and len(lengths) < stages - 1:
        lengths.append(value)
        value = min(target_context, value * 2)
    lengths.append(target_context)
    lengths = sorted(set(lengths))
    budget = total_tokens // len(lengths)
    remainder = total_tokens - budget * len(lengths)
    rows = []
    for index, length in enumerate(lengths):
        rows.append(CurriculumStage(
            name=f"stage-{index + 1}-{length}",
            context_length=length,
            token_budget=budget + (remainder if index == len(lengths) - 1 else 0),
            long_sample_ratio=min(1.0, 0.25 + index * 0.25),
            learning_rate_scale=max(0.25, 1.0 / math.sqrt(index + 1)),
            minimum_pass_rate=min(0.95, 0.70 + index * 0.08),
        ))
    return LongContextCurriculum(model_id, target_context, tuple(rows))


def retrieval_report(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score exact retrieval cases produced by an external exact-checkpoint runner."""
    if not cases:
        raise ValueError("retrieval report requires cases")
    rows = []
    by_bucket: Dict[str, List[bool]] = {}
    for case in cases:
        expected = str(case.get("expected", "")).strip()
        observed = str(case.get("observed", "")).strip()
        passed = bool(expected) and observed == expected
        position = float(case.get("needle_position", 0.0))
        bucket = "early" if position < 0.34 else "middle" if position < 0.67 else "late"
        by_bucket.setdefault(bucket, []).append(passed)
        rows.append({**dict(case), "passed": passed, "position_bucket": bucket})
    accuracy = sum(row["passed"] for row in rows) / len(rows)
    buckets = {name: {"cases": len(values), "accuracy": sum(values) / len(values)} for name, values in sorted(by_bucket.items())}
    return {"schema": "auro.retrieval_position.v1", "cases": rows, "case_count": len(rows), "accuracy": accuracy, "position_buckets": buckets, "passed": accuracy >= 0.90 and all(item["accuracy"] >= 0.80 for item in buckets.values())}


def perplexity_by_position(token_losses: Sequence[float], bucket_count: int = 8) -> Dict[str, Any]:
    if not token_losses:
        raise ValueError("token_losses cannot be empty")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    losses = [float(value) for value in token_losses]
    rows = []
    for bucket in range(bucket_count):
        start = len(losses) * bucket // bucket_count
        end = len(losses) * (bucket + 1) // bucket_count
        segment = losses[start:end]
        if not segment:
            continue
        mean_loss = sum(segment) / len(segment)
        rows.append({"bucket": bucket, "start_fraction": start / len(losses), "end_fraction": end / len(losses), "tokens": len(segment), "mean_loss": mean_loss, "perplexity": math.exp(min(mean_loss, 50.0))})
    first = rows[0]["perplexity"]
    last = rows[-1]["perplexity"]
    degradation = last / first if first else float("inf")
    return {"schema": "auro.perplexity_by_position.v1", "token_count": len(losses), "buckets": rows, "first_to_last_ratio": degradation, "passed": degradation <= 1.20}


def routing_balance(assignments: Sequence[Sequence[int]], num_experts: int) -> Dict[str, Any]:
    """Measure expert load from per-token top-k expert IDs."""
    if num_experts < 2:
        raise ValueError("num_experts must be at least 2")
    counts = [0] * num_experts
    tokens = 0
    top_k = None
    for route in assignments:
        route = list(route)
        if top_k is None:
            top_k = len(route)
        if not route or len(route) != top_k:
            raise ValueError("all routes must use the same nonzero top-k")
        if len(set(route)) != len(route):
            raise ValueError("one token cannot select the same expert twice")
        for expert in route:
            if expert < 0 or expert >= num_experts:
                raise ValueError("expert id outside configured range")
            counts[expert] += 1
        tokens += 1
    if not tokens:
        raise ValueError("routing assignments cannot be empty")
    total = sum(counts)
    shares = [count / total for count in counts]
    uniform = 1.0 / num_experts
    variance = sum((share - uniform) ** 2 for share in shares) / num_experts
    cv = math.sqrt(variance) / uniform
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    normalized_entropy = entropy / math.log(num_experts)
    dead = [index for index, count in enumerate(counts) if count == 0]
    max_share = max(shares)
    return {"schema": "auro.moe_routing_balance.v1", "tokens": tokens, "top_k": top_k, "num_experts": num_experts, "counts": counts, "shares": shares, "coefficient_of_variation": cv, "normalized_entropy": normalized_entropy, "max_share": max_share, "dead_experts": dead, "passed": not dead and cv <= 0.35 and normalized_entropy >= 0.90 and max_share <= uniform * 1.75}


@dataclass(frozen=True)
class RegressionThresholds:
    retrieval_drop: float = 0.02
    perplexity_ratio_increase: float = 0.05
    routing_cv_increase: float = 0.05
    protected_metric_drop: float = 0.01


def regression_report(baseline: Mapping[str, Any], candidate: Mapping[str, Any], thresholds: RegressionThresholds = RegressionThresholds()) -> Dict[str, Any]:
    checks = []
    def check(name: str, base: float, cand: float, maximum_drop: float, higher_is_better: bool = True) -> None:
        delta = cand - base
        passed = delta >= -maximum_drop if higher_is_better else delta <= maximum_drop
        checks.append({"name": name, "baseline": base, "candidate": cand, "delta": delta, "passed": passed})

    check("retrieval_accuracy", float(baseline["retrieval"]["accuracy"]), float(candidate["retrieval"]["accuracy"]), thresholds.retrieval_drop, True)
    check("perplexity_position_ratio", float(baseline["perplexity"]["first_to_last_ratio"]), float(candidate["perplexity"]["first_to_last_ratio"]), thresholds.perplexity_ratio_increase, False)
    check("routing_cv", float(baseline["routing"]["coefficient_of_variation"]), float(candidate["routing"]["coefficient_of_variation"]), thresholds.routing_cv_increase, False)
    base_protected = baseline.get("protected_metrics", {})
    candidate_protected = candidate.get("protected_metrics", {})
    for name, base_value in sorted(base_protected.items()):
        if name not in candidate_protected:
            checks.append({"name": f"protected:{name}", "passed": False, "reason": "missing candidate metric"})
        else:
            check(f"protected:{name}", float(base_value), float(candidate_protected[name]), thresholds.protected_metric_drop, True)
    passed = all(item["passed"] for item in checks)
    return {"schema": "auro.long_context_regression.v1", "checks": checks, "passed": passed, "decision": "eligible" if passed else "quarantine"}


def build_evidence_receipt(*, model_id: str, checkpoint_sha256: str, curriculum: LongContextCurriculum, retrieval: Mapping[str, Any], perplexity: Mapping[str, Any], routing: Mapping[str, Any], regression: Mapping[str, Any], runner: Mapping[str, Any], exact_checkpoint: bool) -> Dict[str, Any]:
    components_pass = bool(retrieval.get("passed") and perplexity.get("passed") and routing.get("passed") and regression.get("passed"))
    promotable = bool(exact_checkpoint and checkpoint_sha256 and components_pass)
    payload = {
        "schema": SCHEMA,
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "exact_checkpoint": exact_checkpoint,
        "created_at_unix": int(time.time()),
        "curriculum": curriculum.to_dict(),
        "retrieval": dict(retrieval),
        "perplexity": dict(perplexity),
        "routing": dict(routing),
        "regression": dict(regression),
        "runner": dict(runner),
        "promotion": {"eligible": promotable, "decision": "promote" if promotable else "quarantine", "requires_signed_checkpoint_manifest": True},
        "claim_boundary": {"synthetic_smoke_is_quality_evidence": False, "long_context_quality_verified": promotable, "moe_routing_quality_verified": promotable},
    }
    payload["evidence_sha256"] = _hash(payload)
    return payload


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
