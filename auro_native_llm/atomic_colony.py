"""Executable four-atomic specialization framework.

This layer clones a checkpoint identity into independently named specialist
instances, preserves lineage, routes work by declared capability, and emits a
hash-linked experiment receipt. It does not pretend specialization metadata is
weight training; a specialist becomes weight-specialized only after its own
training and evaluation artifacts are attached.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Iterable


@dataclass(frozen=True)
class AtomicSpecialist:
    specialist_id: str
    base_checkpoint: str
    role: str
    instruction: str
    capabilities: tuple[str, ...]
    checkpoint_sha256: str | None = None
    adapter_path: str | None = None

    def public(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicResult:
    specialist_id: str
    role: str
    output: str
    output_sha256: str


class AtomicColony:
    """A governed colony of independently specialized atomic models."""

    def __init__(self, specialists: Iterable[AtomicSpecialist]):
        self.specialists = tuple(specialists)
        if len(self.specialists) != 4:
            raise ValueError("the reference experiment requires exactly four atomic specialists")
        ids = [item.specialist_id for item in self.specialists]
        if len(set(ids)) != len(ids):
            raise ValueError("specialist_id values must be unique")

    @classmethod
    def repository_audit_colony(cls, base_checkpoint: str) -> "AtomicColony":
        return cls(
            (
                AtomicSpecialist("atomic-retriever", base_checkpoint, "retriever", "Locate the most relevant repository evidence.", ("search", "retrieve", "cite")),
                AtomicSpecialist("atomic-code-reader", base_checkpoint, "code_reader", "Inspect implementation details and reconstruct behavior.", ("code", "architecture", "dependency")),
                AtomicSpecialist("atomic-red-team", base_checkpoint, "red_team", "Find contradictions, regressions, and unsupported claims.", ("risk", "contradiction", "claim_boundary")),
                AtomicSpecialist("atomic-consolidator", base_checkpoint, "consolidator", "Combine evidence into one continuity-preserving answer.", ("synthesis", "decision", "receipt")),
            )
        )

    def route(self, task: str) -> AtomicSpecialist:
        lowered = task.lower()
        scored = []
        for specialist in self.specialists:
            score = sum(1 for capability in specialist.capabilities if capability.replace("_", " ") in lowered)
            scored.append((score, specialist.specialist_id, specialist))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    def run(self, task: str, executor: Callable[[AtomicSpecialist, str], str]) -> dict[str, object]:
        results = []
        for specialist in self.specialists:
            output = str(executor(specialist, task))
            results.append(
                AtomicResult(
                    specialist.specialist_id,
                    specialist.role,
                    output,
                    hashlib.sha256(output.encode("utf-8")).hexdigest(),
                )
            )
        routed = self.route(task)
        payload = {
            "schema": "auro.atomic_colony.experiment.v1",
            "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
            "base_checkpoints": sorted({item.base_checkpoint for item in self.specialists}),
            "routed_specialist": routed.specialist_id,
            "specialists": [item.public() for item in self.specialists],
            "results": [asdict(item) for item in results],
            "claim_boundary": "specialist roles are executable routing identities; weight specialization requires checkpoint or adapter evidence",
        }
        payload["receipt_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload
