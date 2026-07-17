from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    plane: str
    purpose: str
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()


DEFAULT_WORKERS = (
    WorkerSpec("ORIGO", "routing", "route intents and assemble execution plans", capabilities=("route", "plan")),
    WorkerSpec("SENSUS", "perception", "ingest repository, prompt, benchmark, and runtime signals", ("ORIGO",), ("ingest", "classify")),
    WorkerSpec("CORPUS", "data", "materialize, redact, deduplicate, and score owned corpus", ("SENSUS",), ("corpus", "provenance")),
    WorkerSpec("MATHESIS", "reasoning", "derive model math, bounds, and quantitative checks", ("SENSUS",), ("math", "verification")),
    WorkerSpec("CODEX", "coding", "generate, repair, and evaluate code", ("CORPUS",), ("code", "patch")),
    WorkerSpec("RTMX", "runtime", "compile governed runtime and execution plans", ("ORIGO",), ("execute", "sandbox")),
    WorkerSpec("PHAI", "research", "retrieve and synthesize first-party research", ("CORPUS",), ("research", "evidence")),
    WorkerSpec("PORT", "integration", "bridge MCP, HTTP, CLI, browser, and mobile surfaces", ("RTMX",), ("integrate", "serve")),
    WorkerSpec("TEST", "evaluation", "run unit, integration, benchmark, and regression suites", ("CODEX", "RTMX"), ("test", "benchmark")),
    WorkerSpec("BENC", "benchmark", "run model and systems benchmarks with receipts", ("TEST",), ("benchmark", "compare")),
    WorkerSpec("SACE", "safety", "redact secrets and enforce safe execution boundaries", ("SENSUS",), ("safety", "redact")),
    WorkerSpec("LAWX", "governance", "enforce release, provenance, and capability policy", ("SACE", "TEST"), ("govern", "approve")),
    WorkerSpec("SUCC", "release", "package checkpoints, reports, manifests, and serving lanes", ("LAWX", "PORT"), ("release", "receipt")),
    WorkerSpec("NOVA", "organism", "register model state and route approved intelligence", ("ORIGO", "LAWX"), ("state", "route")),
    WorkerSpec("CAIN", "adversarial", "challenge claims, unsafe actions, and release assumptions", ("TEST", "SACE"), ("redteam", "deny")),
    WorkerSpec("ORO", "resource", "map approved work to user, compute, and deployment lanes", ("NOVA", "SUCC"), ("allocate", "deploy")),
)


class WorkerRegistry:
    schema = "auro.workers.v1"

    def __init__(self, workers: tuple[WorkerSpec, ...] = DEFAULT_WORKERS) -> None:
        self.workers = workers
        self._index = {worker.worker_id: worker for worker in workers}
        self.validate()

    def validate(self) -> None:
        if len(self._index) != len(self.workers):
            raise ValueError("worker ids must be unique")
        for worker in self.workers:
            missing = [item for item in worker.dependencies if item not in self._index]
            if missing:
                raise ValueError(f"{worker.worker_id} has missing dependencies: {missing}")
        self.topological_order()

    def topological_order(self) -> list[str]:
        pending = {key: set(value.dependencies) for key, value in self._index.items()}
        order: list[str] = []
        while pending:
            ready = sorted(key for key, deps in pending.items() if not deps)
            if not ready:
                raise ValueError("worker dependency cycle detected")
            order.extend(ready)
            for key in ready:
                pending.pop(key)
            for deps in pending.values():
                deps.difference_update(ready)
        return order

    def plan(self, capability: str) -> list[str]:
        selected = {worker.worker_id for worker in self.workers if capability in worker.capabilities}
        expanded = set(selected)
        stack = list(selected)
        while stack:
            current = self._index[stack.pop()]
            for dependency in current.dependencies:
                if dependency not in expanded:
                    expanded.add(dependency)
                    stack.append(dependency)
        return [worker for worker in self.topological_order() if worker in expanded]

    def to_dict(self) -> dict:
        payload = {
            "schema": self.schema,
            "workers": [asdict(worker) for worker in self.workers],
            "order": self.topological_order(),
            "counts": {"workers": len(self.workers), "planes": len({worker.plane for worker in self.workers})},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target
