from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositoryLane:
    repository: str
    role: str
    priority: int
    visibility: str
    contributions: tuple[str, ...]


DEFAULT_LANES = (
    RepositoryLane("ItsNotAILABS/NOVA-private-root", "root-runtime", 100, "private", ("corpus", "workers", "protocols", "evaluation")),
    RepositoryLane("ItsNotAILABS/Auro14B", "model-runtime", 100, "public", ("corpus", "training", "benchmarks")),
    RepositoryLane("ItsNotAILABS/AURO", "model-research", 96, "public", ("corpus", "benchmarks", "protocols")),
    RepositoryLane("ItsNotAILABS/MedinaMemorySystems", "memory", 95, "public", ("corpus", "memory", "workers")),
    RepositoryLane("ItsNotAILABS/NATIVE-NOVA-PROTOCOL", "native-models", 95, "private", ("corpus", "models", "protocols")),
    RepositoryLane("ItsNotAILABS/PRODUCTION-", "production", 92, "public", ("corpus", "execution", "deployment")),
    RepositoryLane("ItsNotAILABS/Enterprise-OS-intelligence", "enterprise-os", 90, "public", ("corpus", "workers", "protocols")),
    RepositoryLane("ItsNotAILABS/cloudcolony", "colony-runtime", 90, "public", ("corpus", "datasets", "workers")),
    RepositoryLane("ItsNotAILABS/Chimeria", "defense-research", 88, "private", ("corpus", "evaluation", "protocols")),
    RepositoryLane("ItsNotAILABS/PARRALAX-AIHFTFUND", "market-intelligence", 88, "public", ("corpus", "evaluation", "math")),
    RepositoryLane("ItsNotAILABS/PARALLAX-Exchange-Clearinghouse", "clearing", 87, "public", ("corpus", "execution", "math")),
    RepositoryLane("ItsNotAILABS/NEUROSWARMAI", "swarm", 90, "public", ("corpus", "workers", "orchestration")),
    RepositoryLane("ItsNotAILABS/LOOM-Memoria-De-Intelligencia-", "loom-memory", 90, "public", ("corpus", "memory", "protocols")),
    RepositoryLane("ItsNotAILABS/CAPSULA", "capsule-runtime", 86, "public", ("corpus", "workers", "apps")),
    RepositoryLane("ItsNotAILABS/PhoneAI", "mobile-runtime", 82, "private", ("corpus", "apps", "evaluation")),
    RepositoryLane("ItsNotAILABS/ForgeBridge-MCP", "mcp-bridge", 86, "private", ("corpus", "benchmarks", "tools")),
    RepositoryLane("ItsNotAILABS/organism-bots-mcp-server", "worker-catalog", 86, "public", ("corpus", "workers", "tools")),
    RepositoryLane("ItsNotAILABS/nova-intelligence", "language-research", 84, "public", ("corpus", "models", "protocols")),
    RepositoryLane("ItsNotAILABS/MatDaemon", "math-runtime", 84, "public", ("corpus", "benchmarks", "math")),
    RepositoryLane("ItsNotAILABS/FABLEBREAKER", "adversarial-evaluation", 82, "public", ("corpus", "benchmarks", "safety")),
    RepositoryLane("ItsNotAILABS/CyberSecurity-AI", "defensive-security", 80, "public", ("corpus", "safety", "evaluation")),
)


class FederationManifest:
    schema = "auro.federation.v1"

    def __init__(self, lanes: tuple[RepositoryLane, ...] = DEFAULT_LANES) -> None:
        self.lanes = lanes

    def selected(self, contribution: str = "corpus") -> list[RepositoryLane]:
        return sorted(
            (lane for lane in self.lanes if contribution in lane.contributions),
            key=lambda lane: (-lane.priority, lane.repository.lower()),
        )

    def to_dict(self) -> dict:
        payload = {
            "schema": self.schema,
            "repositories": [asdict(lane) for lane in self.lanes],
            "counts": {
                "repositories": len(self.lanes),
                "private": sum(lane.visibility == "private" for lane in self.lanes),
                "public": sum(lane.visibility == "public" for lane in self.lanes),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FederationManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        lanes = tuple(RepositoryLane(**item) for item in raw["repositories"])
        return cls(lanes)
