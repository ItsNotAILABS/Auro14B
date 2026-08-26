"""Production service wrapper for durable AURO missions."""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Mapping

from auro_native_llm.tasks import (
    ArtifactStore,
    CouncilTaskExecutor,
    MissionOrchestrator,
    MissionPlanner,
    MissionStore,
    PackageTaskExecutor,
)


_ID = re.compile(r"^[A-Za-z0-9_.:@/-]{1,160}$")


class MissionAuthorizationError(PermissionError):
    pass


class MissionService:
    """Tenant-aware mission API over the durable task orchestrator."""

    def __init__(
        self,
        *,
        store: MissionStore,
        artifacts: ArtifactStore,
        orchestrator: MissionOrchestrator,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.orchestrator = orchestrator

    @classmethod
    def from_env(cls, council_service: Any) -> "MissionService":
        db_path = Path(
            os.getenv("AURO_MISSION_DB", "state/mission-orchestrator.sqlite3")
        )
        artifact_root = Path(
            os.getenv("AURO_MISSION_ARTIFACT_ROOT", "state/mission-artifacts")
        )
        store = MissionStore(db_path)
        artifacts = ArtifactStore(
            artifact_root,
            max_artifact_bytes=int(
                os.getenv("AURO_MISSION_MAX_ARTIFACT_BYTES", str(64 * 1024 * 1024))
            ),
        )
        council_executor = CouncilTaskExecutor(council_service, artifacts)
        package_executor = PackageTaskExecutor(artifacts)
        executors = {
            "reasoning": council_executor,
            "research": council_executor,
            "implementation": council_executor,
            "artifact": council_executor,
            "review": council_executor,
            "synthesis": council_executor,
            "package": package_executor,
            "default": council_executor,
        }
        orchestrator = MissionOrchestrator(
            store,
            artifacts,
            executors,
            planner=MissionPlanner(),
        )
        return cls(store=store, artifacts=artifacts, orchestrator=orchestrator)

    @staticmethod
    def _identity(operator_id: str, organization_id: str) -> tuple[str, str]:
        operator = str(operator_id).strip()
        organization = str(organization_id).strip()
        if not _ID.fullmatch(operator):
            raise MissionAuthorizationError("valid operator identity is required")
        if not _ID.fullmatch(organization):
            raise MissionAuthorizationError("valid organization identity is required")
        return operator, organization

    def _authorized_mission(
        self,
        mission_id: str,
        *,
        operator_id: str,
        organization_id: str,
    ) -> dict[str, Any]:
        operator, organization = self._identity(operator_id, organization_id)
        mission = self.store.get_mission(str(mission_id))
        if mission["organization_id"] != organization:
            raise MissionAuthorizationError("mission belongs to another organization")
        if mission["operator_id"] != operator and operator != "system-admin":
            raise MissionAuthorizationError("mission belongs to another operator")
        return mission

    def create(
        self,
        request: Mapping[str, Any],
        *,
        operator_id: str,
        organization_id: str,
    ) -> dict[str, Any]:
        operator, organization = self._identity(operator_id, organization_id)
        tasks = request.get("tasks")
        if tasks is not None and not isinstance(tasks, list):
            raise ValueError("tasks must be an array")
        deliverables = request.get("deliverables") or []
        if not isinstance(deliverables, list):
            raise ValueError("deliverables must be an array")
        return self.orchestrator.create(
            objective=str(request.get("objective") or ""),
            title=str(request.get("title") or "AURO mission"),
            tasks=tasks,
            deliverables=deliverables,
            operator_id=operator,
            organization_id=organization,
            max_parallel=int(request.get("max_parallel", 3)),
            budget=dict(request.get("budget") or {}),
            deadline_unix=(
                int(request["deadline_unix"])
                if request.get("deadline_unix") is not None
                else None
            ),
            idempotency_key=str(request.get("idempotency_key") or "").strip() or None,
        )

    def list(
        self,
        *,
        operator_id: str,
        organization_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        operator, organization = self._identity(operator_id, organization_id)
        missions = self.store.list_missions(
            organization_id=organization,
            limit=limit,
        )
        if operator == "system-admin":
            return missions
        return [item for item in missions if item["operator_id"] == operator]

    def get(
        self,
        mission_id: str,
        *,
        operator_id: str,
        organization_id: str,
    ) -> dict[str, Any]:
        return self._authorized_mission(
            mission_id,
            operator_id=operator_id,
            organization_id=organization_id,
        )

    def run(
        self,
        mission_id: str,
        request: Mapping[str, Any],
        *,
        operator_id: str,
        organization_id: str,
    ) -> dict[str, Any]:
        self._authorized_mission(
            mission_id,
            operator_id=operator_id,
            organization_id=organization_id,
        )
        capabilities = request.get("capabilities") or []
        if not isinstance(capabilities, list):
            raise ValueError("capabilities must be an array")
        return self.orchestrator.run_burst(
            mission_id,
            worker_id=str(request.get("worker_id") or f"api:{operator_id}"),
            max_tasks=int(request.get("max_tasks", 20)),
            time_budget_seconds=int(request.get("time_budget_seconds", 300)),
            capabilities=[str(item) for item in capabilities],
        )

    def transition(
        self,
        mission_id: str,
        action: str,
        *,
        operator_id: str,
        organization_id: str,
    ) -> dict[str, Any]:
        self._authorized_mission(
            mission_id,
            operator_id=operator_id,
            organization_id=organization_id,
        )
        actions = {
            "pause": self.store.pause,
            "resume": self.store.resume,
            "cancel": self.store.cancel,
        }
        try:
            handler = actions[action]
        except KeyError as exc:
            raise ValueError(f"unsupported mission action: {action}") from exc
        return handler(mission_id)

    def status(self) -> dict[str, Any]:
        value = self.orchestrator.status()
        reasoning = self.orchestrator.executors.get("reasoning")
        council = getattr(reasoning, "council", None)
        value.update(
            {
                "multi_user": True,
                "tenant_isolation": "organization-and-operator-bound",
                "council_configured": bool(
                    council and getattr(council, "configured", False)
                ),
                "long_running_mode": "durable bursts with leases and resumable state",
            }
        )
        return value

    def artifact_path(
        self,
        mission_id: str,
        relative_path: str,
        *,
        operator_id: str,
        organization_id: str,
    ) -> Path:
        self._authorized_mission(
            mission_id,
            operator_id=operator_id,
            organization_id=organization_id,
        )
        root = self.artifacts.mission_root(mission_id)
        candidate = (root / str(relative_path)).resolve()
        candidate.relative_to(root)
        if not candidate.is_file():
            raise FileNotFoundError(relative_path)
        return candidate
