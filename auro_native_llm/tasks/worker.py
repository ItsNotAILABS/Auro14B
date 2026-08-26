"""Long-running worker loop for durable AURO missions."""
from __future__ import annotations

import argparse
import json
import signal
import time
from typing import Any

from auro_native_llm.production_fleet.council_service import CouncilService
from auro_native_llm.production_fleet.task_service import MissionService


class MissionWorker:
    def __init__(
        self,
        service: MissionService,
        *,
        worker_id: str,
        max_tasks_per_burst: int = 8,
        time_budget_seconds: int = 240,
        idle_seconds: float = 2.0,
    ) -> None:
        self.service = service
        self.worker_id = worker_id
        self.max_tasks_per_burst = max(1, min(int(max_tasks_per_burst), 100))
        self.time_budget_seconds = max(1, min(int(time_budget_seconds), 3600))
        self.idle_seconds = max(0.1, min(float(idle_seconds), 60.0))
        self.stopping = False

    def stop(self, *_args: Any) -> None:
        self.stopping = True

    def run_once(self) -> dict[str, Any]:
        missions = self.service.store.list_missions(limit=500)
        candidates = [
            item
            for item in missions
            if item["status"] in {"queued", "running"}
        ]
        results = []
        for mission in candidates:
            if self.stopping:
                break
            result = self.service.orchestrator.run_burst(
                mission["mission_id"],
                worker_id=self.worker_id,
                max_tasks=self.max_tasks_per_burst,
                time_budget_seconds=self.time_budget_seconds,
                capabilities=("council", "artifact-write", "package"),
            )
            results.append(
                {
                    "mission_id": mission["mission_id"],
                    "status": result["mission"]["status"],
                    "tasks_executed": result["tasks_executed"],
                    "failures_or_retries": result["task_failures_or_retries"],
                }
            )
        return {
            "schema": "auro.mission-worker.pass.v1",
            "worker_id": self.worker_id,
            "candidate_missions": len(candidates),
            "results": results,
        }

    def run_forever(self) -> None:
        while not self.stopping:
            report = self.run_once()
            print(json.dumps(report, sort_keys=True), flush=True)
            if not report["results"] or all(
                item["tasks_executed"] == 0 for item in report["results"]
            ):
                time.sleep(self.idle_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AURO durable mission worker")
    parser.add_argument("--worker-id", default="auro-mission-worker")
    parser.add_argument("--max-tasks-per-burst", type=int, default=8)
    parser.add_argument("--time-budget-seconds", type=int, default=240)
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    service = MissionService.from_env(CouncilService.from_env())
    worker = MissionWorker(
        service,
        worker_id=args.worker_id,
        max_tasks_per_burst=args.max_tasks_per_burst,
        time_budget_seconds=args.time_budget_seconds,
        idle_seconds=args.idle_seconds,
    )
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    if args.once:
        print(json.dumps(worker.run_once(), indent=2, sort_keys=True))
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
