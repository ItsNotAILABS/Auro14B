#!/usr/bin/env python3
"""Run the configured Auro-2B council as a durable task worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auro_native_llm.production_fleet.council_service import CouncilService
from auro_native_llm.production_fleet.task_runtime import TaskRuntimeService
from auro_native_llm.production_fleet.task_worker import (
    CouncilStepExecutor,
    CouncilTaskWorker,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run analysis/research/write/review/synthesis steps through the AURO council"
    )
    parser.add_argument("run_id")
    parser.add_argument("--principal-id", required=True)
    parser.add_argument("--organization-id")
    parser.add_argument("--worker-id", default="auro-council-worker")
    parser.add_argument("--lease-seconds", type=int, default=900)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--idle-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    council = CouncilService.from_env()
    if not council.configured:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Auro-2B council is not configured",
                    "council": council.status(),
                },
                indent=2,
            )
        )
        return 3

    service = TaskRuntimeService.from_env(council)
    worker = CouncilTaskWorker(
        service.runtime,
        CouncilStepExecutor(council),
        run_id=args.run_id,
        principal_id=args.principal_id,
        organization_id=args.organization_id,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
    )
    result = worker.run_until_idle(
        max_steps=args.max_steps,
        idle_rounds=args.idle_rounds,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["final_status"] in {"succeeded", "queued", "awaiting_approval"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
