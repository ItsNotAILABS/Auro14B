"""Resident worker for durable AURO harness execution."""
from __future__ import annotations

import argparse
import os
import socket
import time

from .harness import IndependentHarnessFabric
from .harness_orchestrator import HarnessOrchestrator


class HarnessWorker:
    def __init__(
        self,
        fabric: IndependentHarnessFabric | None = None,
        *,
        worker_id: str | None = None,
        poll_seconds: float = 5.0,
        cycles_per_harness: int = 1,
    ) -> None:
        self.fabric = fabric or IndependentHarnessFabric()
        self.orchestrator = HarnessOrchestrator(self.fabric)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.poll_seconds = max(0.25, float(poll_seconds))
        self.cycles_per_harness = max(1, int(cycles_per_harness))
        self.running = True

    def tick(self) -> dict:
        advanced = []
        skipped = []
        for state in self.fabric.store.list():
            if state.state != "active":
                continue
            try:
                if state.child_ids:
                    result = self.orchestrator.advance_tree(
                        state.id,
                        worker_id=self.worker_id,
                        cycles_per_child=self.cycles_per_harness,
                    )
                else:
                    result = self.fabric.run_until_blocked(
                        state.id,
                        worker_id=self.worker_id,
                        max_cycles=self.cycles_per_harness,
                    )
                advanced.append({"harness_id": state.id, "result": result})
            except RuntimeError as exc:
                skipped.append({"harness_id": state.id, "reason": str(exc)})
        return {
            "schema": "auro.harness.worker-tick.v3",
            "worker_id": self.worker_id,
            "advanced": advanced,
            "skipped": skipped,
        }

    def serve(self) -> None:
        while self.running:
            self.tick()
            time.sleep(self.poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="auro-harness-worker")
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("AURO_HARNESS_POLL_SECONDS", "5")))
    parser.add_argument("--cycles", type=int, default=int(os.getenv("AURO_HARNESS_CYCLES_PER_TICK", "1")))
    parser.add_argument("--worker-id", default=os.getenv("AURO_HARNESS_WORKER_ID"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    worker = HarnessWorker(worker_id=args.worker_id, poll_seconds=args.poll_seconds, cycles_per_harness=args.cycles)
    if args.once:
        print(worker.tick())
    else:
        worker.serve()


if __name__ == "__main__":
    main()
