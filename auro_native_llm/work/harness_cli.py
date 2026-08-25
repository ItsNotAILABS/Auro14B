"""CLI for AURO independent harness operations."""
from __future__ import annotations

import argparse
import json

from .harness import IndependentHarnessFabric
from .harness_orchestrator import HarnessOrchestrator


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="auro-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create")
    p.add_argument("objective")
    p.add_argument("--model", default="Auro-2B")

    p = sub.add_parser("orchestrate")
    p.add_argument("objective")
    p.add_argument("--model", default="Auro-2B")
    p.add_argument("--children", type=int, default=6)

    p = sub.add_parser("run")
    p.add_argument("harness_id")
    p.add_argument("--cycles", type=int, default=16)
    p.add_argument("--worker", default="cli")

    p = sub.add_parser("advance-tree")
    p.add_argument("harness_id")
    p.add_argument("--cycles-per-child", type=int, default=8)
    p.add_argument("--worker", default="cli")

    for command in ("pause", "resume", "cancel", "get", "aggregate"):
        p = sub.add_parser(command)
        p.add_argument("harness_id")

    sub.add_parser("list")
    sub.add_parser("manifest")

    args = parser.parse_args(argv)
    fabric = IndependentHarnessFabric()
    orchestrator = HarnessOrchestrator(fabric)

    if args.command == "create":
        _print(fabric.create_harness(args.objective, model_id=args.model).to_dict())
    elif args.command == "orchestrate":
        _print(orchestrator.orchestrate(args.objective, model_id=args.model, max_children=args.children))
    elif args.command == "run":
        _print(fabric.run_until_blocked(args.harness_id, worker_id=args.worker, max_cycles=args.cycles))
    elif args.command == "advance-tree":
        _print(orchestrator.advance_tree(args.harness_id, worker_id=args.worker, cycles_per_child=args.cycles_per_child))
    elif args.command == "pause":
        _print(fabric.pause(args.harness_id).to_dict())
    elif args.command == "resume":
        _print(fabric.resume(args.harness_id).to_dict())
    elif args.command == "cancel":
        _print(fabric.cancel(args.harness_id).to_dict())
    elif args.command == "get":
        _print(fabric.store.load(args.harness_id).to_dict())
    elif args.command == "aggregate":
        _print(fabric.aggregate(args.harness_id))
    elif args.command == "list":
        _print({"harnesses": [x.to_dict() for x in fabric.store.list()]})
    elif args.command == "manifest":
        _print(fabric.manifest())


if __name__ == "__main__":
    main()
