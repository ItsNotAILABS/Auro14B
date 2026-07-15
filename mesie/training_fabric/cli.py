from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import ComputeNode, NodeRole
from .registry import NodeRegistry


def main() -> None:
    parser = argparse.ArgumentParser(prog="mesie-fabric")
    parser.add_argument("--registry", default=".mesie/nodes.json")
    subcommands = parser.add_subparsers(dest="cmd", required=True)

    register = subcommands.add_parser("register-node")
    register.add_argument("--node-id", required=True)
    register.add_argument("--hostname", required=True)
    register.add_argument("--gpu-count", type=int, required=True)
    register.add_argument("--gpu-memory-gb", type=int, required=True)
    register.add_argument("--system-memory-gb", type=int, required=True)
    register.add_argument("--storage-free-gb", type=int, required=True)
    register.add_argument("--role", action="append", default=None)

    subcommands.add_parser("list-nodes")
    args = parser.parse_args()
    registry = NodeRegistry(Path(args.registry))

    if args.cmd == "register-node":
        roles = args.role or [NodeRole.TRAINER.value]
        node = ComputeNode(
            node_id=args.node_id,
            hostname=args.hostname,
            roles=[NodeRole(role) for role in roles],
            gpu_count=args.gpu_count,
            gpu_memory_gb=args.gpu_memory_gb,
            system_memory_gb=args.system_memory_gb,
            storage_free_gb=args.storage_free_gb,
        )
        registry.register(node)
        print(json.dumps(node.to_dict(), indent=2))
    else:
        print(json.dumps({key: value.to_dict() for key, value in registry.nodes.items()}, indent=2))


if __name__ == "__main__":
    main()
