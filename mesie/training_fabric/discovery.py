from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ComputeNode, NodeRole
from .receipts import write_receipt


def _nvidia_query() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [
        executable,
        "--query-gpu=index,name,memory.total,uuid,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, shell=False)
    if completed.returncode != 0:
        return []
    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        devices.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mib": int(float(parts[2])),
                "uuid": parts[3],
                "compute_capability": parts[4],
            }
        )
    return devices


def discover_compute_node(root: str | Path = ".") -> tuple[ComputeNode, dict[str, Any]]:
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    devices = _nvidia_query()
    memory_gb = min((device["memory_mib"] // 1024 for device in devices), default=0)
    storage = shutil.disk_usage(root_path)
    node_id = os.getenv("MESIE_NODE_ID", f"mesie-{socket.gethostname()}")
    roles = [NodeRole.TRAINER, NodeRole.EVALUATOR, NodeRole.INFERENCE]
    if not devices:
        roles = [NodeRole.ORCHESTRATOR, NodeRole.DATA]
    node = ComputeNode(
        node_id=node_id,
        hostname=socket.gethostname(),
        roles=roles,
        gpu_count=len(devices),
        gpu_memory_gb=memory_gb,
        system_memory_gb=_system_memory_gb(),
        storage_free_gb=int(storage.free // (1024**3)),
        labels={
            "mesie.compute": "spectral-transformer",
            "mesie.runner": "github-actions",
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
        },
    )
    facts = {
        "schema": "mesie-compute-node/1.0",
        "node": node.to_dict(),
        "gpus": devices,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "root": str(root_path),
    }
    return node, facts


def emit_node_receipt(path: str | Path, root: str | Path = ".") -> dict[str, Any]:
    _, facts = discover_compute_node(root)
    return write_receipt(path, facts)


def _system_memory_gb() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int((pages * page_size) // (1024**3))
    except (AttributeError, ValueError, OSError):
        return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discover and receipt a MESIE compute node")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="receipts/compute-node.json")
    args = parser.parse_args()
    receipt = emit_node_receipt(args.out, args.root)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
