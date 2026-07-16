from __future__ import annotations

import json
from pathlib import Path

from mesie.training_fabric.discovery import discover_compute_node, emit_node_receipt


def test_cpu_node_discovery_is_honest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mesie.training_fabric.discovery._nvidia_query", lambda: [])
    node, facts = discover_compute_node(tmp_path)
    assert node.gpu_count == 0
    assert facts["schema"] == "mesie-compute-node/1.0"
    assert "orchestrator" in [role.value for role in node.roles]


def test_compute_node_receipt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mesie.training_fabric.discovery._nvidia_query",
        lambda: [{"index": 0, "name": "Test GPU", "memory_mib": 24576, "uuid": "GPU-test", "compute_capability": "8.9"}],
    )
    target = tmp_path / "node.json"
    receipt = emit_node_receipt(target, tmp_path)
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert receipt["node"]["gpu_count"] == 1
    assert receipt["node"]["gpu_memory_gb"] == 24
    assert stored["receipt_sha256"]
