from pathlib import Path

from mesie.training_fabric import (
    ComputeNode,
    DatasetManifest,
    NodeRegistry,
    NodeRole,
    SovereignScheduler,
    TrainingJob,
    write_receipt,
)


def test_registry_and_scheduler(tmp_path: Path):
    registry = NodeRegistry(tmp_path / "nodes.json")
    registry.register(ComputeNode("n1", "trainer-1", [NodeRole.TRAINER], 4, 80, 512, 8000))
    registry.register(ComputeNode("n2", "trainer-2", [NodeRole.TRAINER], 4, 80, 512, 8000))
    dataset = DatasetManifest(
        "python-v1",
        "1",
        "livevault://python-v1",
        "abc",
        10_000,
        "owned",
        "medina-tokenizer-v1",
    )
    job = TrainingJob(
        "run-1",
        "medina-python-4b",
        dataset,
        8,
        80,
        ["torchrun", "train.py"],
    )
    plan = SovereignScheduler(registry).launch_plan(job)
    assert len(plan["nodes"]) == 2
    assert plan["state"] == "running"


def test_receipt(tmp_path: Path):
    receipt = write_receipt(tmp_path / "receipt.json", {"job_id": "run-1"})
    assert len(receipt["receipt_sha256"]) == 64
    assert (tmp_path / "receipt.json").exists()
