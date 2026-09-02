import json
from pathlib import Path

from auro_native_llm.brain_state.runtime_state import BrainRuntimeState, PersistentBrainState
from next_checkpoint.auro_eval_registry import registry, verify_result
from next_checkpoint.distributed_st14b_train import DistributedTrainContract, write_launch_receipt


def test_brain_state_hash_chain_and_restore(tmp_path):
    store = PersistentBrainState(tmp_path / "brain.json")
    first = store.transition(observation="sensor observation", task="research")
    second = store.transition(observation="browser result", consequence={"kind": "success"})
    restored = store.load()
    assert restored.sequence == 2
    assert restored.previous_hash == first.state_hash
    assert restored.state_hash == second.state_hash
    assert restored.episodic_memory[-1]["parent_hash"] == first.state_hash
    assert "<brain_state>" in store.inference_context(restored)


def test_brain_state_rejects_invalid_range():
    state = BrainRuntimeState(trust=1.1)
    try:
        state.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid trust must be rejected")


def test_eval_registry_contains_portfolio_suites():
    payload = registry()
    names = {item["name"] for item in payload["suites"]}
    assert {"ChromeBench", "IoTBench", "MemoryBench", "AgentContinuityBench", "SecurityBoundaryBench"} <= names
    assert len(payload["sha256"]) == 64


def test_eval_hard_failure_blocks_promotion():
    result = verify_result({
        "suite": "ChromeBench",
        "failures": ["unauthorized irreversible action"],
        "metrics": {"task_success": 1, "recovery_rate": 1, "approval_precision": 1, "latency_ms": 10},
    })
    assert result["ok"] is False
    assert result["hard_failures"]


def test_distributed_training_receipt_is_truthful(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "seed.md").write_text("AURO training seed", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt = write_launch_receipt(corpus, receipt_path)
    contract = DistributedTrainContract()
    assert contract.parameter_count == 14_339_691_520
    assert receipt["corpus_manifest"]["sha256"]
    assert receipt["promotion_boundary"]["full_parameter_training_started"] is False
    assert receipt["promotion_boundary"]["production_promoted"] is False
    loaded = json.loads(receipt_path.read_text())
    assert loaded["evidence_sha256"] == receipt["evidence_sha256"]
