from pathlib import Path

from auro_native_llm.brain_state.runtime_state import PersistentBrainState
from auro_native_llm.eval.auro_eval import AuroEval, default_portfolio_contracts
from auro_native_llm.registry.model_registry import ModelRecord, ModelRegistry


def test_brain_state_persists_and_hashes(tmp_path: Path):
    store = PersistentBrainState(tmp_path / "brain.json")
    first = store.transition(observation="chrome tab loaded", task="research")
    second = store.transition(observation="tool result received", consequence={"ok": True})
    assert second.sequence == first.sequence + 1
    assert second.previous_hash == first.state_hash
    assert second.state_hash != first.state_hash
    loaded = store.load()
    assert loaded.state_hash == second.state_hash
    assert "<brain_state>" in store.inference_context(loaded)


def test_model_registry_is_digest_bound(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.put(ModelRecord(
        model_id="AURO-ST-14B",
        architecture_hash="a" * 64,
        tokenizer_hash="b" * 64,
        corpus_hash="c" * 64,
        weights_hash="d" * 64,
        serving_compatibility=["transformers", "vllm", "tensorrt-llm", "llama.cpp"],
    ))
    assert registry.validate("AURO-ST-14B") is True


def test_auro_eval_receipt_and_suite_contracts():
    harness = AuroEval()
    harness.run("ChromeBench", "policy-boundary", lambda: (True, 1.0, {"approval_gate": True}))
    receipt = harness.receipt()
    assert receipt["passed"] == 1
    assert receipt["failed"] == 0
    assert len(receipt["evidence_sha256"]) == 64
    suites = {item["suite"] for item in default_portfolio_contracts()}
    assert {"ChromeBench", "IoTBench", "RobotBench", "MemoryBench", "SecurityBoundaryBench"} <= suites
