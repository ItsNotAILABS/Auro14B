from __future__ import annotations

from pathlib import Path

from auro_foundry.benchmarks import BenchmarkCase, BenchmarkRunner, LmEvalBridge, score_response
from auro_foundry.coding_harness import CodingHarness, CodingTask, extract_source
from auro_foundry.execution import ExecutionHarness, ExecutionPolicy
from auro_foundry.federation import FederationManifest
from auro_foundry.workers import WorkerRegistry


class FakeGenerator:
    def generate(self, prompt: str, **kwargs) -> str:
        if "17 * 19" in prompt:
            return "323"
        if "all A are B" in prompt:
            return "no"
        if "Return the sum" in prompt:
            return "def solution(a, b):\n    return a + b"
        return "NOVA REDACT"


def test_federation_and_workers(tmp_path: Path) -> None:
    federation = FederationManifest()
    payload = federation.to_dict()
    assert payload["schema"] == "auro.federation.v1"
    assert payload["counts"]["repositories"] >= 20
    assert len(payload["sha256"]) == 64
    assert federation.selected("corpus")[0].priority >= federation.selected("corpus")[-1].priority
    path = federation.write(tmp_path / "federation.json")
    assert path.exists()
    loaded = FederationManifest.load(path)
    assert len(loaded.lanes) == len(federation.lanes)

    registry = WorkerRegistry()
    order = registry.topological_order()
    assert len(order) == 16
    assert len(set(order)) == len(order)
    assert "BENC" in registry.plan("benchmark")
    assert "CODEX" in registry.plan("code")
    assert "SUCC" in registry.plan("release")
    assert registry.write(tmp_path / "workers.json").exists()

    for index in range(120):
        lane = federation.lanes[index % len(federation.lanes)]
        assert lane.repository.startswith("ItsNotAILABS/")
        assert lane.priority > 0


def test_execution_and_coding_harness(tmp_path: Path) -> None:
    executor = ExecutionHarness(ExecutionPolicy(timeout_seconds=2.0))
    receipt = executor.run("python", "print(6 * 7)")
    assert receipt.return_code == 0
    assert receipt.stdout.strip() == "42"
    assert not receipt.timed_out
    assert len(receipt.source_sha256) == 64

    timeout = executor.run("python", "while True: pass")
    assert timeout.timed_out
    assert timeout.return_code == -1

    assert extract_source("```python\ndef x():\n    return 1\n```").startswith("def x")
    task = CodingTask("sum", "Return the sum of two numbers.", "assert solution(4, 5) == 9")
    result = CodingHarness(FakeGenerator().generate, executor).run([task], output_path=tmp_path / "coding.json")
    assert result["summary"]["passed"] == 1
    assert result["summary"]["pass_rate"] == 1.0
    assert (tmp_path / "coding.json").exists()


def test_benchmarks_and_lm_eval_command(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase("math", "math", "Answer: 17 * 19", "323", "numeric"),
        BenchmarkCase("logic", "reasoning", "If all A are B and no B are C", "no"),
        BenchmarkCase("root", "first-party", "Name the root", "NOVA"),
    ]
    payload = BenchmarkRunner(FakeGenerator()).run(cases, output_path=tmp_path / "bench.json")
    assert payload["summary"]["passed"] == 3
    assert payload["summary"]["accuracy"] == 1.0
    assert (tmp_path / "bench.json").exists()
    assert score_response("answer 55", "55", "numeric")
    assert score_response("NOVA root", "nova", "contains")
    assert score_response("YES", "^yes$", "regex")

    command = LmEvalBridge("lm-eval").command(
        base_url="http://127.0.0.1:8090",
        model_id="Auro",
        tasks=["mmlu", "gsm8k"],
        output_dir=tmp_path / "lm-eval",
        limit=10,
        chat=True,
    )
    assert command[0] == "lm-eval"
    assert "local-chat-completions" in command
    assert "mmlu,gsm8k" in command
    assert "--log_samples" in command
