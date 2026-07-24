from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .execution import ExecutionHarness


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    prompt: str
    tests: str
    language: str = "python"
    entrypoint: str = "solution"


@dataclass(frozen=True)
class CodingResult:
    task_id: str
    passed: bool
    return_code: int
    timed_out: bool
    generated_source: str
    stdout: str
    stderr: str
    source_sha256: str


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_source(text: str) -> str:
    match = _CODE_FENCE.search(text)
    return (match.group(1) if match else text).strip()


class CodingHarness:
    def __init__(self, generator: Callable[[str], str], executor: ExecutionHarness) -> None:
        self.generator = generator
        self.executor = executor

    def run(self, tasks: Iterable[CodingTask], *, output_path: str | Path | None = None) -> dict:
        results: list[CodingResult] = []
        for task in tasks:
            instruction = (
                f"Write only {task.language} source code. Implement `{task.entrypoint}`.\n"
                f"Task: {task.prompt}\nDo not explain the answer."
            )
            source = extract_source(self.generator(instruction))
            combined = source + "\n\n" + task.tests + "\n"
            receipt = self.executor.run(task.language, combined)
            results.append(CodingResult(
                task_id=task.task_id,
                passed=receipt.return_code == 0 and not receipt.timed_out,
                return_code=receipt.return_code,
                timed_out=receipt.timed_out,
                generated_source=source,
                stdout=receipt.stdout,
                stderr=receipt.stderr,
                source_sha256=receipt.source_sha256,
            ))
        passed = sum(result.passed for result in results)
        payload = {
            "schema": "auro.coding_benchmark.v1",
            "summary": {"tasks": len(results), "passed": passed, "pass_rate": passed / max(len(results), 1)},
            "results": [asdict(result) for result in results],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload


def built_in_smoke_tasks() -> tuple[CodingTask, ...]:
    return (
        CodingTask("add", "Return the sum of two numbers.", "assert solution(2, 3) == 5\nassert solution(-2, 2) == 0"),
        CodingTask("reverse", "Return a string reversed.", "assert solution('auro') == 'orua'\nassert solution('') == ''"),
        CodingTask("fibonacci", "Return the nth Fibonacci number with F(0)=0 and F(1)=1.", "assert solution(0) == 0\nassert solution(1) == 1\nassert solution(10) == 55"),
    )


def load_tasks(path: str | Path) -> list[CodingTask]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("tasks", raw) if isinstance(raw, dict) else raw
    return [CodingTask(**row) for row in rows]
