"""Tests for the growth curriculum and sandboxed execution milestone.

Verifies the curriculum machinery (generators produce verifiable problems,
verifiers accept true answers and reject wrong ones, milestones cover the
ladder with explicit done-criteria) and the execution loop mechanics
(sandbox runs code, captures output, judges module completions).
"""

from __future__ import annotations

import numpy as np
import pytest

from auro_native_llm.growth.curriculum import (
    GENERATORS,
    MILESTONES,
    curriculum_summary,
    milestones_for_rung,
)
from auro_native_llm.growth.exec_sandbox import (
    FIB_BROKEN,
    FIB_REFERENCE,
    ModuleTask,
    example_tasks,
    powershell_available,
    run_powershell,
    run_python,
)


def test_generators_produce_verifiable_problems():
    rng = np.random.default_rng(0)
    for domain, gen in GENERATORS.items():
        for _ in range(25):
            p = gen(rng, difficulty=1)
            assert p.domain == domain, domain
            assert p.prompt.endswith("Answer:")
            # the canonical answer verifies...
            assert p.verify(p.answer), (domain, p.prompt, p.answer)
            # ...and a wrong answer does not (numeric: offset; choice: flip)
            wrong = "definitely not the answer 999999"
            if p.answer.strip().lower() in ("yes", "no"):
                wrong = "no" if p.answer == "yes" else "yes"
            assert not p.verify(wrong), (domain, p.prompt, p.answer)


def test_milestones_cover_ladder_with_done_criteria():
    assert len(MILESTONES) == 5
    domains = {m.domain for m in MILESTONES}
    assert domains == {"math", "physics", "architecture", "ancient-geometry", "coding"}
    for m in MILESTONES:
        assert 0 < m.done.threshold < 1
        assert m.done.min_rungs >= 1
        assert m.rung_start <= m.rung_end
        assert m.data_provenance  # provenance always labeled
    # every rung 0..14 has at least one active milestone
    for rung in range(15):
        assert milestones_for_rung(rung), f"rung {rung} has no milestone"
    # coding-with-execution is the late tool-use milestone
    coding = [m for m in MILESTONES if m.domain == "coding"][0]
    assert coding.rung_start >= 8
    assert "sandbox" in coding.description.lower()
    assert "static code text" in coding.description  # the anti-pattern, named
    assert curriculum_summary()  # renders without error


def test_run_python_executes_and_captures():
    r = run_python("print(40 + 2)")
    assert r.available and r.ok and r.passed
    assert r.stdout.strip() == "42"
    assert r.stderr == ""


def test_run_python_captures_errors():
    r = run_python("raise ValueError('boom')")
    assert r.ok  # the process ran; the *program* failed
    assert not r.passed
    assert r.returncode != 0
    assert "ValueError" in r.stderr and "boom" in r.stderr


def test_run_python_timeout():
    r = run_python("import time; time.sleep(30)", timeout=2)
    assert r.timed_out and not r.ok


def test_run_powershell_degrades_gracefully():
    r = run_powershell("Write-Output 'hi'")
    if powershell_available():
        assert r.ok and r.passed and "hi" in r.stdout
    else:
        assert not r.available and not r.ok


def test_module_task_reference_passes_broken_fails():
    tasks = example_tasks()
    fib = [t for t in tasks if t.name == "fib-complete"][0]
    good = fib.evaluate(FIB_REFERENCE)
    bad = fib.evaluate(FIB_BROKEN)
    assert good["passed"] and not good.get("skipped")
    assert not bad["passed"] and not bad.get("skipped")
    assert "fib module OK" in good["stdout"]
    assert bad["stderr"] or bad["returncode"] != 0


def test_module_task_without_hole_still_judged():
    # a completion that ignores the hole but passes tests still passes:
    # the gate judges behavior, not form.
    fib = [t for t in example_tasks() if t.name == "fib-complete"][0]
    alt = FIB_REFERENCE.replace('"""Fibonacci module with a missing core."""',
                                '"""Done differently."""')
    assert fib.evaluate(alt)["passed"]
