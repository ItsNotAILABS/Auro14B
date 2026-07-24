"""Dependency-free runner for the repository's plain-function production tests."""
from __future__ import annotations

import inspect
import runpy
import sys
import tempfile
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    passed = 0
    with tempfile.TemporaryDirectory(prefix="auro-gate-") as root:
        for path in sorted(Path("tests").glob("test_*.py")):
            namespace = runpy.run_path(str(path))
            for name, function in sorted(namespace.items()):
                if not name.startswith("test_") or not callable(function):
                    continue
                parameters = inspect.signature(function).parameters
                if not parameters:
                    function()
                elif tuple(parameters) == ("tmp_path",):
                    case = Path(root) / f"{path.stem}-{name}"
                    case.mkdir(parents=True)
                    function(case)
                else:
                    raise RuntimeError(f"Unsupported test fixture: {path}:{name}{inspect.signature(function)}")
                passed += 1
                print(f"PASS {path.name}::{name}")
    print(f"AURO_PRODUCTION_GATE_PASSED tests={passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
