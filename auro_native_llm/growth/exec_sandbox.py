"""Sandboxed code execution for the coding-with-execution milestone.

The training loop EMBEDS these runtimes: the model writes code, the code
actually runs, and the observed output becomes training feedback. This is
the tool-use milestone made concrete -- execution feedback in the loop,
not static code text.

Isolation honesty: this module provides *process-level* isolation
(subprocess + timeout + output caps + fresh temp working directory). It is
the loop mechanics, NOT a security boundary. Real training runs untrusted
model-generated code inside containers/seccomp/gVisor with network and
filesystem jails. Never point this at adversarial code without that layer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_TIMEOUT_S = 10
MAX_OUTPUT_CHARS = 65_536


@dataclass
class ExecResult:
    ok: bool                 # process ran to completion inside the timeout
    returncode: int          # -1 when the runtime is missing or timed out
    stdout: str
    stderr: str
    timed_out: bool
    available: bool          # False when the runtime binary is missing
    truncated: bool = False
    runtime: str = ""

    @property
    def passed(self) -> bool:
        return self.ok and self.returncode == 0


def _cap(text: str, limit: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) > limit:
        return text[:limit], True
    return text, False


def _run(argv: List[str], code_path: Path, timeout: int, runtime: str) -> ExecResult:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(code_path.parent),
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(code_path.parent),
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired as e:
        out, t1 = _cap((e.stdout or "") if isinstance(e.stdout, str) else "")
        err, t2 = _cap((e.stderr or "") if isinstance(e.stderr, str) else "")
        return ExecResult(ok=False, returncode=-1, stdout=out, stderr=err,
                          timed_out=True, available=True, truncated=t1 or t2,
                          runtime=runtime)
    out, t1 = _cap(proc.stdout or "")
    err, t2 = _cap(proc.stderr or "")
    return ExecResult(ok=True, returncode=proc.returncode, stdout=out, stderr=err,
                      timed_out=False, available=True, truncated=t1 or t2,
                      runtime=runtime)


def run_python(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> ExecResult:
    """Execute Python source in a subprocess; capture stdout/stderr."""
    with tempfile.TemporaryDirectory(prefix="auro_py_") as tmp:
        path = Path(tmp) / "task.py"
        path.write_text(code, encoding="utf-8")
        return _run([sys.executable, str(path)], path, timeout, runtime="python")


def powershell_binary() -> Optional[str]:
    return shutil.which("pwsh") or shutil.which("powershell")


def powershell_available() -> bool:
    return powershell_binary() is not None


def run_powershell(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> ExecResult:
    """Execute PowerShell source in a subprocess; capture stdout/stderr.

    Returns ``available=False`` (not an exception) when no PowerShell
    runtime is installed, so the training loop can route around it.
    """
    binary = powershell_binary()
    if binary is None:
        return ExecResult(ok=False, returncode=-1, stdout="", stderr="",
                          timed_out=False, available=False, runtime="powershell")
    with tempfile.TemporaryDirectory(prefix="auro_ps_") as tmp:
        path = Path(tmp) / "task.ps1"
        path.write_text(code, encoding="utf-8")
        return _run([binary, "-NoProfile", "-NonInteractive", "-File", str(path)],
                    path, timeout, runtime="powershell")


# ----------------------------------------------------------------------------
# Whole-module tasks: complete/extend/debug a full module, then run its tests
# ----------------------------------------------------------------------------

@dataclass
class ModuleTask:
    """One tool-use training task: a module with a HOLE, plus its tests.

    The model receives ``module_src`` (containing the marker
    ``# __AURO_HOLE__`` where code is missing) and must produce a complete
    module. ``evaluate`` writes the completion, runs ``test_src`` against it
    in the sandbox, and reports pass/fail with the observed output.
    """

    name: str
    runtime: str                    # "python" | "powershell"
    module_src: str                 # full module, with a "# __AURO_HOLE__" marker
    test_src: str                   # test script; exit 0 == pass
    timeout: int = DEFAULT_TIMEOUT_S

    HOLE = "# __AURO_HOLE__"

    def prompt(self) -> str:
        return (f"Complete the module below (replace the {self.HOLE} line).\n"
                f"Output ONLY the finished {self.runtime} source.\n\n{self.module_src}")

    def evaluate(self, completion: str) -> Dict:
        """Run the completed module against its tests in the sandbox."""
        runner = run_python if self.runtime == "python" else run_powershell
        if self.runtime == "powershell" and not powershell_available():
            return {"name": self.name, "passed": False, "skipped": True,
                    "reason": "powershell runtime not installed"}
        with tempfile.TemporaryDirectory(prefix="auro_mod_") as tmp:
            d = Path(tmp)
            (d / ("module.py" if self.runtime == "python" else "module.ps1")).write_text(
                completion, encoding="utf-8")
            test_name = "test_module.py" if self.runtime == "python" else "test_module.ps1"
            (d / test_name).write_text(self.test_src, encoding="utf-8")
            argv = ([sys.executable, test_name] if self.runtime == "python"
                    else [powershell_binary(), "-NoProfile", "-NonInteractive",
                          "-File", test_name])
            result = _run(argv, d / test_name, self.timeout, runtime=self.runtime)
        return {"name": self.name, "passed": result.passed, "skipped": False,
                "returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "timed_out": result.timed_out}


# -- built-in example tasks (mechanics verification, not the curriculum) ------

FIB_MODULE = '''"""Fibonacci module with a missing core."""


# __AURO_HOLE__

def fib_list(n):
    return [fib(i) for i in range(n)]
'''

FIB_TEST = '''import sys
sys.path.insert(0, ".")
from module import fib, fib_list
assert fib(0) == 0 and fib(1) == 1 and fib(10) == 55
assert fib_list(6) == [0, 1, 1, 2, 3, 5]
print("fib module OK")
'''

FIB_REFERENCE = '''"""Fibonacci module with a missing core."""


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_list(n):
    return [fib(i) for i in range(n)]
'''

FIB_BROKEN = '''"""Fibonacci module with a missing core."""


def fib(n):
    return n  # wrong: identity is not fibonacci


def fib_list(n):
    return [fib(i) for i in range(n)]
'''


def example_tasks() -> List[ModuleTask]:
    tasks = [ModuleTask(name="fib-complete", runtime="python",
                        module_src=FIB_MODULE, test_src=FIB_TEST)]
    if powershell_available():
        ps_mod = '# PowerShell module with a hole\n# __AURO_HOLE__\n'
        ps_test = '. ./module.ps1\nif ((Get-Double 21) -ne 42) { exit 1 }\n"ps module OK"\n'
        tasks.append(ModuleTask(name="ps-double", runtime="powershell",
                                module_src=ps_mod, test_src=ps_test))
    return tasks
