from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionPolicy:
    allowed_languages: tuple[str, ...] = ("python",)
    timeout_seconds: float = 5.0
    max_output_bytes: int = 128_000
    max_source_bytes: int = 256_000
    allow_network: bool = False


@dataclass(frozen=True)
class ExecutionReceipt:
    language: str
    return_code: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str
    source_sha256: str
    command: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class ExecutionHarness:
    """Shell-free execution harness for generated benchmark code.

    It uses an argv allowlist, isolated temporary directory, minimal environment,
    timeouts, output caps, and receipts. Network isolation is policy-declared;
    operating-system sandboxing can be layered underneath through MESIE runners.
    """

    def __init__(self, policy: ExecutionPolicy = ExecutionPolicy()) -> None:
        self.policy = policy

    def run(self, language: str, source: str, *, stdin: str = "") -> ExecutionReceipt:
        if language not in self.policy.allowed_languages:
            raise ValueError(f"language not allowed: {language}")
        encoded = source.encode("utf-8")
        if len(encoded) > self.policy.max_source_bytes:
            raise ValueError("source exceeds execution policy")
        if language != "python":
            raise ValueError(f"executor not implemented for language: {language}")

        source_sha = hashlib.sha256(encoded).hexdigest()
        with tempfile.TemporaryDirectory(prefix="auro-exec-") as directory:
            root = Path(directory)
            program = root / "main.py"
            program.write_text(source, encoding="utf-8")
            command = ("python", "-I", "-B", str(program))
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "HOME": str(root),
                "TMPDIR": str(root),
                "AURO_NETWORK_POLICY": "allow" if self.policy.allow_network else "deny",
            }
            started = time.monotonic()
            timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    input=stdin,
                    text=True,
                    capture_output=True,
                    timeout=self.policy.timeout_seconds,
                    shell=False,
                    check=False,
                )
                return_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                return_code = -1
                stdout = _decode(exc.stdout)
                stderr = _decode(exc.stderr)
            duration = time.monotonic() - started

        cap = self.policy.max_output_bytes
        return ExecutionReceipt(
            language=language,
            return_code=return_code,
            timed_out=timed_out,
            duration_seconds=round(duration, 6),
            stdout=stdout[:cap],
            stderr=stderr[:cap],
            source_sha256=source_sha,
            command=command,
        )

    @staticmethod
    def write_receipt(receipt: ExecutionReceipt, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = receipt.to_dict()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
