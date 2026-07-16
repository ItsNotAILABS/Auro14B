from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


class PolicyViolation(RuntimeError):
    """Raised when a launch command violates the governed runner policy."""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Allowlist policy for local sovereign training execution.

    The runner never invokes a shell. The first command token must be an
    approved executable, arguments are passed as an argv list, and working
    directories must remain inside one of the configured roots.
    """

    allowed_executables: frozenset[str] = field(
        default_factory=lambda: frozenset({"python", "python3", "torchrun"})
    )
    allowed_roots: tuple[Path, ...] = field(default_factory=lambda: (Path.cwd(),))
    denied_arguments: frozenset[str] = field(
        default_factory=lambda: frozenset({"sudo", "su", "rm", "mkfs", "shutdown", "reboot"})
    )
    max_arguments: int = 256

    @classmethod
    def from_roots(
        cls,
        roots: Iterable[str | Path],
        allowed_executables: Iterable[str] | None = None,
    ) -> "ExecutionPolicy":
        executables = (
            frozenset(allowed_executables)
            if allowed_executables is not None
            else frozenset({"python", "python3", "torchrun"})
        )
        return cls(
            allowed_executables=executables,
            allowed_roots=tuple(Path(root).expanduser().resolve() for root in roots),
        )

    def validate(self, command: Sequence[str], cwd: str | Path) -> None:
        if not command:
            raise PolicyViolation("Command cannot be empty")
        if len(command) > self.max_arguments:
            raise PolicyViolation(f"Command exceeds {self.max_arguments} arguments")

        executable = Path(command[0]).name
        if executable not in self.allowed_executables:
            raise PolicyViolation(f"Executable is not allowed: {executable}")

        for argument in command[1:]:
            token = Path(argument).name
            if token in self.denied_arguments:
                raise PolicyViolation(f"Denied command argument: {argument}")
            if "\x00" in argument:
                raise PolicyViolation("NUL bytes are forbidden in arguments")

        resolved_cwd = Path(cwd).expanduser().resolve()
        if not any(_is_relative_to(resolved_cwd, root) for root in self.allowed_roots):
            raise PolicyViolation(f"Working directory is outside governed roots: {resolved_cwd}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
