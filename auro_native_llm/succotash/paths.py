"""Locate / materialize FreddyCreates/potential-succotash."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

SUCCOTASH_URL = "https://github.com/FreddyCreates/potential-succotash"
SUCCOTASH_REPO = "FreddyCreates/potential-succotash"

_HOME = Path.home()
_CACHE_CLONE = _HOME / ".auro_corpus" / "github" / "FreddyCreates" / "potential-succotash"

# Preferred local search order
_CANDIDATES = (
    _CACHE_CLONE,
    _HOME / "Documents" / "GitHub" / "potential-succotash",
    _HOME / "Documents" / "GitHub" / "FreddyCreates" / "potential-succotash",
    _HOME / "potential-succotash",
    Path(__file__).resolve().parents[3] / "potential-succotash",
)


def succotash_root() -> Optional[Path]:
    """Return existing potential-succotash root if present."""
    for p in _CANDIDATES:
        if p.exists() and (p / "README.md").exists():
            return p.resolve()
        # shallow marker: registers alone are enough
        if p.exists() and (p / "AI_Model_Families_Register.csv").exists():
            return p.resolve()
    return None


def ensure_succotash(
    *,
    clone: bool = True,
    depth: int = 1,
    timeout: int = 600,
) -> Path:
    """Return path to potential-succotash, shallow-cloning if needed."""
    existing = succotash_root()
    if existing is not None:
        return existing
    if not clone:
        raise FileNotFoundError(
            f"{SUCCOTASH_REPO} not found under candidates; clone={clone}"
        )
    dest = _CACHE_CLONE
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        return dest.resolve()
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            str(depth),
            "--single-branch",
            f"{SUCCOTASH_URL}.git",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if not dest.exists():
        raise FileNotFoundError(f"failed to clone {SUCCOTASH_URL} → {dest}")
    return dest.resolve()
