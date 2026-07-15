from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

SUPPORTED = {".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json"}


def iter_owned_files(roots: Iterable[str | Path]) -> Iterator[Path]:
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix.lower() in SUPPORTED:
            yield root
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED and ".git" not in path.parts:
                    yield path


def content_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
