from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RepositoryLane:
    repository: str
    role: str
    priority: int = 50
    enabled: bool = True
    visibility: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    contribution_types: tuple[str