"""Load Sovereign doctrine as provenance-bound Auro training material."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from auro_foundry.corpus import CorpusBuilder

CONTRACT_PATH = Path("integration/training-contract.v1.json")
EXPECTED_SCHEMA = "sovereign.training.contract.v1"
EXPECTED_REPOSITORY = "FreddyCreates/sovereign"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_value(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode:
        return None
    return completed.stdout.strip() or None


def discover_sovereign_root(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for name in ("AURO_SOVEREIGN_ROOT", "SOVEREIGN_ROOT"):
        if os.environ.get(name):
            candidates.append(Path(os.environ[name]))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root.parent / "sovereign",
        Path.home() / "sovereign",
        Path.home() / "Documents" / "GitHub" / "sovereign",
    ])
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if (resolved / CONTRACT_PATH).is_file():
            return resolved
    return None


@dataclass(frozen=True)
class SovereignBinding:
    root: Path
    contract: dict[str, Any]
    commit: str
    remote: str | None
    dirty: bool
    records: tuple[dict[str, Any], ...]
    skipped: int
    redactions: int

    @property
    def contract_sha256(self) -> str:
        return _sha256((self.root / CONTRACT_PATH).read_bytes())

    def training_blocks(self, max_blocks: int | None = None) -> list[str]:
        records = self.records if max_blocks is None else self.records[:max_blocks]
        return [record["text"] for record in records]

    def receipt(self) -> dict[str, Any]:
        total_bytes = sum(int(record["bytes"]) for record in self.records)
        payload = {
            "schema": "auro.sovereign.binding.v1",
            "contract_id": self.contract["contract_id"],
            "contract_sha256": self.contract_sha256,
            "repository": self.contract["repository"],
            "commit": self.commit,
            "remote": self.remote,
            "dirty": self.dirty,
            "records": len(self.records),
            "text_bytes": total_bytes,
            "skipped": self.skipped,
            "redactions": self.redactions,
            "files": [
                {
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in self.records
            ],
            "attribution": self.contract["attribution"],
        }
        payload["receipt_sha256"] = _sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return payload

    def write_receipt(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.receipt(), indent=2) + "\n", encoding="utf-8")
        return target


def _iter_contract_paths(root: Path, globs: Iterable[str], excluded: set[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file() or any(part in excluded for part in path.parts):
                continue
            relative = path.relative_to(root).as_posix()
            paths[relative] = path
    return [paths[key] for key in sorted(paths)]


def bind_sovereign(
    root: str | Path | None = None,
    *,
    required: bool = True,
    max_file_bytes: int = 2 * 1024 * 1024,
) -> SovereignBinding | None:
    resolved = discover_sovereign_root(root)
    if resolved is None:
        if required:
            raise FileNotFoundError(
                "Sovereign training contract not found. Set AURO_SOVEREIGN_ROOT or pass --sovereign-root."
            )
        return None

    contract_file = resolved / CONTRACT_PATH
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Unsupported Sovereign contract schema: {contract.get('schema')!r}")
    if contract.get("repository") != EXPECTED_REPOSITORY:
        raise ValueError(f"Unexpected Sovereign repository: {contract.get('repository')!r}")

    missing = [
        path for path in contract.get("required_files", [])
        if not (resolved / path).is_file()
    ]
    if missing:
        raise ValueError(f"Sovereign checkout is missing required files: {', '.join(missing)}")

    commit = _git_value(resolved, "rev-parse", "HEAD") or "unversioned-local-source"
    remote = _git_value(resolved, "remote", "get-url", "origin")
    dirty = bool(_git_value(resolved, "status", "--porcelain"))
    excluded = set(contract.get("exclude_parts", []))
    records: list[dict[str, Any]] = []
    skipped = 0
    redactions = 0
    for path in _iter_contract_paths(resolved, contract.get("include_globs", []), excluded):
        raw = path.read_bytes()
        if not raw or len(raw) > max_file_bytes or b"\x00" in raw[:4096]:
            skipped += 1
            continue
        text = raw.decode("utf-8", errors="replace")
        text, count = CorpusBuilder._redact(text)
        redactions += count
        normalized = CorpusBuilder._normalize(text)
        if not normalized:
            skipped += 1
            continue
        relative = path.relative_to(resolved).as_posix()
        digest = _sha256(normalized.encode("utf-8"))
        records.append({
            "path": relative,
            "sha256": digest,
            "bytes": len(normalized.encode("utf-8")),
            "text": (
                f"<|repository:{EXPECTED_REPOSITORY}|>\n"
                f"<|commit:{commit}|>\n"
                f"<|path:{relative}|>\n{normalized}\n"
            ),
        })
    if required and not records:
        raise ValueError("Sovereign contract selected no usable training records")
    return SovereignBinding(
        root=resolved,
        contract=contract,
        commit=commit,
        remote=remote,
        dirty=dirty,
        records=tuple(records),
        skipped=skipped,
        redactions=redactions,
    )
