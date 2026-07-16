from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SUPPORTED_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html", ".java",
    ".jl", ".js", ".json", ".jsx", ".md", ".mjs", ".mo", ".py", ".rb",
    ".rs", ".rst", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt",
    ".yaml", ".yml",
}

EXCLUDED_PARTS = {
    ".git", ".github-cache", ".mypy_cache", ".next", ".nova", ".pytest_cache",
    ".venv", "build", "coverage", "dist", "htmlcov", "node_modules", "out",
    "target", "vendor", "venv",
}

EXCLUDED_NAMES = {
    ".env", ".env.local", ".npmrc", "id_rsa", "id_ed25519", "credentials",
    "credentials.json", "secrets.json", "token.json",
}

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
)


@dataclass(frozen=True)
class RepositorySource:
    repository: str
    visibility: str = "unknown"
    ref: str = "main"
    ownership_basis: str = "Medina-owned repository"

    @property
    def slug(self) -> str:
        return self.repository.replace("/", "__")


@dataclass(frozen=True)
class CorpusRecord:
    repository: str
    path: str
    sha256: str
    bytes: int
    quality_score: float
    spectral_signature: dict[str, float]
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


class CorpusBuilder:
    """Build a deduplicated, provenance-preserving corpus from owned repos."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
        max_file_bytes: int = 2 * 1024 * 1024,
        min_text_chars: int = 48,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir or self.output_dir / "repo-cache").expanduser().resolve()
        self.max_file_bytes = int(max_file_bytes)
        self.min_text_chars = int(min_text_chars)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def materialize(self, source: RepositorySource) -> Path:
        target = self.cache_dir / source.slug
        if target.exists() and (target / ".git").exists():
            self._run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", source.ref])
            self._run(["git", "-C", str(target), "reset", "--hard", "FETCH_HEAD"])
            return target
        if target.exists():
            shutil.rmtree(target)
        if shutil.which("gh"):
            command = ["gh", "repo", "clone", source.repository, str(target), "--", "--depth", "1", "--branch", source.ref]
        else:
            command = ["git", "clone", "--depth", "1", "--branch", source.ref, f"https://github.com/{source.repository}.git", str(target)]
        self._run(command)
        return target

    @staticmethod
    def _run(command: Sequence[str]) -> None:
        completed = subprocess.run(list(command), text=True, capture_output=True, check=False, shell=False)
        if completed.returncode:
            message = completed.stderr.strip() or completed.stdout.strip() or "command failed"
            raise RuntimeError(f"{command[0]} failed: {message}")

    @staticmethod
    def discover_org(org: str, *, include_private: bool = True, limit: int = 1000) -> list[RepositorySource]:
        if not shutil.which("gh"):
            raise RuntimeError("GitHub CLI is required for organization discovery: install gh and run gh auth login")
        command = [
            "gh", "repo", "list", org, "--limit", str(limit),
            "--json", "nameWithOwner,visibility,isArchived,defaultBranchRef",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False, shell=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "gh repo list failed")
        rows = json.loads(completed.stdout)
        output: list[RepositorySource] = []
        for row in rows:
            if row.get("isArchived"):
                continue
            visibility = str(row.get("visibility", "unknown")).lower()
            if visibility == "private" and not include_private:
                continue
            branch = (row.get("defaultBranchRef") or {}).get("name") or "main"
            output.append(RepositorySource(row["nameWithOwner"], visibility, branch))
        return sorted(output, key=lambda item: item.repository.lower())

    def build(
        self,
        *,
        repositories: Iterable[RepositorySource] = (),
        local_roots: Iterable[str | Path] = (),
        corpus_name: str = "auro-owned-corpus",
    ) -> dict:
        started = time.time()
        roots: list[tuple[str, Path, str, str]] = []
        for source in repositories:
            roots.append((source.repository, self.materialize(source), source.visibility, source.ownership_basis))
        for root_value in local_roots:
            root = Path(root_value).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(root)
            roots.append((root.name, root, "local", "operator-authorized local source"))

        corpus_path = self.output_dir / "corpus.jsonl"
        manifest_path = self.output_dir / "manifest.json"
        seen: set[str] = set()
        file_count = 0
        byte_count = 0
        redaction_count = 0
        skipped_count = 0
        repo_stats: dict[str, dict[str, int | str]] = {}

        with corpus_path.open("w", encoding="utf-8") as output:
            for repository, root, visibility, ownership in roots:
                stats = {"files": 0, "bytes": 0, "visibility": visibility}
                for path in self._iter_files(root):
                    try:
                        raw = path.read_bytes()
                    except OSError:
                        skipped_count += 1
                        continue
                    if not raw or len(raw) > self.max_file_bytes or b"\x00" in raw[:4096]:
                        skipped_count += 1
                        continue
                    text = raw.decode("utf-8", errors="replace")
                    text, redactions = self._redact(text)
                    redaction_count += redactions
                    if len(text.strip()) < self.min_text_chars:
                        skipped_count += 1
                        continue
                    normalized = self._normalize(text)
                    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    if digest in seen:
                        skipped_count += 1
                        continue
                    seen.add(digest)
                    relative = str(path.relative_to(root)).replace(os.sep, "/")
                    record = CorpusRecord(
                        repository=repository,
                        path=relative,
                        sha256=digest,
                        bytes=len(normalized.encode("utf-8")),
                        quality_score=self._quality_score(normalized, path.suffix.lower()),
                        spectral_signature=self._spectral_signature(normalized),
                        text=f"<|repository:{repository}|>\n<|path:{relative}|>\n{normalized}\n",
                    )
                    output.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
                    file_count += 1
                    byte_count += record.bytes
                    stats["files"] = int(stats["files"]) + 1
                    stats["bytes"] = int(stats["bytes"]) + record.bytes
                repo_stats[repository] = stats

        corpus_sha = self._sha256_file(corpus_path)
        manifest = {
            "schema": "auro.foundry.corpus_manifest.v1",
            "corpus_id": corpus_name,
            "created_at_unix": int(time.time()),
            "duration_seconds": round(time.time() - started, 3),
            "corpus_path": str(corpus_path),
            "corpus_sha256": corpus_sha,
            "records": file_count,
            "text_bytes": byte_count,
            "redactions": redaction_count,
            "skipped": skipped_count,
            "repositories": repo_stats,
            "ownership_policy": "operator-authorized Medina repositories and local roots only",
            "secret_policy": "known credential patterns redacted; secret-like files excluded",
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def _iter_files(self, root: Path) -> Iterator[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name in EXCLUDED_NAMES or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            yield path

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _redact(text: str) -> tuple[str, int]:
        count = 0
        for pattern in SECRET_PATTERNS:
            text, replacements = pattern.subn("<|redacted_secret|>", text)
            count += replacements
        return text, count

    @staticmethod
    def _quality_score(text: str, extension: str) -> float:
        length_score = min(1.0, math.log10(max(len(text), 10)) / 5.0)
        replacement_penalty = min(0.5, text.count("�") / max(len(text), 1) * 20)
        structure = 0.15 if extension in {".py", ".rs", ".go", ".java", ".cpp", ".ts", ".md"} else 0.05
        return round(max(0.0, min(1.0, 0.25 + length_score * 0.6 + structure - replacement_penalty)), 4)

    @staticmethod
    def _spectral_signature(text: str) -> dict[str, float]:
        raw = text.encode("utf-8", errors="replace")
        if not raw:
            return {"entropy": 0.0, "centroid": 0.0, "line_periodicity": 0.0}
        counts = [0] * 256
        for value in raw:
            counts[value] += 1
        total = float(len(raw))
        probabilities = [count / total for count in counts if count]
        entropy = -sum(prob * math.log2(prob) for prob in probabilities) / 8.0
        centroid = sum(index * count for index, count in enumerate(counts)) / (255.0 * total)
        lines = text.splitlines() or [text]
        mean = sum(len(line) for line in lines) / len(lines)
        variance = sum((len(line) - mean) ** 2 for line in lines) / max(len(lines), 1)
        periodicity = 1.0 / (1.0 + math.sqrt(variance))
        return {"entropy": round(entropy, 6), "centroid": round(centroid, 6), "line_periodicity": round(periodicity, 6)}

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
