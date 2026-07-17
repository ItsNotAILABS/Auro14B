"""Harvest docs + code from Medina / ItsNotAILABS / FreddyCreates local + cloned repos."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


# File types that carry training value (docs + code)
_INCLUDE_SUFFIXES = {
    ".md", ".txt", ".rst", ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx",
    ".mo", ".rs", ".jl", ".hs", ".go", ".java", ".kt",
    ".toml", ".yaml", ".yml", ".json", ".cff", ".csv",
    ".sh", ".ps1", ".sql", ".css", ".html",
}

_SKIP_DIR_NAMES = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", ".turbo", "coverage", ".pytest_cache", ".mypy_cache",
    "target", "vendor", ".cargo", "eggs", "*.egg-info",
    "package-lock.json",  # not a dir but
}

_SKIP_FILE_NAMES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.lock",
    "poetry.lock", "uv.lock", ".DS_Store",
}

_SKIP_NAME_PARTS = (
    ".stale.", "min.js", ".map", ".min.css", "chrome_data",
)

_SECRET_HINTS = re.compile(
    r"(api[_-]?key|secret|password|private[_-]?key|BEGIN (RSA |OPENSSH )?PRIVATE)",
    re.I,
)

_HOME = Path.home()
_DEFAULT_CACHE = _HOME / ".auro_corpus"


@dataclass
class CorpusDocument:
    path: str
    repo: str
    kind: str  # doc | code | config
    text: str
    chars: int
    source: str  # local | github_clone

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "repo": self.repo,
            "kind": self.kind,
            "chars": self.chars,
            "source": self.source,
            "preview": self.text[:200],
        }

    def training_block(self) -> str:
        return (
            f"[CORPUS repo={self.repo} kind={self.kind} path={self.path}]\n"
            f"{self.text}\n"
            f"[/CORPUS]"
        )


@dataclass
class CorpusIndex:
    documents: List[CorpusDocument] = field(default_factory=list)
    roots: List[str] = field(default_factory=list)
    harvested_at: float = field(default_factory=time.time)

    @property
    def total_chars(self) -> int:
        return sum(d.chars for d in self.documents)

    def texts(self, max_chars: Optional[int] = None) -> List[str]:
        out: List[str] = []
        total = 0
        for d in self.documents:
            block = d.training_block()
            if max_chars is not None and total + len(block) > max_chars:
                remain = max_chars - total
                if remain > 200:
                    out.append(block[:remain])
                break
            out.append(block)
            total += len(block)
        return out

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q = [w for w in query.lower().split() if len(w) > 2]
        if not q:
            return []
        scored = []
        for d in self.documents:
            blob = (d.path + " " + d.repo + " " + d.text[:4000]).lower()
            score = sum(1 for w in q if w in blob)
            if score:
                idx = min((blob.find(w) for w in q if w in blob), default=0)
                # map idx roughly to text
                snip = d.text[max(0, idx - len(d.path)) : max(0, idx - len(d.path)) + 280]
                if not snip:
                    snip = d.text[:280]
                scored.append((score, {
                    "title": f"{d.repo}:{Path(d.path).name}",
                    "url": f"corpus://{d.repo}/{d.path}",
                    "snippet": snip,
                    "source": "mesie_corpus",
                    "repo": d.repo,
                    "kind": d.kind,
                    "score": score,
                }))
        scored.sort(key=lambda x: -x[0])
        return [h for _, h in scored[:top_k]]

    def stats(self) -> Dict[str, Any]:
        by_repo: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for d in self.documents:
            by_repo[d.repo] = by_repo.get(d.repo, 0) + 1
            by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
        return {
            "documents": len(self.documents),
            "total_chars": self.total_chars,
            "roots": list(self.roots),
            "by_repo": dict(sorted(by_repo.items(), key=lambda x: -x[1])[:40]),
            "by_kind": by_kind,
            "harvested_at": self.harvested_at,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # store without full text in summary; separate shards optional
        payload = {
            "stats": self.stats(),
            "documents": [
                {
                    **d.to_dict(),
                    "text": d.text[:120_000],
                }
                for d in self.documents
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = [
            CorpusDocument(
                path=d["path"],
                repo=d["repo"],
                kind=d["kind"],
                text=d.get("text", ""),
                chars=int(d.get("chars", len(d.get("text", "")))),
                source=d.get("source", "local"),
            )
            for d in data.get("documents", [])
        ]
        idx = cls(documents=docs, roots=list(data.get("stats", {}).get("roots", [])))
        return idx


def default_roots() -> List[Path]:
    """Local Medina monorepo roots that typically hold docs + code.

    Primary: FreddyCreates/potential-succotash (engines, models, agents, docs).
    """
    # Prefer potential-succotash first (engines + models + all training areas)
    succotash_candidates = [
        _HOME / ".auro_corpus" / "github" / "FreddyCreates" / "potential-succotash",
        _HOME / "Documents" / "GitHub" / "potential-succotash",
        _HOME / "Documents" / "GitHub" / "FreddyCreates" / "potential-succotash",
    ]
    candidates = [
        *succotash_candidates,
        _HOME / "Documents" / "GitHub",
        _HOME / "GPTREPO",
        _HOME / "Multi-Element-Spectral-Intelligence-Engine-MESIE-",
        _HOME / "MatDaemon",
        _HOME / "Documents" / "Sovereign-OS-OpenSource",
        _HOME / "Downloads" / "neuroemergence-core",
        _HOME / "Downloads" / "meridian",
        _HOME / "Downloads" / "ironclad",
        _HOME / "Downloads" / "world-doctrine-hub",
        _HOME / "Downloads" / "NOVA_CODEX_DESKTOP_SOURCE_v0.3.8",
        _HOME / "Downloads" / "nova_agents_orchestrator_v0_3_2_terminal_fixed_20260705",
        _HOME / ".auro_corpus" / "github",
        Path(__file__).resolve().parents[2],  # Auro14B itself
    ]
    out: List[Path] = []
    seen: Set[str] = set()
    for p in candidates:
        if not p.exists():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _is_skipped_dir(name: str) -> bool:
    if name in _SKIP_DIR_NAMES:
        return True
    if name.endswith(".egg-info"):
        return True
    if ".stale." in name:
        return True
    return False


def _kind_for(path: Path) -> str:
    s = path.suffix.lower()
    if s in {".md", ".txt", ".rst", ".html"}:
        return "doc"
    if s in {".json", ".yaml", ".yml", ".toml", ".cff"}:
        return "config"
    return "code"


def _read(path: Path, limit: int = 80_000) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            raw = path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""
    if _SECRET_HINTS.search(raw[:2000]) and path.suffix.lower() not in {".md", ".py"}:
        # still allow md/py with word "secret" in docs; skip obvious key files
        if path.suffix.lower() in {".pem", ".key", ".env"}:
            return ""
    return raw[:limit]


def _iter_repo_files(repo_root: Path, max_file_bytes: int):
    """Yield candidate files under a repo (pruned walk)."""
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if not _is_skipped_dir(d)]
        for fn in filenames:
            if fn in _SKIP_FILE_NAMES:
                continue
            if any(p in fn for p in _SKIP_NAME_PARTS):
                continue
            path = Path(dirpath) / fn
            if path.suffix.lower() not in _INCLUDE_SUFFIXES:
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            yield path


_GENERIC_LEAF = {
    "src", "scripts", "apps", "lib", "dist", "test", "tests", "docs",
    "platform", "scaffolds", "skills", "desktop", "browser_terminal",
    "codex_runtime", "assets", "public", "static", "components",
}


def _looks_like_project(p: Path) -> bool:
    if (p / ".git").exists():
        return True
    markers = (
        "README.md", "pyproject.toml", "package.json", "Cargo.toml",
        "go.mod", "dfx.json", "CITATION.cff", "AGENTS.md",
    )
    return any((p / m).exists() for m in markers)


def _collect_repo_roots(roots: Sequence[Path]) -> List[Path]:
    """Expand parent folders into individual project roots."""
    repos: List[Path] = []
    seen: Set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            continue
        batch: List[Path] = []
        if (root / ".git").exists() or _looks_like_project(root):
            batch.append(root)
        for p in sorted(root.iterdir()):
            if not p.is_dir() or _is_skipped_dir(p.name):
                continue
            # skip generic leaves unless they look like real projects
            if p.name.lower() in _GENERIC_LEAF and not _looks_like_project(p):
                continue
            if _looks_like_project(p) or any(
                (p / m).exists() for m in ("src", "mesie", "docs", "lib")
            ):
                batch.append(p)
        for r in batch:
            key = str(r.resolve())
            if key not in seen:
                seen.add(key)
                repos.append(r)
    return repos


def harvest_paths(
    roots: Sequence[Path],
    *,
    max_files: int = 4000,
    max_file_bytes: int = 400_000,
    max_total_chars: int = 8_000_000,
    source: str = "local",
    per_repo_cap: int = 120,
) -> CorpusIndex:
    """Round-robin harvest so one fat monorepo cannot starve the rest."""
    docs: List[CorpusDocument] = []
    total = 0
    seen: Set[str] = set()
    repo_roots = _collect_repo_roots(roots)
    # iterators + per-repo counts
    file_iters = {str(r): _iter_repo_files(r, max_file_bytes) for r in repo_roots}
    repo_counts = {str(r): 0 for r in repo_roots}
    active = list(file_iters.keys())

    while active and len(docs) < max_files and total < max_total_chars:
        next_active = []
        for key in active:
            if repo_counts[key] >= per_repo_cap:
                continue
            it = file_iters[key]
            try:
                path = next(it)
            except StopIteration:
                continue
            next_active.append(key)
            rkey = str(path.resolve())
            if rkey in seen:
                continue
            text = _read(path, limit=min(80_000, max_file_bytes))
            if len(text.strip()) < 40:
                continue
            seen.add(rkey)
            repo_root = Path(key)
            try:
                rel = str(path.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                rel = path.name
            doc = CorpusDocument(
                path=rel,
                repo=repo_root.name,
                kind=_kind_for(path),
                text=text,
                chars=len(text),
                source=source,
            )
            docs.append(doc)
            total += doc.chars
            repo_counts[key] += 1
            if len(docs) >= max_files or total >= max_total_chars:
                break
        active = next_active

    return CorpusIndex(documents=docs, roots=[str(r) for r in roots])


def list_github_org_repos(org: str = "ItsNotAILABS", limit: int = 80) -> List[Dict[str, Any]]:
    """List org repos via gh CLI (public names). Falls back to empty on auth failure."""
    try:
        proc = subprocess.run(
            ["gh", "repo", "list", org, "--limit", str(limit), "--json", "name,isPrivate,url,description"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return []
        return json.loads(proc.stdout or "[]")
    except Exception:
        return []


def materialize_github_org(
    org: str = "ItsNotAILABS",
    *,
    cache_dir: Optional[Path] = None,
    include_private: bool = False,
    skip_stale: bool = True,
    max_repos: int = 40,
    depth: int = 1,
) -> List[Path]:
    """Shallow-clone public org repos into ~/.auro_corpus/github/<org>/."""
    cache_dir = Path(cache_dir or (_DEFAULT_CACHE / "github" / org))
    cache_dir.mkdir(parents=True, exist_ok=True)
    repos = list_github_org_repos(org, limit=100)
    paths: List[Path] = []
    count = 0
    for r in repos:
        name = r.get("name") or ""
        if not name:
            continue
        if skip_stale and ".stale." in name:
            continue
        if r.get("isPrivate") and not include_private:
            continue
        dest = cache_dir / name
        if dest.exists() and any(dest.iterdir()):
            paths.append(dest)
            count += 1
            if count >= max_repos:
                break
            continue
        url = r.get("url") or f"https://github.com/{org}/{name}.git"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    str(depth),
                    "--single-branch",
                    url,
                    str(dest),
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if dest.exists():
                paths.append(dest)
                count += 1
        except Exception:
            continue
        if count >= max_repos:
            break
    return paths


def harvest_all(
    *,
    include_github_clones: bool = True,
    orgs: Optional[Sequence[str]] = None,
    max_files: int = 5000,
    max_total_chars: int = 12_000_000,
    clone_max_repos: int = 35,
) -> CorpusIndex:
    """Harvest local Medina trees + optional shallow GitHub clones."""
    roots = default_roots()
    if include_github_clones:
        for org in orgs or ("ItsNotAILABS", "FreddyCreates"):
            try:
                paths = materialize_github_org(org, max_repos=clone_max_repos)
                roots.extend(paths)
            except Exception:
                # still harvest whatever is already in cache
                cache = _DEFAULT_CACHE / "github" / org
                if cache.exists():
                    roots.append(cache)
    # de-dupe roots
    uniq_roots = []
    seen = set()
    for r in roots:
        key = str(Path(r).resolve()) if Path(r).exists() else str(r)
        if key not in seen and Path(r).exists():
            seen.add(key)
            uniq_roots.append(Path(r))
    return harvest_paths(
        uniq_roots,
        max_files=max_files,
        max_total_chars=max_total_chars,
        source="local+github",
    )
