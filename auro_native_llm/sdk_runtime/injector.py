"""Discover and inject SDKs from all harvested GitHub repos into runtime.

At mind start / harness start:
  - scan local monorepo roots + ~/.auro_corpus/github
  - find packages named *sdk*, sdk/, packages with pyproject/package.json
  - register in RepoSDKRegistry
  - add paths to sys.path for Python import when safe
  - expose sdk.list / sdk.get / sdk.search via portal + python organ globals
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


_HOME = Path.home()
_REGISTRY: Optional["RepoSDKRegistry"] = None


@dataclass
class SDKPackage:
    name: str
    path: str
    repo: str
    kind: str  # python | js | julia | other
    entry: str = ""
    docs: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "repo": self.repo,
            "kind": self.kind,
            "entry": self.entry,
            "docs": self.docs[:200],
        }


class RepoSDKRegistry:
    def __init__(self) -> None:
        self.packages: Dict[str, SDKPackage] = {}
        self.injected_paths: List[str] = []
        self.scanned_at: float = 0.0

    def register(self, pkg: SDKPackage) -> None:
        key = f"{pkg.repo}/{pkg.name}"
        self.packages[key] = pkg

    def list_packages(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.packages.values()]

    def search(self, query: str, top_k: int = 12) -> List[Dict[str, Any]]:
        q = query.lower().split()
        scored = []
        for p in self.packages.values():
            blob = f"{p.name} {p.repo} {p.kind} {p.docs}".lower()
            s = sum(1 for t in q if t in blob)
            if s:
                scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        return [p.to_dict() for _, p in scored[:top_k]]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        for p in self.packages.values():
            if p.name == name or p.name.endswith(name):
                return p.to_dict()
        return None

    def stats(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        by_repo: Dict[str, int] = {}
        for p in self.packages.values():
            by_kind[p.kind] = by_kind.get(p.kind, 0) + 1
            by_repo[p.repo] = by_repo.get(p.repo, 0) + 1
        return {
            "schema": "auro.sdk_runtime.v1",
            "n_packages": len(self.packages),
            "by_kind": by_kind,
            "by_repo": dict(sorted(by_repo.items(), key=lambda x: -x[1])[:40]),
            "injected_paths": len(self.injected_paths),
            "scanned_at": self.scanned_at,
        }


def _candidate_roots() -> List[Path]:
    roots = [
        _HOME / "Documents" / "GitHub",
        _HOME / ".auro_corpus" / "github",
        _HOME / "GPTREPO",
        Path(__file__).resolve().parents[2],  # Auro14B
    ]
    return [r for r in roots if r.exists()]


def _looks_like_sdk(path: Path) -> bool:
    name = path.name.lower()
    if "sdk" in name or name in ("sdk", "lib", "packages"):
        return True
    if (path / "pyproject.toml").exists() or (path / "package.json").exists():
        # only if under an sdk-ish parent or name contains mesie/nova/auro
        if any(k in name for k in ("mesie", "nova", "auro", "medina", "solus", "phantom", "loom")):
            return True
    return False


def scan_sdks(max_packages: int = 400) -> RepoSDKRegistry:
    reg = RepoSDKRegistry()
    seen: Set[str] = set()
    count = 0
    # Always register installed mesie package as primary SDK
    try:
        import mesie as _mesie_pkg

        mp = Path(getattr(_mesie_pkg, "__file__", "") or "").resolve().parent
        if mp.exists():
            reg.register(
                SDKPackage(
                    name="mesie",
                    path=str(mp),
                    repo="mesie-installed",
                    kind="python",
                    entry="package",
                    docs=f"MESIE v{getattr(_mesie_pkg, '__version__', '?')} transformers+intelligence",
                )
            )
            count += 1
    except Exception:
        pass
    for root in _candidate_roots():
        # shallow walk
        try:
            children = list(root.iterdir())
        except Exception:
            continue
        # if root is github cache org folders
        stack = list(children)
        depth_guard = 0
        while stack and count < max_packages and depth_guard < 1200:
            depth_guard += 1
            p = stack.pop()
            try:
                if not p.is_dir():
                    continue
                if p.name in (
                    ".git", "node_modules", "__pycache__", "dist", "build",
                    ".venv", "venv", ".tox", "target", "artifacts", "checkpoints",
                    ".cache", "wheels",
                ):
                    continue
                # avoid resolve() on every node (slow on Windows deep trees)
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                # expand one level of monorepo only (keep scan bounded)
                if p.parent == root or p.parent.parent == root:
                    try:
                        for c in list(p.iterdir())[:80]:
                            if c.is_dir() and c.name not in (".git", "node_modules", "__pycache__"):
                                stack.append(c)
                    except Exception:
                        pass
                if not _looks_like_sdk(p) and "sdk" not in p.name.lower():
                    # still pick *sdk* paths deeper
                    if "sdk" in p.name.lower() or (p / "sdk").is_dir():
                        if (p / "sdk").is_dir():
                            stack.append(p / "sdk")
                    continue
                # classify
                kind = "other"
                entry = ""
                if (p / "pyproject.toml").exists() or any(p.glob("*.py")):
                    kind = "python"
                    entry = "pyproject.toml" if (p / "pyproject.toml").exists() else ""
                if (p / "package.json").exists():
                    kind = "js" if kind == "other" else kind
                    entry = entry or "package.json"
                if any(p.glob("*.jl")) or (p / "Project.toml").exists():
                    kind = "julia" if kind == "other" else kind
                if any(p.glob("*.hs")):
                    kind = "haskell" if kind == "other" else kind
                # docs preview
                docs = ""
                for readme in ("README.md", "readme.md", "README.rst"):
                    rp = p / readme
                    if rp.exists():
                        try:
                            docs = rp.read_text(encoding="utf-8", errors="ignore")[:400]
                        except Exception:
                            pass
                        break
                repo = p.parent.name if p.parent != root else p.name
                # better repo name: walk up for .git
                cur = p
                for _ in range(4):
                    if (cur / ".git").exists():
                        repo = cur.name
                        break
                    if cur.parent == cur:
                        break
                    cur = cur.parent
                pkg = SDKPackage(
                    name=p.name,
                    path=str(p),
                    repo=repo,
                    kind=kind,
                    entry=entry,
                    docs=docs,
                )
                reg.register(pkg)
                count += 1
            except Exception:
                continue
    reg.scanned_at = time.time()
    return reg


def _is_shadow_risk(path: str) -> bool:
    """Avoid injecting corpus trees that override live auro_foundry / mesie."""
    low = path.replace("\\", "/").lower()
    # harvested github mirrors often ship older auro_foundry that hard-imports torch
    if "/.auro_corpus/" in low:
        if "auro14b" in low or "auro_foundry" in low:
            return True
    return False


def inject_repo_sdks(mind: Any = None, *, max_packages: int = 400) -> Dict[str, Any]:
    """Scan all repos, inject Python paths, attach registry to mind organs."""
    global _REGISTRY
    reg = scan_sdks(max_packages=max_packages)
    _REGISTRY = reg

    # Keep live workspace first so local auro_foundry (lazy/no-torch) wins
    live_root = str(Path(__file__).resolve().parents[2])
    if live_root in sys.path:
        sys.path.remove(live_root)
    sys.path.insert(0, live_root)

    injected = 0
    for pkg in reg.packages.values():
        if pkg.kind != "python":
            continue
        path = pkg.path
        parent = str(Path(path).parent)
        for candidate in (path, parent):
            if _is_shadow_risk(candidate):
                continue
            if candidate == live_root:
                continue
            if candidate not in sys.path:
                # append after live root (index 1+) so we never shadow auro_foundry
                sys.path.insert(1, candidate)
                reg.injected_paths.append(candidate)
                injected += 1
                break

    # attach to mind
    if mind is not None:
        mind.organs.sdk_registry = reg  # type: ignore[attr-defined]
        # inject into python organ globals if present
        if getattr(mind.organs, "python", None) is not None:
            # store callable for scripts
            mind.organs.python._sdk = reg  # type: ignore[attr-defined]
        # register MCP tools if portal/mcp exists
        mcp = getattr(mind.organs, "mcp", None)
        if mcp is not None:
            try:
                from auro_native_llm.embedded.mcp_hub import MCPTool

                mcp.hub.register(
                    MCPTool(
                        "sdk.list",
                        "List SDKs injected from all Medina/GitHub repos",
                        lambda a: {"ok": True, "packages": reg.list_packages()[: int(a.get("limit", 50))]},
                    )
                )
                mcp.hub.register(
                    MCPTool(
                        "sdk.search",
                        "Search injected multi-repo SDKs",
                        lambda a: {"ok": True, "hits": reg.search(a.get("query", ""), top_k=int(a.get("top_k", 12)))},
                    )
                )
                mcp.hub.register(
                    MCPTool(
                        "sdk.stats",
                        "SDK runtime injection stats",
                        lambda a: reg.stats(),
                    )
                )
            except Exception:
                pass
        # absorb catalog into trainer
        try:
            from auro_native_llm.organism.self_train import Experience

            if mind.organs.trainer:
                catalog = "\n".join(
                    f"SDK {p['repo']}/{p['name']} kind={p['kind']}"
                    for p in reg.list_packages()[:80]
                )
                mind.organs.trainer.absorb(
                    Experience(
                        text=f"MULTI_REPO_SDK_RUNTIME\n{catalog}",
                        kind="sdk_inject",
                        model_id=mind.model_id,
                        reward=0.9,
                        meta=reg.stats(),
                    )
                )
        except Exception:
            pass

    stats = reg.stats()
    stats["ok"] = True
    stats["packages"] = stats["n_packages"]
    stats["paths_injected"] = injected
    return stats


def get_sdk_registry() -> Optional[RepoSDKRegistry]:
    return _REGISTRY
