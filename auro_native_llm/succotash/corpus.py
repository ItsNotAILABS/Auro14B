"""Multi-area training corpus from potential-succotash.

Covers docs, words/lexicon, agents, engines, models, protocols, SDKs,
research, governance, extensions, workers — all training domains.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from auro_native_llm.corpus.harvest import CorpusDocument, CorpusIndex, harvest_paths
from auro_native_llm.succotash.paths import ensure_succotash, succotash_root
from auro_native_llm.succotash.registry import load_registry

# All training areas → subpaths under potential-succotash
TRAINING_AREAS: Dict[str, Dict[str, Any]] = {
    "registers": {
        "desc": "Model/engine/protocol/extension CSV+JSON registers",
        "globs": ["*.csv", "*.json", "README.md", "COMMERCIAL.md", "CONTRIBUTING.md"],
        "kinds": ("doc", "config"),
    },
    "docs": {
        "desc": "Architecture docs and reports",
        "dirs": ["docs"],
    },
    "research": {
        "desc": "Research papers and theory",
        "dirs": ["research"],
    },
    "models": {
        "desc": "AI model families + multimodal + frontier registers as text",
        "files": [
            "AI_Model_Families_Register.csv",
            "Multimodal_Families_Register.csv",
            "Frontend_Frontier_100_Register.csv",
            "Phantom_Blockchain_Model_Register.csv",
            "SDK_Model_Manifest.json",
        ],
    },
    "engines": {
        "desc": "Compute engines + ai-model-engines SDK",
        "dirs": ["sdk/engines", "sdk/ai-model-engines", "sdk/runtime"],
    },
    "agents": {
        "desc": "Autonomous agents (researcher, crawler, corpus, memoria…)",
        "dirs": ["sdk/agents"],
    },
    "protocols": {
        "desc": "Sovereign protocols + register",
        "dirs": ["protocols"],
        "files": ["AI_Protocols_Register.csv"],
    },
    "sdk": {
        "desc": "All SDKs (memory, routing, enterprise, windows…)",
        "dirs": ["sdk"],
    },
    "extensions": {
        "desc": "Browser extensions (Sonic Ninja organism panels)",
        "dirs": ["extensions"],
    },
    "organism": {
        "desc": "Organism runtime + CLI",
        "dirs": ["organism", "organism-cli"],
    },
    "governance": {
        "desc": "Governance framework + architectural laws",
        "dirs": ["governance"],
        "files": ["Architectural_Laws_Register.csv"],
    },
    "memory": {
        "desc": "Memory Temple CIVOS-PRIME",
        "dirs": ["memory_temple", "sdk/sovereign-memory-sdk"],
    },
    "defense": {
        "desc": "Defense organism / Sentry",
        "dirs": ["defense-organism"],
    },
    "workers": {
        "desc": "Cloudflare edge workers",
        "dirs": ["workers"],
    },
    "workflows": {
        "desc": "Automation workflows",
        "dirs": ["workflows"],
    },
    "marketplace": {
        "desc": "Organism marketplace + call marketplace",
        "files": [
            "Organism_Marketplace_Register.csv",
            "CallMarketplaceSpec_v1.json",
            "AI_Extensions_Register.csv",
        ],
        "dirs": ["sdk/organism-marketplace"],
    },
    "architecture": {
        "desc": "Architecture diagrams/specs",
        "dirs": ["architecture", "atlas"],
    },
    "builder": {
        "desc": "Production-grade builder",
        "dirs": ["production-grade-builder"],
    },
    "phantom": {
        "desc": "Phantom native / QSHA",
        "dirs": ["phantom_native", "phantom_qsha"],
    },
    "words": {
        "desc": "Lexicon from registers (capabilities, rings, placement)",
        "synthetic": True,
    },
    "examples": {
        "desc": "Examples / academic sticks",
        "dirs": ["examples"],
    },
}


def _read_file(path: Path, limit: int = 80_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _doc(path: Path, root: Path, kind: str, source: str = "succotash") -> Optional[CorpusDocument]:
    text = _read_file(path)
    if len(text.strip()) < 40:
        return None
    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel = path.name
    return CorpusDocument(
        path=rel,
        repo="potential-succotash",
        kind=kind,
        text=text,
        chars=len(text),
        source=source,
    )


def harvest_succotash_corpus(
    *,
    areas: Optional[Sequence[str]] = None,
    max_files: int = 2500,
    max_total_chars: int = 6_000_000,
    clone: bool = True,
    per_area_cap: int = 200,
) -> CorpusIndex:
    """Harvest multi-area training corpus from potential-succotash."""
    root = ensure_succotash(clone=clone)
    chosen = list(areas) if areas else list(TRAINING_AREAS.keys())
    docs: List[CorpusDocument] = []
    total = 0
    seen: set[str] = set()

    # Prefer structured harvest of whole tree with high per-repo cap
    # but also emit area-tagged synthetic blocks for routing.
    area_roots: List[Path] = []
    for area in chosen:
        spec = TRAINING_AREAS.get(area) or {}
        if spec.get("synthetic"):
            continue
        for d in spec.get("dirs") or []:
            p = root / d
            if p.exists():
                area_roots.append(p)
        for f in spec.get("files") or []:
            p = root / f
            if p.is_file():
                doc = _doc(p, root, "config" if p.suffix in {".csv", ".json"} else "doc")
                if doc and doc.path not in seen:
                    docs.append(doc)
                    seen.add(doc.path)
                    total += doc.chars
        for g in spec.get("globs") or []:
            for p in root.glob(g):
                if p.is_file():
                    doc = _doc(p, root, "config" if p.suffix in {".csv", ".json"} else "doc")
                    if doc and doc.path not in seen:
                        docs.append(doc)
                        seen.add(doc.path)
                        total += doc.chars

    # Bulk harvest from area directories (round-robin via harvest_paths)
    if area_roots and len(docs) < max_files and total < max_total_chars:
        bulk = harvest_paths(
            area_roots,
            max_files=max_files - len(docs),
            max_total_chars=max_total_chars - total,
            source="succotash",
            per_repo_cap=per_area_cap,
        )
        for d in bulk.documents:
            key = f"{d.repo}:{d.path}"
            if key in seen:
                continue
            # retag repo for clarity
            d.repo = "potential-succotash"
            d.source = "succotash"
            docs.append(d)
            seen.add(key)
            total += d.chars
            if len(docs) >= max_files or total >= max_total_chars:
                break

    # Words / lexicon area
    if "words" in chosen or areas is None:
        try:
            reg = load_registry(clone=False)
            lexicon = reg.training_lexicon(max_items=500)
            block = (
                "[SUCCOTASH LEXICON words/capabilities/rings/models/engines/agents]\n"
                + "\n".join(lexicon)
                + "\n[/SUCCOTASH LEXICON]"
            )
            docs.append(
                CorpusDocument(
                    path="registers/lexicon.words.txt",
                    repo="potential-succotash",
                    kind="doc",
                    text=block,
                    chars=len(block),
                    source="succotash",
                )
            )
            # Engine/model catalogue as training doc
            cat = (
                "[SUCCOTASH ENGINES+MODELS for Auro LLM routing]\n"
                + json_safe_catalogue(reg)
                + "\n[/SUCCOTASH ENGINES+MODELS]"
            )
            docs.append(
                CorpusDocument(
                    path="registers/engines_models.catalogue.txt",
                    repo="potential-succotash",
                    kind="doc",
                    text=cat,
                    chars=len(cat),
                    source="succotash",
                )
            )
        except Exception:
            pass

    return CorpusIndex(
        documents=docs,
        roots=[str(root)],
        harvested_at=time.time(),
    )


def json_safe_catalogue(reg) -> str:
    lines = ["# Models"]
    for m in reg.models_for_llm()[:80]:
        lines.append(
            f"- {m.get('family_id')} {m.get('family_name')} alpha={m.get('alpha_model')} "
            f"cap={m.get('primary_capability')} ring={m.get('ring_affinity')} "
            f"wire={m.get('wire_protocol')}"
        )
    lines.append("# Engines")
    for e in reg.engines_for_llm()[:60]:
        lines.append(
            f"- {e.get('engine_id')} {e.get('name')} kind={e.get('kind')} "
            f"path={e.get('path')} :: {e.get('description')}"
        )
    lines.append("# Agents")
    for a in reg.agents_for_llm()[:40]:
        lines.append(f"- {a.get('agent_id')} {a.get('name')} role={a.get('role')}")
    lines.append("# Protocols")
    for p in reg.protocols[:30]:
        lines.append(
            f"- {p.get('protocol_id')} {p.get('protocol_name')} "
            f"fn={p.get('primary_function')}"
        )
    return "\n".join(lines)


def collect_area_texts(
    area: str,
    *,
    max_chars: int = 400_000,
    clone: bool = True,
) -> List[str]:
    """Training texts for one area."""
    idx = harvest_succotash_corpus(areas=[area], max_total_chars=max_chars, clone=clone)
    return idx.texts(max_chars=max_chars)


def collect_all_area_texts(
    *,
    max_chars: int = 1_500_000,
    clone: bool = True,
    areas: Optional[Sequence[str]] = None,
) -> List[str]:
    """All-area training texts from potential-succotash."""
    idx = harvest_succotash_corpus(
        areas=areas,
        max_total_chars=max_chars,
        clone=clone,
    )
    return idx.texts(max_chars=max_chars)


def area_manifest() -> Dict[str, Any]:
    root = succotash_root()
    return {
        "source": "https://github.com/FreddyCreates/potential-succotash",
        "root": str(root) if root else None,
        "areas": {
            k: {"desc": v.get("desc"), "dirs": v.get("dirs"), "files": v.get("files")}
            for k, v in TRAINING_AREAS.items()
        },
    }
