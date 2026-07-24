"""Bridge multi-repo harvest into training + work text APIs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from auro_native_llm.corpus.harvest import CorpusIndex, harvest_all, default_roots, harvest_paths

# process-wide cache
_INDEX: Optional[CorpusIndex] = None


def get_index(*, refresh: bool = False, include_github: bool = True) -> CorpusIndex:
    global _INDEX
    if _INDEX is not None and not refresh:
        return _INDEX
    cache = Path.home() / ".auro_corpus" / "index.json"
    if not refresh and cache.exists():
        try:
            _INDEX = CorpusIndex.load(cache)
            if _INDEX.documents:
                return _INDEX
        except Exception:
            pass
    # Prefer local harvest first so training never blocks on git clone storms.
    # GitHub clones are opt-in and only when include_github=True.
    try:
        _INDEX = harvest_all(include_github_clones=bool(include_github), max_files=2000)
    except Exception:
        from auro_native_llm.corpus.harvest import harvest_paths, default_roots

        roots = [r for r in default_roots()[:3] if Path(r).exists()]
        _INDEX = harvest_paths(roots, max_files=400, max_total_chars=2_000_000)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _INDEX.save(cache)
    except Exception:
        pass
    return _INDEX


def collect_corpus_texts(
    root: Optional[str | Path] = None,
    max_files: int = 500,
    max_chars: int = 1_500_000,
    *,
    multi_repo: bool = True,
    include_github: bool = True,
) -> List[str]:
    """Drop-in replacement for model.corpus.collect_corpus_texts — multi-repo by default."""
    seeds = [
        "Auro is a native language model family. Compute plane is MESIE. "
        "ItsNotAILABS / FreddyCreates / Medina repos form the training corpus. "
        "Golden ratio phi, Latin ratio lumen ordo, Sanskrit rta satya, Nahuatl teotl.",
        "MESIE Multi-Element Spectral Intelligence Engine. NOVA protocol. PARALLAX clearing. "
        "LOOM memory. CAPSULA. MatDaemon. NEXUS MCP. Career MCP triple. Phantom SDK.",
        "Native sovereign AI: receipts, gates, doctrine, multi-embedded sub-agents, Chrome CDP, "
        "Monaco, Jupyter, self-spinning MCP, continuous mind training.",
    ]
    if not multi_repo:
        # single-root fallback
        from auro_native_llm.model import corpus as legacy

        # avoid recursion if we re-export — call harvest_paths only
        roots = [Path(root)] if root else default_roots()[:1]
        idx = harvest_paths(roots, max_files=max_files, max_total_chars=max_chars)
        return seeds + idx.texts(max_chars=max_chars)

    if root is not None:
        idx = harvest_paths([Path(root)], max_files=max_files, max_total_chars=max_chars)
    else:
        idx = get_index(include_github=include_github)
        # trim to budget
        texts = seeds + idx.texts(max_chars=max_chars)
        # also enforce max_files-ish by truncating list
        return texts[: max(len(seeds) + max_files, len(seeds) + 50)]

    return seeds + idx.texts(max_chars=max_chars)


def collect_work_corpus(max_chars: int = 800_000) -> List[str]:
    """Corpus slices optimized for work agents (code-heavy)."""
    idx = get_index()
    code_docs = [d for d in idx.documents if d.kind == "code"]
    doc_docs = [d for d in idx.documents if d.kind == "doc"]
    # interleave code + docs
    out: List[str] = []
    total = 0
    i = j = 0
    while total < max_chars and (i < len(code_docs) or j < len(doc_docs)):
        if i < len(code_docs):
            block = code_docs[i].training_block()
            out.append(block)
            total += len(block)
            i += 1
        if j < len(doc_docs) and total < max_chars:
            block = doc_docs[j].training_block()
            out.append(block)
            total += len(block)
            j += 1
    return out
