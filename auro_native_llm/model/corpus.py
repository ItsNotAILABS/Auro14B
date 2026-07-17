"""Build Auro LM pretrain corpus — multi-repo MESIE/ItsNotAILABS by default.

Delegates to ``auro_native_llm.corpus`` harvest of all Medina monorepos +
optional shallow GitHub clones. Keeps the old function signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

# Re-export multi-repo collector as the primary API
from auro_native_llm.corpus.bridge import collect_corpus_texts  # noqa: F401
from auro_native_llm.corpus.bridge import collect_work_corpus  # noqa: F401


_REPO = Path(__file__).resolve().parents[2]

_DEFAULT_GLOBS = [
    "README.md",
    "native_llm/**/*.md",
    "docs/**/*.md",
    "examples/**/*.py",
    "deliverables/**/*.md",
    "AGENTS.md",
    "CHANGELOG.md",
]


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def collect_corpus_texts_single_repo(
    root: Optional[str | Path] = None,
    max_files: int = 80,
    max_chars: int = 400_000,
) -> List[str]:
    """Legacy single-repo harvest (Auro14B only)."""
    root = Path(root) if root else _REPO
    texts: List[str] = []
    total = 0
    seeds = [
        "Auro is a native language model family. Compute plane is MESIE.",
    ]
    texts.extend(seeds)
    total = sum(len(s) for s in seeds)
    paths = []
    for pattern in _DEFAULT_GLOBS:
        paths.extend(root.glob(pattern))
    uniq = sorted({p.resolve() for p in paths if p.is_file()}, key=lambda p: p.stat().st_size)
    for p in uniq[:max_files]:
        t = _read_text(p)
        if len(t) < 40:
            continue
        texts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return texts


def iter_training_sequences(texts: Iterable[str], tokenizer, max_len: int = 128) -> List[List[int]]:
    seqs: List[List[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_length=None)
        for i in range(0, max(1, len(ids) - 1), max_len):
            chunk = ids[i : i + max_len]
            if len(chunk) < 8:
                continue
            seqs.append(chunk)
    return seqs
