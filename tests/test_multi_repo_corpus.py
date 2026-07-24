"""Multi-repo MESIE corpus harvest from local Medina monorepos."""

from __future__ import annotations

from pathlib import Path

from auro_native_llm.corpus.harvest import default_roots, harvest_paths, CorpusIndex
from auro_native_llm.corpus.bridge import collect_corpus_texts, collect_work_corpus


def test_default_roots_nonempty():
    roots = default_roots()
    assert any(Path(r).exists() for r in roots)


def test_harvest_local_has_docs_and_code():
    roots = default_roots()[:6]
    idx = harvest_paths(roots, max_files=200, max_total_chars=500_000)
    assert isinstance(idx, CorpusIndex)
    assert len(idx.documents) >= 10
    kinds = {d.kind for d in idx.documents}
    assert "doc" in kinds or "code" in kinds
    stats = idx.stats()
    assert stats["total_chars"] > 1000
    assert stats["by_repo"]


def test_collect_corpus_texts_multi_repo():
    texts = collect_corpus_texts(max_files=100, max_chars=200_000, multi_repo=True, include_github=False)
    assert len(texts) >= 5
    blob = "\n".join(texts[:20])
    # should see corpus markers or MESIE/Auro content
    assert "CORPUS" in blob or "MESIE" in blob or "Auro" in blob


def test_collect_work_corpus():
    texts = collect_work_corpus(max_chars=100_000)
    assert len(texts) >= 1


def test_corpus_search():
    roots = default_roots()[:5]
    idx = harvest_paths(roots, max_files=150, max_total_chars=300_000)
    hits = idx.search("spectral", top_k=5)
    # may be empty on sparse trees but should not crash
    assert isinstance(hits, list)
