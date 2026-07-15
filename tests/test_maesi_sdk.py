"""MAESI SDK v1.1 tests."""

import numpy as np

from mesie.sdk import (
    MAESIClient,
    get_research_catalog,
    get_technical_library,
    FastSpectralCompute,
    search_research,
)


def test_knowledge_counts():
    assert len(get_technical_library()) >= 15
    assert len(get_research_catalog()) >= 20


def test_search_research():
    hits = search_research("LSH approximate", top_k=3)
    assert len(hits) >= 1
    assert "LSH" in hits[0].title or "ANN" in hits[0].title


def test_fast_compute_index():
    from mesie.core.records import MultiElementRecord, SpectralComponent

    def rec(i: int):
        f = np.linspace(0.1, 10, 32)
        return MultiElementRecord(
            record_id=f"r{i}",
            components=[SpectralComponent(name="c", frequency=f, amplitude=np.sin(f) + i * 0.01)],
        )

    fc = FastSpectralCompute()
    fc.build_index([rec(0), rec(1), rec(2)])
    hits = fc.cosine_search(rec(0), top_k=2)
    assert hits[0][0] == "r0"
    assert hits[0][1] >= hits[1][1]


def test_maesi_client_run():
    from data import list_references, load_reference_record

    refs = [load_reference_record(n) for n in list_references()]
    report = MAESIClient().run_full(refs, benchmark=True)
    assert report.knowledge.research_entries >= 20
    assert report.speed is not None
    assert report.speed.speedup_ratio >= 1.0