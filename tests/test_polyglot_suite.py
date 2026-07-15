"""Tests for AISVectorPolyglot polyglot integration engine."""

from data import list_references, load_reference_record
from mesie.polyglot import AISVectorPolyglotSuite, RuntimeId, SUITE_NAME, run_integration_suite
from mesie.engines.registry import build_default_registry


def _refs():
    return [load_reference_record(n) for n in list_references()]


def test_suite_name():
    s = AISVectorPolyglotSuite()
    assert s.health().name == SUITE_NAME


def test_all_runtimes_validate():
    refs = _refs()
    s = AISVectorPolyglotSuite()
    for rid in (RuntimeId.PYTHON, RuntimeId.RUST, RuntimeId.JULIA, RuntimeId.MOTOKO, RuntimeId.TYPESCRIPT):
        resp = s.validate(refs[0], runtime=rid)
        assert resp.ok, f"{rid}: {resp.error}"


def test_parity_matrix():
    refs = _refs()
    s = AISVectorPolyglotSuite()
    p = s.parity_matrix(refs[0], refs[1])
    assert p["parity_ok"]
    assert len(p["results"]) == 5


def test_vector_bridge():
    refs = _refs()
    s = AISVectorPolyglotSuite()
    s.index_corpus(refs)
    q = s.vector_query(refs[0])
    assert q.neighbors
    assert q.embedding


def test_third_party_tools():
    s = AISVectorPolyglotSuite()
    schema = s.third_party.openai_tools_schema()
    names = {t["function"]["name"] for t in schema}
    assert "mesie_match_spectra" in names
    assert "mesie_validate_spectrum" in names


def test_polyglot_engine_on_bus():
    reg = build_default_registry()
    assert "polyglot" in reg.names()


def test_integration_suite():
    report = run_integration_suite(_refs())
    assert report.total == 20
    assert report.passed == report.total