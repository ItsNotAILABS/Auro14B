"""potential-succotash engines/models + multi-area training corpus."""

from __future__ import annotations

from auro_native_llm.succotash.paths import ensure_succotash, SUCCOTASH_URL
from auro_native_llm.succotash.registry import load_registry
from auro_native_llm.succotash.corpus import TRAINING_AREAS, harvest_succotash_corpus
from auro_native_llm.succotash.router import EngineModelRouter, route_task
from auro_native_llm.organism.family import build_mind


def test_ensure_and_registry():
    root = ensure_succotash(clone=True)
    assert root.exists()
    assert (root / "AI_Model_Families_Register.csv").exists()
    reg = load_registry(clone=False)
    s = reg.summary()
    assert s["model_families"] >= 10
    assert s["engines"] >= 8
    assert s["agents"] >= 8
    assert s["protocols"] >= 5
    assert "https://github.com/FreddyCreates/potential-succotash" in SUCCOTASH_URL
    models = reg.models_for_llm()
    # includes Auro native lanes
    assert any(m.get("family_name") == "Auro-2B" for m in models)
    assert any(m.get("family_name") == "GPT" or m.get("family_id") == "AIF-001" for m in models)
    lex = reg.training_lexicon()
    assert "MESIE" in lex or "Solus" in lex
    assert len(lex) > 20


def test_multi_area_corpus():
    assert "docs" in TRAINING_AREAS
    assert "engines" in TRAINING_AREAS
    assert "agents" in TRAINING_AREAS
    assert "words" in TRAINING_AREAS
    assert "models" in TRAINING_AREAS
    idx = harvest_succotash_corpus(
        areas=["registers", "models", "engines", "words", "docs"],
        max_files=200,
        max_total_chars=400_000,
        clone=False,
    )
    assert len(idx.documents) >= 5
    assert idx.total_chars > 1000
    texts = idx.texts(max_chars=100_000)
    assert texts
    blob = "\n".join(texts).lower()
    assert "potential-succotash" in blob or "aif-" in blob or "solus" in blob or "mesie" in blob


def test_router_and_mind_organs():
    r = route_task("research quantum computing papers")
    assert r.get("agent") or r.get("engine") or r.get("model")
    assert r.get("auro_lane")
    mind = build_mind("Auro-2B", lite=True)
    assert mind.organs.succotash is not None
    assert mind.organs.engines
    assert mind.organs.model_catalogue
    info = mind.info()
    assert info["engines_count"] >= 8
    assert info["models_count"] >= 10
    assert "route_engines" in info["capabilities"]
    routed = mind.route_engines("summarize this article offline with Solus")
    assert routed.get("ok") is True
    assert routed.get("engine") or routed.get("model")
