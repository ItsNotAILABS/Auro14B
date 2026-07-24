from auro_native_llm.language.spatial_lexicon import ExecutiveComposer, LanguageEngine, SpatialLexicon, stable_vector


def test_stable_vector_is_deterministic_and_spatial():
    assert stable_vector("retrieval") == stable_vector("retrieval")
    assert stable_vector("retrieval") != stable_vector("authorization")
    assert abs(sum(value * value for value in stable_vector("retrieval")) - 1.0) < 1e-6


def test_lexicon_preserves_characters_bytes_phrases_and_positions():
    lexicon = SpatialLexicon()
    receipt = lexicon.ingest("LangChain SSRF hardening blocks unsafe redirects and DNS rebinding. ✓", source="security")
    assert receipt["tokens"] >= 8
    assert receipt["utf8_bytes"] >= receipt["characters"]
    assert lexicon.character_counts["✓"] == 1
    assert lexicon.phrase_counts["ssrf hardening"] == 1
    assert lexicon.lexemes["ssrf"].neighbors
    assert lexicon.lexemes["ssrf"].occurrences[0].start >= 0
    assert lexicon.retrieve("unsafe SSRF redirect", top_k=1)[0]["source_id"] == receipt["source_id"]
    assert "SSRF" in lexicon.grab("SSRF redirect", top_k=1)[0]["snippet"]


def test_ingestion_is_idempotent():
    lexicon = SpatialLexicon()
    first = lexicon.ingest("one two three", source="same")
    digest = lexicon.manifest()["sha256"]
    second = lexicon.ingest("one two three", source="same")
    assert second["duplicate"] is True
    assert first["source_id"] == second["source_id"]
    assert lexicon.total_tokens == 3
    assert lexicon.manifest()["sha256"] == digest


def test_snapshot_roundtrip_preserves_manifest(tmp_path):
    lexicon = SpatialLexicon()
    lexicon.ingest("Unicode 日本語 and code_path.py survive.", source="roundtrip")
    destination = tmp_path / "lexicon.json"
    expected = lexicon.save(destination)["manifest"]["sha256"]
    loaded = SpatialLexicon.load(destination)
    assert loaded.manifest()["sha256"] == expected
    assert loaded.retrieve("code path", top_k=1)


def test_composer_combines_distinct_paragraphs_and_marks_uncertainty():
    composer = ExecutiveComposer()
    result = composer.compose(
        "Can we claim this checkpoint is production-ready?",
        [
            "Architecture targets are not benchmark evidence.",
            "I cannot verify production readiness because checkpoint hashes, training history, and official evaluations were not inspected.\n\nA release needs exact tests and failure samples.",
        ],
        evidence=["No official benchmark run is present."],
        require_uncertainty=True,
    )
    assert result["text"].startswith("I cannot verify")
    assert result["score"]["uncertainty_marked"] == 1.0
    assert result["selected_paragraphs"]
    assert result["grabs"]


def test_language_engine_unifies_grab_plan_compose_and_revision():
    engine = LanguageEngine()
    engine.observe("Fair comparison requires fixed prompts, equivalent tools, repeated trials, and exact model IDs.", source="benchmark")
    grabbed = engine.grab("fixed prompts and model IDs")
    planned = engine.plan("Design a fair competitive evaluation", branches=4)
    composed = engine.compose("How should systems be compared?", ["Use fixed prompts and exact model IDs.", "Repeat trials and disclose tool access."], evidence=[grabbed["documents"][0]["text"]])
    revised = engine.revise("How should systems be compared?", composed["text"], ["Add blinded human review and latency accounting."], evidence=[grabbed["documents"][0]["text"]])
    assert grabbed["spans"]
    assert len(planned["branches"]) == 4
    assert composed["text"]
    assert revised["schema"] == "auro.language-revision.v1"
