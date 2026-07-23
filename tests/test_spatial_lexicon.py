from auro_native_llm.language.spatial_lexicon import ExecutiveComposer, SpatialLexicon, stable_vector


def test_stable_vector_is_deterministic_and_spatial():
    assert stable_vector("retrieval") == stable_vector("retrieval")
    assert stable_vector("retrieval") != stable_vector("authorization")
    assert abs(sum(x * x for x in stable_vector("retrieval")) - 1.0) < 1e-6


def test_lexicon_scans_words_letters_and_neighbors():
    lex = SpatialLexicon()
    receipt = lex.ingest("LangChain SSRF hardening blocks unsafe redirects and DNS rebinding.", source="security")
    assert receipt["tokens"] >= 8
    assert "ssrf" in lex.lexemes
    assert lex.lexemes["ssrf"].neighbors
    assert lex.retrieve("unsafe SSRF redirect", top_k=1)[0]["source_id"] == receipt["source_id"]


def test_composer_prefers_direct_grounded_answer_and_marks_uncertainty():
    composer = ExecutiveComposer()
    result = composer.compose(
        "Can we claim this checkpoint is production-ready?",
        [
            "Auro uses MESIE and GHOST and has many architecture targets.",
            "I cannot verify production readiness because the checkpoint files, hashes, training history, and evaluations were not inspected.",
        ],
        require_uncertainty=True,
    )
    assert result["text"].startswith("I cannot verify")
    assert result["score"]["uncertainty_marked"] == 1.0


def test_deduplicate_removes_near_duplicate_paragraphs():
    composer = ExecutiveComposer()
    text = "The proxy enforces authentication and budgets.\n\nThe proxy enforces authentication and budgets.\n\nRate limits are separate controls."
    clean = composer.deduplicate(text)
    assert clean.count("The proxy enforces") == 1
    assert "Rate limits" in clean
