from auro_native_llm.model.taxonomy import (
    MODEL_CLASSES,
    ModelClass,
    classify_parameter_count,
    release_ladder,
)


def test_taxonomy_boundaries_are_complete_and_non_overlapping():
    cases = [
        (1, ModelClass.ATOMIC),
        (156_000, ModelClass.ATOMIC),
        (999_999_999, ModelClass.ATOMIC),
        (1_000_000_000, ModelClass.MICRO),
        (2_000_000_000, ModelClass.MICRO),
        (4_999_999_999, ModelClass.MICRO),
        (5_000_000_000, ModelClass.CORE),
        (8_000_000_000, ModelClass.CORE),
        (10_000_000_000, ModelClass.ORCHESTRATOR),
        (14_000_000_000, ModelClass.ORCHESTRATOR),
        (30_000_000_000, ModelClass.FRONTIER),
        (100_000_000_000, ModelClass.FRONTIER),
    ]
    for count, expected in cases:
        assert classify_parameter_count(count) is expected
    assert len(MODEL_CLASSES) == 5


def test_invalid_parameter_count_is_rejected():
    for count in (0, -1, -156_000):
        try:
            classify_parameter_count(count)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {count}")


def test_release_ladder_uses_canonical_classes_and_claim_boundaries():
    ladder = release_ladder()
    assert ladder["Auro-156K"]["model_class"] == "atomic"
    assert ladder["Auro-2B"]["model_class"] == "micro"
    assert ladder["Auro-4B"]["model_class"] == "micro"
    assert ladder["Auro-8B"]["model_class"] == "core"
    assert ladder["Auro-14B"]["model_class"] == "orchestrator"
    assert ladder["Auro-100B"]["model_class"] == "frontier"
    assert "not a finished 14B checkpoint" in ladder["Auro-14B"]["release_policy"]
    assert "architecture target only" in ladder["Auro-100B"]["release_policy"]


def test_release_ladder_returns_a_copy():
    first = release_ladder()
    first["Auro-156K"]["role"] = "mutated"
    second = release_ladder()
    assert second["Auro-156K"]["role"] != "mutated"
