from auro_native_llm.brain import HIMBrain


def test_canonical_topology_has_exactly_44_regions():
    brain=HIMBrain(); snap=brain.snapshot()
    assert snap["region_count"] == 44
    assert len({x["abbreviation"] for x in snap["topology"]}) == 44


def test_cycle_is_bounded_salience_routed_and_hash_linked():
    brain=HIMBrain(); first=brain.cycle("urgent security risk: verify and execute",importance=1,execute_requested=True)
    second=brain.cycle("remember this language context",importance=.7)
    assert first.route == "execute"
    assert first.receipt_hash != second.receipt_hash
    assert all(0 <= x <= 1 for x in brain.snapshot()["activations"].values())
    assert second.dominant_system in brain.snapshot()["systems"]


def test_state_persists_without_mesie(tmp_path):
    path=tmp_path/"brain.json"; brain=HIMBrain(path); brain.cycle("build and verify a document",importance=.8)
    restored=HIMBrain(path)
    assert restored.snapshot()["cycle"] == 1
    assert restored.snapshot()["working_memory"] == ["build and verify a document"]
    assert restored.legacy_parity()["canonical_count"] == 44
