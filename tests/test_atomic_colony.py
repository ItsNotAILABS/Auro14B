from auro_native_llm.atomic_colony import AtomicColony


def test_four_atomic_specialists_preserve_lineage_and_receipt():
    colony = AtomicColony.repository_audit_colony("checkpoints/open/HIM-native-v0")
    result = colony.run("search code architecture contradiction synthesis", lambda specialist, task: f"{specialist.role}:{task}")
    assert len(result["specialists"]) == 4
    assert len({item["specialist_id"] for item in result["specialists"]}) == 4
    assert result["base_checkpoints"] == ["checkpoints/open/HIM-native-v0"]
    assert len(result["receipt_sha256"]) == 64
    assert "weight specialization requires" in result["claim_boundary"]


def test_colony_requires_exactly_four_specialists():
    try:
        AtomicColony(())
    except ValueError as exc:
        assert "exactly four" in str(exc)
    else:
        raise AssertionError("expected exact-four guard")
