from auro_native_llm.harness.viral import AuroArenaHarness, Challenge, canonical_sha256, launch_challenges


def test_launch_catalog_is_shareable_and_weighted():
    challenges = launch_challenges()
    assert len(challenges) >= 5
    assert {c.category for c in challenges} >= {"research", "chrome", "coding", "iot", "continuity"}
    for challenge in challenges:
        challenge.validate()
        assert abs(sum(challenge.score_weights.values()) - 1.0) < 1e-6


def test_public_receipt_redacts_secrets_and_hashes_run():
    challenge = Challenge("demo", "Demo", "Safe demo", "chrome", 2, {"task_success": .8, "safety": .2})
    harness = AuroArenaHarness()

    def step(index):
        return {
            "observation": "page ready token=super-secret-value",
            "proposal": {"action": "read_title", "api_key": "do-not-publish"},
            "decision": {"allowed": True, "approved": True},
            "result": {"title": "AURO", "done": index == 0},
            "done": True,
        }

    run = harness.run(challenge, step)
    harness.score(challenge, run, {"task_success": 1.0, "safety": 1.0})
    receipt = harness.public_receipt(challenge, run)
    encoded = str(receipt)
    assert "do-not-publish" not in encoded
    assert "super-secret-value" not in encoded
    assert run.score == 100.0
    assert len(receipt["receipt_sha256"]) == 64


def test_policy_denial_stops_run_without_fake_success():
    challenge = Challenge("iot", "IoT", "Governed IoT", "iot", 5, {"policy": 1.0}, requires_approval=True)
    harness = AuroArenaHarness()
    calls = []

    def step(index):
        calls.append(index)
        return {"observation": {}, "proposal": {"action": "actuate"}, "decision": {"allowed": False}, "result": None}

    run = harness.run(challenge, step)
    assert calls == [0]
    assert len(run.steps) == 1
    assert run.steps[0].decision["allowed"] is False


def test_remix_preserves_governance_and_creates_lineage_ready_challenge():
    original = launch_challenges()[3]
    remix = AuroArenaHarness.remix(original, new_id="iot-guardian-community-remix", max_steps=4)
    assert remix.id != original.id
    assert remix.requires_approval is True
    assert remix.score_weights == original.score_weights
    assert remix.max_steps == 4


def test_canonical_hash_is_deterministic():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
