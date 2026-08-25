import hashlib
import json
from pathlib import Path

from auro_native_llm.release_train import (
    BASE_LANES,
    RELEASE_SEQUENCE,
    TRIAD_LANES,
    build_release_train,
)


def manifest(path: Path, schema: str) -> Path:
    payload = {
        "schema": schema,
        "content_sha256": hashlib.sha256(schema.encode()).hexdigest(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def lane(model_id: str, *, promoted: bool = False, integrity: bool = False):
    return {
        "model_id": model_id,
        "candidate_count": 1 if promoted or integrity else 0,
        "artifact_present": promoted or integrity,
        "integrity_verified": promoted or integrity,
        "training_provenance_verified": promoted,
        "evaluation_verified": promoted,
        "promotion_ready": promoted,
        "best_candidate": f"/checkpoints/{model_id}" if promoted or integrity else None,
        "blockers": [] if promoted else ["checkpoint not promoted"],
    }


def fake_inventory(*, promoted=()):
    promoted = set(promoted)
    through = {
        model_id: lane(model_id, promoted=model_id in promoted)
        for model_id in BASE_LANES
    }
    triad = {
        model_id: lane(model_id, promoted=model_id in promoted)
        for model_id in TRIAD_LANES
    }
    return {
        "schema": "auro.checkpoint.inventory.v3",
        "root": "/checkpoints",
        "through_2b": through,
        "triad": triad,
        "claim_boundary": "fixture inventory",
    }


def by_id(plan):
    return {item["lane"]["model_id"]: item for item in plan["lanes"]}


def test_missing_evidence_produces_explicit_nonexecuting_release_plan(tmp_path):
    plan = build_release_train(fake_inventory())
    lanes = by_id(plan)

    assert plan["release_sequence"] == list(RELEASE_SEQUENCE)
    assert plan["release_ready"] is False
    assert plan["training_executed"] is False
    assert plan["checkpoints_created"] is False
    assert plan["checkpoints_promoted"] is False
    assert lanes["Auro-156K"]["state"] == "missing"
    assert lanes["Auro-156K"]["training_job"]["approved"] is False
    assert lanes["Auro-156K"]["training_job"]["execution_ready"] is False
    assert "corpus manifest is not supplied or readable" in lanes["Auro-156K"]["execution_blockers"]
    assert "tokenizer manifest is not supplied or readable" in lanes["Auro-156K"]["execution_blockers"]
    assert lanes["Auro-500M-SENSUS"]["training_job"]["command"] is None
    assert any("adapter trainer" in value for value in lanes["Auro-500M-SENSUS"]["execution_blockers"])


def test_base_jobs_become_operator_runnable_only_with_input_custody(tmp_path):
    corpus = manifest(tmp_path / "corpus.json", "auro.corpus.manifest.v1")
    tokenizer = manifest(tmp_path / "tokenizer.json", "auro.tokenizer.manifest.v2")
    plan = build_release_train(
        fake_inventory(),
        corpus_manifest=corpus,
        tokenizer_manifest=tokenizer,
    )
    lanes = by_id(plan)

    for model_id in BASE_LANES[:-1]:
        job = lanes[model_id]["training_job"]
        assert job["command"][:4] == ["python", "-m", "auro_native_llm.model.train", "--model"]
        assert job["execution_ready"] is True
        assert job["approved"] is False
        assert job["corpus_manifest_sha256"]
        assert job["tokenizer_manifest_sha256"]

    # The parent remains blocked until every atomic and specialist prerequisite is promoted.
    parent = lanes["Auro-2B"]
    assert parent["training_job"]["execution_ready"] is False
    assert set(parent["missing_prerequisites"]) == {
        "Auro-156K",
        "Auro-250M",
        "Auro-500M",
        *TRIAD_LANES,
    }


def test_promoted_atomic_and_triad_lanes_unlock_parent_training(tmp_path):
    corpus = manifest(tmp_path / "corpus.json", "auro.corpus.manifest.v1")
    tokenizer = manifest(tmp_path / "tokenizer.json", "auro.tokenizer.manifest.v2")
    prerequisites = {
        "Auro-156K",
        "Auro-250M",
        "Auro-500M",
        *TRIAD_LANES,
    }
    plan = build_release_train(
        fake_inventory(promoted=prerequisites),
        corpus_manifest=corpus,
        tokenizer_manifest=tokenizer,
    )
    parent = by_id(plan)["Auro-2B"]

    assert parent["missing_prerequisites"] == []
    assert parent["training_job"]["execution_ready"] is True
    assert parent["next_action"] == "run-bounded-family-training-job"
    assert plan["release_ready"] is False


def test_all_seven_promoted_identities_make_release_group_ready(tmp_path):
    corpus = manifest(tmp_path / "corpus.json", "auro.corpus.manifest.v1")
    tokenizer = manifest(tmp_path / "tokenizer.json", "auro.tokenizer.manifest.v2")
    plan = build_release_train(
        fake_inventory(promoted=RELEASE_SEQUENCE),
        corpus_manifest=corpus,
        tokenizer_manifest=tokenizer,
    )

    assert plan["release_ready"] is True
    assert plan["checkpoints_promoted"] is True
    assert all(item["state"] == "promoted" for item in plan["lanes"])
    assert all(item["training_job"]["execution_ready"] is False for item in plan["lanes"])


def test_release_plan_is_deterministic_for_the_same_inventory_and_inputs(tmp_path):
    corpus = manifest(tmp_path / "corpus.json", "auro.corpus.manifest.v1")
    tokenizer = manifest(tmp_path / "tokenizer.json", "auro.tokenizer.manifest.v2")
    inventory = fake_inventory(promoted={"Auro-156K"})

    first = build_release_train(
        inventory,
        corpus_manifest=corpus,
        tokenizer_manifest=tokenizer,
    )
    second = build_release_train(
        inventory,
        corpus_manifest=corpus,
        tokenizer_manifest=tokenizer,
    )
    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
