import hashlib
import hmac
import json
from pathlib import Path

from auro_native_llm.substrate import (
    build_constitutional_checkpoint,
    write_constitutional_manifest,
)
from scripts.inventory_auro_checkpoints import (
    THROUGH_2B_LANES,
    TRIAD_VARIANTS,
    inspect_checkpoint,
    inventory,
)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(payload: dict) -> dict:
    value = dict(payload)
    value["receipt_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def geometry() -> dict:
    return {
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "ffn_dim": 1024,
        "vocab_size": 4096,
        "max_seq_len": 1024,
    }


def write_legacy_promoted_checkpoint(
    root: Path,
    model_id: str,
    parameter_target: int,
    signing_key: str,
) -> Path:
    checkpoint_id = f"{model_id}-fixture-1"
    checkpoint = root / checkpoint_id
    checkpoint.mkdir(parents=True)

    weights = checkpoint / "model.safetensors"
    tokenizer = checkpoint / "tokenizer.json"
    config = checkpoint / "config.json"
    training = checkpoint / "training_receipt.json"
    evaluation = checkpoint / "evaluation.json"

    weights.write_bytes(f"fixture-weights:{model_id}".encode("utf-8"))
    tokenizer.write_text(
        json.dumps({"schema": "fixture.tokenizer.v1", "model_id": model_id}),
        encoding="utf-8",
    )
    config.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "parameter_target": parameter_target,
                **geometry(),
            }
        ),
        encoding="utf-8",
    )
    training.write_text(
        json.dumps(
            receipt(
                {
                    "schema": "auro.training.execution-receipt.v1",
                    "checkpoint_id": checkpoint_id,
                    "output_checkpoint": str(checkpoint),
                    "ok": True,
                }
            )
        ),
        encoding="utf-8",
    )
    evaluation.write_text(
        json.dumps(
            {
                "schema": "auro.exact-checkpoint-evaluation.v1",
                "checkpoint_id": checkpoint_id,
                "all_passed": True,
            }
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema": "auro.legacy-checkpoint-manifest.v2",
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "parameter_count": parameter_target,
        "parameter_target": parameter_target,
        "geometry": geometry(),
        "files": {
            item.name: sha(item)
            for item in (weights, tokenizer, config, training, evaluation)
        },
        "promotion_status": "promoted",
        "authorized_by": "test-constitutional-gate",
    }
    manifest["signature"] = hmac.new(
        signing_key.encode("utf-8"),
        canonical(manifest),
        hashlib.sha256,
    ).hexdigest()
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint


def test_dummy_manifest_and_weights_are_not_evidence_complete(tmp_path):
    checkpoint = tmp_path / "Auro-2B-dummy"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"model_id": "Auro-2B"}),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(b"dummy-weights")

    result = inspect_checkpoint(checkpoint)
    assert result["artifact_present"] is True
    assert result["integrity_verified"] is False
    assert result["promotion_ready"] is False
    assert result["evidence_complete"] is False
    assert "no tokenizer custody artifact" in result["blockers"]
    assert any("hash agreement" in blocker for blocker in result["blockers"])


def test_fake_64_character_signature_no_longer_counts_as_promotion(tmp_path):
    key = "test-signing-key-which-is-long-enough"
    checkpoint = write_legacy_promoted_checkpoint(
        tmp_path,
        "Auro-2B",
        2_000_000_000,
        key,
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["signature"] = "a" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_checkpoint(checkpoint, promotion_signing_key=key)
    assert result["integrity_verified"] is True
    assert result["evaluation_verified"] is True
    assert result["training_provenance_verified"] is True
    assert result["promotion_signature_present"] is True
    assert result["promotion_signature_verified"] is False
    assert result["signed_promotion"] is False
    assert result["promotion_ready"] is False


def test_hash_bound_training_evaluation_and_real_hmac_promotion_are_required(tmp_path):
    key = "test-signing-key-which-is-long-enough"
    checkpoint = write_legacy_promoted_checkpoint(
        tmp_path,
        "Auro-2B",
        2_000_000_000,
        key,
    )

    without_key = inspect_checkpoint(checkpoint)
    assert without_key["integrity_verified"] is True
    assert without_key["evaluation_verified"] is True
    assert without_key["training_provenance_verified"] is True
    assert without_key["promotion_signature_present"] is True
    assert without_key["promotion_signature_verified"] is False
    assert without_key["promotion_ready"] is False

    result = inspect_checkpoint(checkpoint, promotion_signing_key=key)
    assert result["integrity_verified"] is True
    assert result["evaluation_verified"] is True
    assert result["training_provenance_verified"] is True
    assert result["signed_promotion"] is True
    assert result["promotion_ready"] is True
    assert result["evidence_complete"] is True

    report = inventory(tmp_path, promotion_signing_key=key)
    assert report["auro_2b_artifact_present"] is True
    assert report["auro_2b_integrity_verified"] is True
    assert report["auro_2b_promotion_ready"] is True


def test_canonical_constitutional_checkpoint_is_verified_with_authority(tmp_path):
    key = "constitutional-test-key-which-is-long-enough"
    model_id = "Auro-500M"
    checkpoint_id = "Auro-500M-constitutional-1"
    checkpoint = tmp_path / checkpoint_id
    checkpoint.mkdir()

    (checkpoint / "weights.npz").write_bytes(b"constitutional-fixture-weights")
    (checkpoint / "tokenizer.json").write_text(
        json.dumps({"schema": "fixture.tokenizer.v1"}),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "parameter_target": 500_000_000,
                **geometry(),
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "meta.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "num_params": 500_000_000,
                "parameter_target": 500_000_000,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "training_receipt.json").write_text(
        json.dumps(
            receipt(
                {
                    "schema": "auro.training.execution-receipt.v1",
                    "checkpoint_id": checkpoint_id,
                    "output_checkpoint": str(checkpoint),
                    "ok": True,
                }
            )
        ),
        encoding="utf-8",
    )
    (checkpoint / "evaluation.json").write_text(
        json.dumps(
            {
                "schema": "auro.exact-checkpoint-evaluation.v1",
                "checkpoint_id": checkpoint_id,
                "all_passed": True,
            }
        ),
        encoding="utf-8",
    )

    files = [
        "weights.npz",
        "tokenizer.json",
        "config.json",
        "meta.json",
        "training_receipt.json",
        "evaluation.json",
    ]
    constitutional = build_constitutional_checkpoint(
        root=checkpoint,
        checkpoint_id=checkpoint_id,
        checkpoint_class="weights",
        model_id=model_id,
        files=files,
        rollback={"safe_checkpoint_id": "Auro-500M-safe"},
        evidence={
            "protected_capabilities_pass": True,
            "rollback_target": "Auro-500M-safe",
            "rollback_verified": True,
        },
        promotion_requested=True,
        signing_key=key,
        authorized_by="test-constitutional-gate",
    )
    write_constitutional_manifest(checkpoint, constitutional)

    result = inspect_checkpoint(checkpoint, promotion_signing_key=key)
    assert result["manifest_verification"]["kind"] == "constitutional"
    assert result["manifest_verification"]["seal_matches"] is True
    assert result["manifest_verification"]["protocols_verified"] is True
    assert result["promotion_signature_verified"] is True
    assert result["promotion_ready"] is True


def test_ship_through_2b_requires_all_atomic_lanes_and_three_specialists(tmp_path):
    key = "release-family-key-which-is-long-enough"
    targets = {
        "Auro-156K": 156_000,
        "Auro-250M": 250_000_000,
        "Auro-500M": 500_000_000,
        "Auro-2B": 2_000_000_000,
        "Auro-500M-SENSUS": 500_000_000,
        "Auro-500M-PRAXIS": 500_000_000,
        "Auro-500M-VERBUM": 500_000_000,
    }
    for model_id, target in targets.items():
        write_legacy_promoted_checkpoint(tmp_path, model_id, target, key)

    report = inventory(tmp_path, promotion_signing_key=key)
    assert set(report["through_2b"]) == set(THROUGH_2B_LANES)
    assert set(report["triad"]) == set(TRIAD_VARIANTS)
    assert report["through_2b_promotion_ready"] is True
    assert report["triad_promotion_ready"] is True
    assert report["ship_through_2b_ready"] is True
    assert all(item["promotion_ready"] for item in report["through_2b"].values())
    assert all(item["promotion_ready"] for item in report["triad"].values())
