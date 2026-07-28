import hashlib
import json
from pathlib import Path

from scripts.inventory_auro_checkpoints import inspect_checkpoint, inventory


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dummy_manifest_and_weights_are_not_evidence_complete(tmp_path):
    checkpoint = tmp_path / "Auro-2B-dummy"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"model_id": "Auro-2B"}), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"dummy-weights")

    result = inspect_checkpoint(checkpoint)
    assert result["artifact_present"] is True
    assert result["integrity_verified"] is False
    assert result["promotion_ready"] is False
    assert result["evidence_complete"] is False
    assert "no tokenizer custody artifact" in result["blockers"]
    assert any("hash agreement" in blocker for blocker in result["blockers"])


def test_hash_bound_geometry_evaluation_and_signed_promotion_are_required(tmp_path):
    checkpoint = tmp_path / "Auro-2B-promoted"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    tokenizer = checkpoint / "tokenizer.json"
    weights.write_bytes(b"real-test-fixture-weights")
    tokenizer.write_text('{"version":1}', encoding="utf-8")
    (checkpoint / "evaluation.json").write_text(json.dumps({"all_passed": True}), encoding="utf-8")
    manifest = {
        "model_id": "Auro-2B",
        "parameter_count": 2_000_000_000,
        "geometry": {
            "hidden_dim": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "vocab_size": 65536,
            "max_seq_len": 8192,
        },
        "files": {
            "model.safetensors": sha(weights),
            "tokenizer.json": sha(tokenizer),
        },
        "promotion_status": "promoted",
        "authorized_by": "constitutional-gate",
        "signature": "a" * 64,
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_checkpoint(checkpoint)
    assert result["integrity_verified"] is True
    assert result["evaluation_verified"] is True
    assert result["signed_promotion"] is True
    assert result["promotion_ready"] is True
    assert result["evidence_complete"] is True

    report = inventory(tmp_path)
    assert report["auro_2b_artifact_present"] is True
    assert report["auro_2b_integrity_verified"] is True
    assert report["auro_2b_promotion_ready"] is True
