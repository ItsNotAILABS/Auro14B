import hashlib
import hmac
import json
from pathlib import Path

import pytest

from auro_native_llm.substrate.checkpoint_constitution import (
    AIOPS_DOMAINS,
    PROTOCOLS,
    ConstitutionalGateError,
    build_constitutional_checkpoint,
    load_and_verify_constitutional_manifest,
    write_constitutional_manifest,
)
from auro_native_llm.substrate.promotion_evidence import (
    PromotionEvidenceError,
    evidence_flags,
    load_signed_promotion_evidence,
)

SIGNING_KEY = "test-signing-key-not-for-production"


def complete_evidence():
    return {
        "resume_state_present": True,
        "optimization_candidate": True,
        "matched_benchmark": True,
        "protected_capabilities_pass": True,
        "continual_learning": True,
        "replay_or_forgetting_eval": True,
        "architecture_changed": True,
        "reversible_module_boundary": True,
        "tool_capabilities_present": True,
        "tool_registry_receipt": True,
        "identity_state_present": True,
        "continuity_parent_known": True,
        "rollback_target": "safe-001",
        "rollback_verified": True,
    }


def signed_receipt(tmp_path: Path):
    payload = {
        "schema": "auro.promotion.evidence.v1",
        "benchmark": {"candidate": True, "passed": True, "artifact_sha256": "a" * 64},
        "rollback": {"checkpoint_id": "safe-001", "passed": True, "artifact_sha256": "b" * 64},
        "capabilities": {"passed": True, "artifact_sha256": "c" * 64},
        "forgetting": {"applicable": True, "passed": True, "artifact_sha256": "d" * 64},
        "architecture": {"changed": True, "rollback_passed": True, "artifact_sha256": "e" * 64},
        "tools": {"present": True, "passed": True, "artifact_sha256": "f" * 64},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["authorization_hmac_sha256"] = hmac.new(SIGNING_KEY.encode(), canonical, hashlib.sha256).hexdigest()
    path = tmp_path / "promotion-evidence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_signed_promotion_manifest_is_authorized_and_loadable(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(
        root=tmp_path, checkpoint_id="auro-001", checkpoint_class="organism",
        model_id="Auro-HIM", files=["weights.npz"], parent_checkpoint_id="safe-001",
        evidence=complete_evidence(), promotion_requested=True,
        signing_key=SIGNING_KEY, authorized_by="release-bot",
    )
    write_constitutional_manifest(tmp_path, checkpoint)
    loaded = load_and_verify_constitutional_manifest(tmp_path, signing_key=SIGNING_KEY)
    assert checkpoint.promotion_status == "promoted"
    assert loaded["authorized"] is True
    assert loaded["authorized_by"] == "release-bot"
    assert {item.protocol for item in checkpoint.protocols} == set(PROTOCOLS)
    assert set(checkpoint.aiops_domains) == set(AIOPS_DOMAINS)


def test_promoted_checkpoint_cannot_be_built_without_authorization_key(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    with pytest.raises(ConstitutionalGateError, match="authorization signing key"):
        build_constitutional_checkpoint(root=tmp_path, checkpoint_id="auro-002",
            checkpoint_class="weights", model_id="Auro", files=["weights.npz"],
            evidence=complete_evidence(), promotion_requested=True)


def test_attacker_cannot_reseal_promoted_manifest_with_plain_sha256(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(root=tmp_path, checkpoint_id="auro-003",
        checkpoint_class="weights", model_id="Auro", files=["weights.npz"],
        evidence=complete_evidence(), promotion_requested=True, signing_key=SIGNING_KEY)
    path = write_constitutional_manifest(tmp_path, checkpoint)
    payload = json.loads(path.read_text())
    payload["model_id"] = "attacker-rewrite"
    unsigned = dict(payload)
    unsigned["manifest_sha256"] = None
    unsigned["authorization_hmac_sha256"] = None
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    payload["authorization_hmac_sha256"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ConstitutionalGateError, match="authorization"):
        load_and_verify_constitutional_manifest(tmp_path, signing_key=SIGNING_KEY)


def test_quarantined_checkpoint_is_rejected_by_default_but_available_for_explicit_analysis(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    evidence = complete_evidence(); evidence["matched_benchmark"] = False
    checkpoint = build_constitutional_checkpoint(root=tmp_path, checkpoint_id="auro-004",
        checkpoint_class="weights", model_id="Auro", files=["weights.npz"],
        evidence=evidence, promotion_requested=True, signing_key=SIGNING_KEY)
    write_constitutional_manifest(tmp_path, checkpoint)
    with pytest.raises(ConstitutionalGateError, match="quarantined"):
        load_and_verify_constitutional_manifest(tmp_path, signing_key=SIGNING_KEY)
    loaded = load_and_verify_constitutional_manifest(tmp_path, signing_key=SIGNING_KEY, require_promoted=False)
    assert loaded["promotion_status"] == "quarantined"


def test_tampered_artifact_breaks_verification(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(root=tmp_path, checkpoint_id="auro-005",
        checkpoint_class="weights", model_id="Auro", files=["weights.npz"], evidence=complete_evidence())
    write_constitutional_manifest(tmp_path, checkpoint)
    (tmp_path / "weights.npz").write_bytes(b"tampered")
    with pytest.raises(ConstitutionalGateError):
        load_and_verify_constitutional_manifest(tmp_path, require_promoted=False)


def test_promotion_evidence_requires_valid_signature_and_artifacts(tmp_path: Path):
    path = signed_receipt(tmp_path)
    receipt = load_signed_promotion_evidence(path, SIGNING_KEY)
    flags = evidence_flags(receipt)
    assert flags["matched_benchmark"] is True
    assert flags["rollback_verified"] is True
    payload = json.loads(path.read_text()); payload["benchmark"]["passed"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(PromotionEvidenceError, match="signature mismatch"):
        load_signed_promotion_evidence(path, SIGNING_KEY)


def test_manifest_metadata_rewrite_is_detected(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(root=tmp_path, checkpoint_id="auro-006",
        checkpoint_class="weights", model_id="Auro", files=["weights.npz"], evidence=complete_evidence())
    path = write_constitutional_manifest(tmp_path, checkpoint)
    payload = json.loads(path.read_text()); payload["model_id"] = "rewritten"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConstitutionalGateError):
        load_and_verify_constitutional_manifest(tmp_path, require_promoted=False)
