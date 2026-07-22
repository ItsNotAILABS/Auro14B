import json
from pathlib import Path

import pytest

from auro_native_llm.substrate.checkpoint_constitution import (
    AIOPS_DOMAINS,
    PROTOCOLS,
    ConstitutionalGateError,
    build_constitutional_checkpoint,
    load_and_verify_constitutional_manifest,
    require_promotable,
    write_constitutional_manifest,
)


def complete_evidence():
    return {
        "resume_state_present": True,
        "matched_benchmark": True,
        "protected_capabilities_pass": True,
        "replay_or_forgetting_eval": True,
        "reversible_module_boundary": True,
        "tool_registry_receipt": True,
        "identity_state_present": True,
        "continuity_parent_known": True,
        "rollback_target": "safe-001",
        "rollback_verified": True,
    }


def test_ten_protocols_and_aiops_domains_are_load_bearing(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(
        root=tmp_path,
        checkpoint_id="auro-test-001",
        checkpoint_class="organism",
        model_id="Auro-HIM",
        files=["weights.npz"],
        parent_checkpoint_id="safe-001",
        evidence=complete_evidence(),
        promotion_requested=True,
    )
    assert {item.protocol for item in checkpoint.protocols} == set(PROTOCOLS)
    assert set(checkpoint.aiops_domains) == set(AIOPS_DOMAINS)
    assert checkpoint.promotion_status == "promoted"


def test_tampering_breaks_constitutional_verification(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(
        root=tmp_path,
        checkpoint_id="auro-test-002",
        checkpoint_class="weights",
        model_id="Auro",
        files=["weights.npz"],
        evidence=complete_evidence(),
    )
    write_constitutional_manifest(tmp_path, checkpoint)
    (tmp_path / "weights.npz").write_bytes(b"tampered")
    with pytest.raises(ConstitutionalGateError):
        load_and_verify_constitutional_manifest(tmp_path)


def test_unproven_optimization_is_quarantined(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    evidence = complete_evidence()
    evidence["matched_benchmark"] = False
    evidence["optimization_candidate"] = True
    checkpoint = build_constitutional_checkpoint(
        root=tmp_path,
        checkpoint_id="auro-test-003",
        checkpoint_class="weights",
        model_id="Auro",
        files=["weights.npz"],
        evidence=evidence,
        promotion_requested=True,
    )
    write_constitutional_manifest(tmp_path, checkpoint)
    loaded = load_and_verify_constitutional_manifest(tmp_path)
    assert loaded["promotion_status"] == "quarantined"
    assert "matched_optimization_experiment" in loaded["failed_protocols"]
    with pytest.raises(ConstitutionalGateError):
        require_promotable(loaded)


def test_modular_growth_and_economic_interfaces_are_protected(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    evidence = complete_evidence()
    evidence.update(
        {
            "architecture_changed": True,
            "reversible_module_boundary": False,
            "tool_capabilities_present": True,
            "tool_registry_receipt": False,
        }
    )
    checkpoint = build_constitutional_checkpoint(
        root=tmp_path,
        checkpoint_id="auro-test-004",
        checkpoint_class="capability",
        model_id="Auro",
        files=["weights.npz"],
        evidence=evidence,
        promotion_requested=True,
    )
    failed = {item.protocol for item in checkpoint.protocols if not item.passed}
    assert "structured_architecture_introduction" in failed
    assert "mcp_tool_capability_checkpoint" in failed
    assert checkpoint.promotion_status == "quarantined"


def test_manifest_seal_detects_metadata_rewrite(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"weights")
    checkpoint = build_constitutional_checkpoint(
        root=tmp_path,
        checkpoint_id="auro-test-005",
        checkpoint_class="weights",
        model_id="Auro",
        files=["weights.npz"],
        evidence=complete_evidence(),
    )
    path = write_constitutional_manifest(tmp_path, checkpoint)
    payload = json.loads(path.read_text())
    payload["model_id"] = "rewritten-identity"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConstitutionalGateError):
        load_and_verify_constitutional_manifest(tmp_path)
