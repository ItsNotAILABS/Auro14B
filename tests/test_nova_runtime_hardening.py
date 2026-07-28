import hashlib
import json
from pathlib import Path

import pytest

from auro_native_llm.nova import (
    NovaRuntimeState,
    SignalLensRelayConfig,
    SignalLensRelayPerception,
)


def test_approval_is_session_principal_action_and_single_use_bound(tmp_path):
    path = tmp_path / "nova.sqlite3"
    state = NovaRuntimeState(path)
    session_id = state.create_session("operator@example.com")
    action = {"tool": "filesystem.write", "path": "artifact.txt", "sha256": "a" * 64}
    approval = state.issue_approval(session_id, "operator@example.com", action)

    consumed = state.consume_approval(
        approval["approval_id"], session_id, "operator@example.com", action
    )
    assert consumed["authorized"] is True
    with pytest.raises(PermissionError, match="already consumed"):
        state.consume_approval(
            approval["approval_id"], session_id, "operator@example.com", action
        )
    state.close()

    reopened = NovaRuntimeState(path)
    assert reopened.session(session_id)["principal"] == "operator@example.com"
    with pytest.raises(PermissionError, match="already consumed"):
        reopened.consume_approval(
            approval["approval_id"], session_id, "operator@example.com", action
        )
    reopened.close()


def test_action_mismatch_and_replayed_event_are_rejected(tmp_path):
    state = NovaRuntimeState(tmp_path / "nova.sqlite3")
    session_id = state.create_session("operator")
    action = {"tool": "relay.read", "url": "https://example.com"}
    approval = state.issue_approval(session_id, "operator", action)
    with pytest.raises(PermissionError, match="action mismatch"):
        state.consume_approval(
            approval["approval_id"], session_id, "operator", {**action, "url": "https://other.example"}
        )

    first = state.record_receipt(session_id, "sensor.read", {"ok": True}, replay_key="event-1")
    assert len(first["receipt_sha256"]) == 64
    with pytest.raises(ValueError, match="replayed NOVA event"):
        state.record_receipt(session_id, "sensor.read", {"ok": True}, replay_key="event-1")
    state.close()


def test_signallens_perception_has_no_simulation_fallback_and_verifies_hash():
    missing = SignalLensRelayPerception(SignalLensRelayConfig("", ""))
    assert missing.health()["configured"] is False
    assert missing.health()["simulation_fallback"] is False
    with pytest.raises(RuntimeError, match="not configured"):
        missing.perceive("https://example.com", session_id="s", approval_receipt={"authorized": True, "action_sha256": "a" * 64})

    content = "verified SignalLens observation"
    response = {
        "text": content,
        "intelligence": {"citations": ["https://example.com/source"]},
        "receipt": {
            "request_id": "request-1",
            "receipt_sha256": "b" * 64,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        },
    }

    def transport(req, timeout):
        assert req.full_url == "https://relay.example/v1/read"
        assert req.headers["Authorization"] == "Bearer token"
        return json.dumps(response).encode()

    perception = SignalLensRelayPerception(
        SignalLensRelayConfig("https://relay.example", "token"), transport=transport
    )
    result = perception.perceive(
        "https://example.com/source",
        session_id="nova-session",
        approval_receipt={"authorized": True, "action_sha256": "c" * 64},
    )
    assert result["cross_repository_deployment_verified"] is True
    assert result["simulation"] is False
