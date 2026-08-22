from __future__ import annotations

from dataclasses import dataclass
import inspect
import time

import pytest

from auro_native_llm.production_fleet.capabilities import Capability, NativeCapabilities, _obj, capability_action
from auro_native_llm.production_fleet.organ_sdk import ApprovalReplayStore, AuroOrganSDK, build_server_approval
from auro_native_llm.production_fleet.server import production_security_status


@dataclass
class _Receipt:
    sequence: int = 1
    timestamp_ns: int = 1
    kind: str = "capability"
    subject: str = "test.mutate"
    ok: bool = True
    previous_hash: str = "GENESIS"
    payload_hash: str = "payload"
    receipt_hash: str = "receipt"
    metadata: dict | None = None


class _Ledger:
    def record(self, kind, subject, ok, payload, metadata=None):
        return _Receipt(kind=kind, subject=subject, ok=ok, metadata=dict(metadata or {}))


def _capabilities(sdk):
    instance = object.__new__(NativeCapabilities)
    instance.sdk = sdk
    instance._items = {
        "test.mutate": Capability(
            "test.mutate",
            "test mutation",
            "test",
            "tool",
            _obj({"value": {"type": "string"}}, ("value",)),
            mutating=True,
            approval_required=True,
        )
    }
    instance.ledger = _Ledger()
    instance._dispatch = lambda name, arguments: {"name": name, "value": arguments["value"]}
    return instance


def test_mutating_capabilities_no_longer_accept_caller_boolean():
    signature = inspect.signature(NativeCapabilities.call)
    assert "approved" not in signature.parameters
    assert "approval_grant" in signature.parameters


def test_signed_capability_grant_is_one_time(monkeypatch, tmp_path):
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "unit-test-secret")
    sdk = AuroOrganSDK(replay_store=ApprovalReplayStore(tmp_path / "replay"))
    capabilities = _capabilities(sdk)
    arguments = {"value": "alpha"}
    action = capability_action("test.mutate", arguments)
    grant = build_server_approval([action], "test-operator", "approval-1", key="unit-test-secret", nonce="nonce-1")

    denied = capabilities.call("test.mutate", arguments)
    assert denied["ok"] is False
    assert denied["denied"] is True

    first = capabilities.call("test.mutate", arguments, approval_grant=grant)
    assert first["ok"] is True
    assert first["approval_id"] == "approval-1"

    replay = capabilities.call("test.mutate", arguments, approval_grant=grant)
    assert replay["ok"] is False
    assert replay["denied"] is True


def test_tampered_expired_and_wrong_action_grants_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "unit-test-secret")
    sdk = AuroOrganSDK(replay_store=ApprovalReplayStore(tmp_path / "replay"))
    action = capability_action("test.mutate", {"value": "alpha"})
    now = int(time.time() * 1000)
    valid = build_server_approval([action], "operator", "approval-valid", now_ms=now, ttl_ms=10_000, key="unit-test-secret", nonce="nonce-valid")
    assert sdk.verify_server_approval(valid, [action], now_ms=now + 1)

    tampered = dict(valid)
    tampered["subject"] = "attacker"
    assert not sdk.verify_server_approval(tampered, [action], now_ms=now + 1)

    expired = build_server_approval([action], "operator", "approval-expired", now_ms=now - 20_000, ttl_ms=1_000, key="unit-test-secret", nonce="nonce-expired")
    assert not sdk.verify_server_approval(expired, [action], now_ms=now)

    wrong = capability_action("test.mutate", {"value": "beta"})
    assert not sdk.consume_server_approval(valid, wrong)


def test_replay_store_is_cross_instance_atomic(monkeypatch, tmp_path):
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "unit-test-secret")
    root = tmp_path / "replay"
    action = capability_action("test.mutate", {"value": "alpha"})
    grant = build_server_approval([action], "operator", "approval-atomic", key="unit-test-secret", nonce="nonce-atomic")
    sdk_a = AuroOrganSDK(replay_store=ApprovalReplayStore(root))
    sdk_b = AuroOrganSDK(replay_store=ApprovalReplayStore(root))

    assert sdk_a.consume_server_approval(grant, action) is True
    assert sdk_b.consume_server_approval(grant, action) is False


def test_non_positive_approval_ttl_is_rejected(monkeypatch):
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "unit-test-secret")
    action = capability_action("test.mutate", {"value": "alpha"})
    with pytest.raises(ValueError):
        build_server_approval([action], "operator", "approval-bad-ttl", ttl_ms=0)


def test_production_mode_fails_readiness_without_strong_distinct_secrets(monkeypatch):
    monkeypatch.setenv("AURO_ENV", "production")
    monkeypatch.delenv("AURO_API_TOKEN", raising=False)
    monkeypatch.delenv("AURO_EXECUTION_TOKEN", raising=False)
    monkeypatch.delenv("AURO_APPROVAL_HMAC_KEY", raising=False)
    status = production_security_status("0.0.0.0")
    assert status["mode"] == "production"
    assert status["ready"] is False
    assert not all(status["secret_checks"].values())


def test_production_mode_requires_distinct_secrets(monkeypatch):
    shared = "x" * 40
    monkeypatch.setenv("AURO_ENV", "production")
    monkeypatch.setenv("AURO_API_TOKEN", shared)
    monkeypatch.setenv("AURO_EXECUTION_TOKEN", shared)
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "y" * 40)
    assert production_security_status()["ready"] is False

    monkeypatch.setenv("AURO_API_TOKEN", "a" * 40)
    monkeypatch.setenv("AURO_EXECUTION_TOKEN", "b" * 40)
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "c" * 40)
    status = production_security_status()
    assert status["ready"] is True
    assert status["secrets_distinct"] is True
