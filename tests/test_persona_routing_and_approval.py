from __future__ import annotations

from auro_native_llm.production_fleet.model_orchestrator import ModelLane, MultiModelOrchestrator
from auro_native_llm.production_fleet.organ_sdk import AuroOrganSDK, build_server_approval


def _generator(name):
    def generate(messages, options):
        return {"text": '{"answer":"ok","reasoning_summary":[],"confidence":1,"actions":[]}', "raw_model": name}
    return generate


def test_persona_preferred_model_changes_real_route_order():
    lanes = [
        ModelLane("lane-2b", "Auro-2B", "general", "local", _generator("2b"), capabilities=("general", "research"), priority=0, local=True),
        ModelLane("lane-4b", "Auro-4B", "general", "local", _generator("4b"), capabilities=("general", "research"), priority=10, local=True),
    ]
    router = MultiModelOrchestrator(lanes)
    assert router.route([{"role": "user", "content": "research evidence"}]).selected_lane == "lane-2b"
    router.set_preferred_models(("Auro-4B", "Auro-2B"))
    decision = router.route([{"role": "user", "content": "research evidence"}])
    assert decision.selected_lane == "lane-4b"
    assert "persona_preferred=True" in decision.reason


def test_server_approval_binds_full_action_set_and_membership(monkeypatch):
    key = "approval-test-key"
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", key)
    actions = [
        {"tool": "capsula", "arguments": {"operation": "run", "parameters": {"session_id": "s1"}}},
        {"tool": "matdaemon", "arguments": {"name": "matdaemon_matmul", "arguments": {"a": [[1]], "b": [[2]]}}},
    ]
    grant = build_server_approval(actions, "user-1", "approval-1", now_ms=1000, ttl_ms=5000, key=key)
    sdk = AuroOrganSDK()
    assert sdk.verify_server_approval(grant, actions, now_ms=2000)
    assert not sdk.verify_server_approval(grant, list(reversed(actions)), now_ms=2000)
    assert not sdk.verify_server_approval({**grant, "actions": actions[:1]}, actions[:1], now_ms=2000)
    assert not sdk.verify_server_approval(grant, actions, now_ms=7000)


def test_individual_execution_must_be_member_of_signed_set(monkeypatch):
    key = "approval-test-key"
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", key)
    approved = {"tool": "capsula", "arguments": {"operation": "run", "parameters": {"session_id": "s1"}}}
    denied = {"tool": "capsula", "arguments": {"operation": "run", "parameters": {"session_id": "s2"}}}
    grant = build_server_approval([approved], "user-1", "approval-2", key=key)
    sdk = AuroOrganSDK()
    sdk.capsula.run = lambda session_id: {"session_id": session_id, "ran": True}
    assert sdk.execute(approved, approval_grant=grant)["ok"] is True
    try:
        sdk.execute(denied, approval_grant=grant)
    except PermissionError:
        pass
    else:
        raise AssertionError("unapproved action executed")
