#!/usr/bin/env python3
"""Deterministic production-hardening verifier with no third-party dependencies."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import time

from auro_native_llm.brain.feline_neuromorphic import FelineNeuromorphicEngine
from auro_native_llm.brain.neuromorphic_state import NeuromorphicStateStore
from auro_native_llm.brain.timing_plasticity import TimingPlasticityController
from auro_native_llm.production_fleet.capabilities import NativeCapabilities, capability_action
from auro_native_llm.production_fleet.organ_sdk import ApprovalReplayStore, AuroOrganSDK, build_server_approval


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_authority_api() -> dict:
    path = Path("auro_native_llm/production_fleet/capabilities.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "call"]
    require(bool(calls), "NativeCapabilities.call not found")
    args = [arg.arg for arg in calls[0].args.args + calls[0].args.kwonlyargs]
    require("approved" not in args, "caller-controlled approved boolean is still present")
    require("approval_grant" in args, "approval_grant boundary missing")
    return {"caller_boolean_removed": True, "signed_grant_parameter": True}


def verify_replay() -> dict:
    with tempfile.TemporaryDirectory(prefix="auro-approval-") as temp:
        os.environ["AURO_APPROVAL_HMAC_KEY"] = "verification-secret"
        action = capability_action("wallet.transfer_paper", {"source": "a", "destination": "b", "amount": "1"})
        now = int(time.time() * 1000)
        grant = build_server_approval([action], "verifier", "verify-approval", now_ms=now, ttl_ms=30_000, key="verification-secret", nonce="verify-nonce")
        sdk = AuroOrganSDK(replay_store=ApprovalReplayStore(temp))
        require(sdk.verify_server_approval(grant, [action], now_ms=now + 1), "valid signed grant did not verify")
        require(sdk.consume_server_approval(grant, action), "first approval consumption failed")
        require(not sdk.consume_server_approval(grant, action), "approval replay was accepted")
        tampered = dict(grant); tampered["subject"] = "tampered"
        require(not sdk.verify_server_approval(tampered, [action], now_ms=now + 1), "tampered approval verified")
        return {"signed_grant_verified": True, "one_time_consumption": True, "tamper_denied": True}


def verify_persistence() -> dict:
    with tempfile.TemporaryDirectory(prefix="auro-neuro-") as temp:
        path = Path(temp) / "brain.neuromorphic.json"
        regions = ("V1", "V2V3", "SC", "THL_L", "LC")
        engine = FelineNeuromorphicEngine(regions)
        timing = TimingPlasticityController(regions)
        engine.cycle({"V1": 1.0, "SC": 0.8}, salience=0.8, novelty=0.9)
        timing.last_spike_cycle["V1"] = 3
        store = NeuromorphicStateStore(path)
        receipt = store.save(engine, timing)
        require(receipt["durable_atomic_write"], "durable atomic write not reported")

        restored = FelineNeuromorphicEngine(regions)
        restored_timing = TimingPlasticityController(regions)
        require(store.load(restored, restored_timing), "valid state did not restore")
        require(restored.cycle_number == engine.cycle_number, "cycle state did not restore")
        require(restored_timing.last_spike_cycle["V1"] == 3, "timing plasticity did not restore")

        body = json.loads(path.read_text(encoding="utf-8"))
        body["cycle"] = body["cycle"] + 1
        path.write_text(json.dumps(body), encoding="utf-8")
        target = FelineNeuromorphicEngine(regions)
        target_timing = TimingPlasticityController(regions)
        require(not store.load(target, target_timing), "tampered state was accepted")
        require(target.cycle_number == 0, "tampered load partially mutated live state")
        require(bool(store.status()["quarantined_path"]), "tampered state was not quarantined")
        return {"round_trip": True, "timing_restored": True, "tamper_quarantined": True, "transactional_load": True}


def main() -> int:
    report = {
        "schema": "auro.production-hardening-verification.v1",
        "authority": verify_authority_api(),
        "replay": verify_replay(),
        "neuromorphic_persistence": verify_persistence(),
        "checkpoint_quality_promoted": False,
        "physical_energy_claim": False,
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
