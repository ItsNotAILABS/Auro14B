import hashlib
import json

import pytest

from auro_native_llm.model import (
    Auro2BCouncilRuntime,
    ModelExecutor,
    ModelIdentity,
    fluidize_report,
)
from auro_native_llm.production_fleet.council_service import (
    CONFIG_SCHEMA,
    CouncilService,
    CouncilUnavailable,
)


class FakeMesie:
    def analyze(self, text: str, model_id: str):
        payload = {
            "schema": "test.mesie.receipt.v1",
            "model_id": model_id,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "spectral_metrics": {
                "spectral_entropy": 0.5,
                "spectral_centroid": 0.25,
            },
            "backend": "test-mesie",
        }
        payload["receipt_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return payload


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _executor(model_id: str, parameter_target: int, calls: list[dict]):
    identity = ModelIdentity(
        model_id=model_id,
        parameter_target=parameter_target,
        checkpoint_id=f"checkpoint:{model_id}",
        checkpoint_sha256=_sha(f"checkpoint:{model_id}"),
        adapter_id=f"adapter:{model_id}" if model_id.startswith("Auro-500M-") else None,
        adapter_sha256=_sha(f"adapter:{model_id}") if model_id.startswith("Auro-500M-") else None,
        measured_parameters=parameter_target,
        provider="test",
    )

    def generate(messages, options):
        system = messages[0]["content"]
        user = messages[1]["content"]
        calls.append({"model_id": model_id, "system": system, "user": user})
        if "acting only as" in system:
            return {
                "text": json.dumps(
                    {
                        "answer": f"{model_id} completed its bounded role.",
                        "confidence": 0.84,
                        "evidence": [f"receipt:{model_id}"],
                    }
                )
            }
        if "Produce a bounded specialist report" in system:
            return {
                "text": json.dumps(
                    {
                        "analysis": f"{model_id} reviewed atomic reports.",
                        "draft": f"{model_id} recommends a verified staged plan.",
                        "recommendations": ["Validate before promotion"],
                        "evidence": [f"specialist:{model_id}"],
                        "confidence": 0.87,
                    }
                )
            }
        if "Review all three specialist reports" in system:
            return {
                "text": json.dumps(
                    {
                        "consensus": "Use the staged plan and preserve evidence boundaries.",
                        "confidence": 0.9,
                        "disagreements": [],
                        "evidence": [f"vote:{model_id}"],
                    }
                )
            }
        if "final parent synthesizer" in system:
            return {
                "text": json.dumps(
                    {
                        "answer": "Ship the bounded implementation only after its exact evidence gates pass.",
                        "key_points": ["Keep each checkpoint identity separate"],
                        "recommendations": ["Run the exact-checkpoint benchmark"],
                        "caveats": ["Architecture is not trained capability"],
                        "confidence": 0.91,
                        "citations": [],
                    }
                )
            }
        raise AssertionError(f"unexpected prompt for {model_id}: {system}")

    return ModelExecutor(identity, generate)


def _runtime(calls: list[dict], *, signed: bool = True):
    main = _executor("Auro-2B", 2_000_000_000, calls)
    specialists = [
        _executor("Auro-500M-SENSUS", 500_000_000, calls),
        _executor("Auro-500M-PRAXIS", 500_000_000, calls),
        _executor("Auro-500M-VERBUM", 500_000_000, calls),
    ]
    atomics = {
        "Auro-156K": _executor("Auro-156K", 156_000, calls),
        "Auro-250M": _executor("Auro-250M", 250_000_000, calls),
    }
    return Auro2BCouncilRuntime(
        main_2b=main,
        specialists=specialists,
        atomic_executors=atomics,
        mesie=FakeMesie(),
        signing_key="s" * 32 if signed else None,
        signer_id="test-signer",
    )


def test_council_runs_three_specialists_atomic_swarms_and_two_synthesis_rounds():
    calls = []
    runtime = _runtime(calls)
    secret_parent_context = "PRIVATE-PARENT-CONTEXT " * 800
    result = runtime.run_turn(
        "Research the evidence, design Python code, and write a clear deployment plan.",
        full_parent_context=secret_parent_context,
    )

    assert len(result.specialist_reports) == 3
    assert len(result.consensus_votes) == 3
    assert result.atomic_agent_count >= 9
    assert result.model_backed_atomic_count == result.atomic_agent_count
    assert len(result.mesie_receipts) == 1 + 3 + result.atomic_agent_count + 1
    assert result.estimated_dispatch_tokens < result.estimated_naive_broadcast_tokens
    assert result.estimated_text_reduction > 0.5
    assert result.evidence_class == "E4-signed-receipt"
    assert result.release_evidence_ready is True
    assert result.blockers == ()
    assert result.runtime_receipt["signature"]
    assert "architecture is not trained capability" in result.text.lower()

    atomic_calls = [item for item in calls if "acting only as" in item["system"]]
    assert atomic_calls
    assert all("PRIVATE-PARENT-CONTEXT" not in item["user"] for item in atomic_calls)


def test_missing_atomic_models_and_signer_are_visible_blockers_not_silent_fallbacks():
    calls = []
    main = _executor("Auro-2B", 2_000_000_000, calls)
    specialists = [
        _executor("Auro-500M-SENSUS", 500_000_000, calls),
        _executor("Auro-500M-PRAXIS", 500_000_000, calls),
        _executor("Auro-500M-VERBUM", 500_000_000, calls),
    ]
    runtime = Auro2BCouncilRuntime(
        main_2b=main,
        specialists=specialists,
        atomic_executors={},
        mesie=FakeMesie(),
    )
    result = runtime.run_turn("Write a safe plan.")

    assert result.model_backed_atomic_count == 0
    assert result.release_evidence_ready is False
    assert result.evidence_class in {"E2-execution-log", "E3-validated-output"}
    assert any("MESIE-only" in blocker for blocker in result.blockers)
    assert any("signing" in blocker for blocker in result.blockers)
    assert result.runtime_receipt["signature"] is None


def test_council_rejects_duplicate_or_missing_specialist_identity():
    calls = []
    main = _executor("Auro-2B", 2_000_000_000, calls)
    specialists = [
        _executor("Auro-500M-SENSUS", 500_000_000, calls),
        _executor("Auro-500M-SENSUS", 500_000_000, calls),
        _executor("Auro-500M-VERBUM", 500_000_000, calls),
    ]
    with pytest.raises(ValueError, match="requires exactly"):
        Auro2BCouncilRuntime(
            main_2b=main,
            specialists=specialists,
            mesie=FakeMesie(),
        )


def test_fluidizer_is_deterministic_and_does_not_add_unprovided_facts():
    report = {
        "answer": "The source contract is present.",
        "key_points": ["Deployment evidence is absent."],
        "caveats": ["Do not claim production."],
    }
    first = fluidize_report(report)
    second = fluidize_report(report)
    assert first == second
    assert "checkpoint" not in first.text.lower()
    assert "production" in first.text.lower()
    assert first.source_sha256
    assert first.output_sha256


def test_unconfigured_production_service_fails_closed():
    service = CouncilService()
    assert service.status()["configured"] is False
    with pytest.raises(CouncilUnavailable):
        service.respond("hello")


def test_endpoint_config_contract_builds_without_executing_network_calls(monkeypatch):
    monkeypatch.setenv("AURO_COUNCIL_RECEIPT_HMAC_KEY", "k" * 32)
    config = {
        "schema": CONFIG_SCHEMA,
        "main": {
            "model_id": "Auro-2B",
            "parameter_target": 2_000_000_000,
            "base_url": "http://127.0.0.1:8088/v1",
            "model": "auro-2b",
            "checkpoint_id": "main",
            "checkpoint_sha256": _sha("main"),
        },
        "specialists": [
            {
                "model_id": model_id,
                "parameter_target": 500_000_000,
                "base_url": "http://127.0.0.1:8088/v1",
                "model": model_id.lower(),
                "checkpoint_id": "base-500m",
                "checkpoint_sha256": _sha("base-500m"),
                "adapter_id": model_id,
                "adapter_sha256": _sha(model_id),
            }
            for model_id in (
                "Auro-500M-SENSUS",
                "Auro-500M-PRAXIS",
                "Auro-500M-VERBUM",
            )
        ],
        "atomic": [
            {
                "model_id": model_id,
                "parameter_target": target,
                "base_url": "http://127.0.0.1:8088/v1",
                "model": model_id.lower(),
                "checkpoint_id": model_id,
                "checkpoint_sha256": _sha(model_id),
            }
            for model_id, target in (
                ("Auro-156K", 156_000),
                ("Auro-250M", 250_000_000),
            )
        ],
    }
    service = CouncilService(
        config=config,
        runtime=CouncilService._build_runtime(config),
        source="test",
    )
    status = service.status()
    assert status["configured"] is True
    assert status["runtime"]["main"]["model_id"] == "Auro-2B"
    assert len(status["runtime"]["specialists"]) == 3
    assert set(status["runtime"]["atomic_executors"]) == {
        "Auro-156K",
        "Auro-250M",
    }
