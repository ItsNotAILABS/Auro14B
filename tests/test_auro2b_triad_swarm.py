import hashlib
import json

from auro_native_llm.model.atomic_family import AURO_500M_TRIAD
from auro_native_llm.model.triad_swarm import Auro2BTriadSwarm, ModelExecutor, ModelIdentity


class FakeMesie:
    def __init__(self):
        self.calls = []

    def analyze(self, text, model_id):
        self.calls.append((model_id, text))
        payload = {
            "model_id": model_id,
            "backend": "mesie.fixture",
            "spectral_metrics": {"spectral_entropy": 0.5, "spectral_centroid": 0.25},
            "embedding_sha256": hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest(),
        }
        payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload


def fake_generator(label):
    def generate(messages, options):
        system = messages[0]["content"]
        if "bounded AURO atomic expert" in system:
            value = {"answer": f"{label} atomic conclusion", "confidence": 0.82, "evidence": ["fixture:atomic"]}
        elif "one member of a three-model AURO consensus" in system:
            value = {"consensus": "Use the evidence-backed implementation and preserve claim boundaries", "confidence": 0.88, "disagreements": [], "evidence": ["fixture:consensus"]}
        elif "Auro-2B, parent" in system:
            value = {
                "answer": "Ship the wired triad only after exact checkpoint evaluation",
                "reasoning_summary": ["The three specialists agreed", "MESIE receipts were produced at every stage"],
                "key_points": ["Use bounded capsules", "Keep every checkpoint identity separate"],
                "caveats": ["Fixture outputs are not model-quality evidence"],
                "next_steps": ["Run exact checkpoints"],
                "citations": ["https://example.test/evidence"],
                "confidence": 0.9,
            }
        else:
            value = {
                "analysis": f"{label} specialist analysis",
                "draft": f"{label} specialist draft",
                "recommendations": [f"apply {label} recommendation"],
                "evidence": [f"fixture:{label}"],
                "confidence": 0.86,
            }
        return {"text": json.dumps(value), "usage": {"completion_tokens": 32}}

    return generate


def identity(model_id, target, *, adapter=False, verified=True):
    return ModelIdentity(
        model_id=model_id,
        parameter_target=target,
        checkpoint_id=("Auro-500M" if adapter else model_id) if verified else None,
        checkpoint_sha256=(model_id[0].lower() * 64) if verified else None,
        adapter_id=(model_id + "-adapter") if adapter and verified else None,
        adapter_sha256=(model_id[-1].lower() * 64) if adapter and verified else None,
        measured_parameters=target,
        provider="fixture",
    )


def build_runtime(*, verified=True, with_atomic_models=True):
    main = ModelExecutor(identity("Auro-2B", 2_000_000_000, verified=verified), fake_generator("main"))
    specialists = [
        ModelExecutor(identity(item.variant_id, 500_000_000, adapter=True, verified=verified), fake_generator(item.variant_id))
        for item in AURO_500M_TRIAD
    ]
    atomics = {}
    if with_atomic_models:
        atomics[("Auro-156K", "*")] = ModelExecutor(identity("Auro-156K", 156_000, verified=verified), fake_generator("156K"))
        atomics[("Auro-250M", "*")] = ModelExecutor(identity("Auro-250M", 250_000_000, verified=verified), fake_generator("250M"))
    mesie = FakeMesie()
    runtime = Auro2BTriadSwarm(main_2b=main, specialists=specialists, atomic_executors=atomics, mesie=mesie)
    return runtime, mesie


def test_full_triad_runs_atomic_swarms_two_consensus_levels_and_fluidizer():
    runtime, mesie = build_runtime()
    message = "Research the evidence, debug the code, plan execution, remember context, and write a creative conversational explanation."
    result = runtime.run_turn(message, full_parent_context=(message + " prior context ") * 250)
    assert len(result.specialist_reports) == 3
    assert len(result.consensus_votes) == 3
    assert result.atomic_agent_count == 12
    assert result.model_backed_atomic_count == 12
    assert len(result.mesie_receipts) == 17  # ingress + 3 specialists + 12 atomics + egress
    assert len(mesie.calls) == 17
    assert result.promotion_ready is True
    assert result.blockers == ()
    assert result.estimated_text_reduction > 0.5
    assert len(result.runtime_receipt_sha256) == 64
    assert "Ship the wired triad" in result.text
    assert result.structured_answer["model_identity"]["parameter_target"] == 2_000_000_000
    loaded_targets = [report.model_identity["parameter_target"] for report in result.specialist_reports]
    assert loaded_targets == [500_000_000, 500_000_000, 500_000_000]


def test_missing_atomic_checkpoints_and_specialization_evidence_quarantine_result():
    runtime, _ = build_runtime(verified=False, with_atomic_models=False)
    result = runtime.run_turn("Review evidence and code safely.")
    assert result.promotion_ready is False
    assert result.model_backed_atomic_count == 0
    assert any("Auro-2B exact checkpoint" in item for item in result.blockers)
    assert any("lacks distinct checkpoint" in item for item in result.blockers)
    assert any("atomic agents used MESIE compute" in item for item in result.blockers)


def test_runtime_rejects_wrong_triad_identity_or_count():
    main = ModelExecutor(identity("Auro-2B", 2_000_000_000), fake_generator("main"))
    wrong = [ModelExecutor(identity("Auro-500M-SENSUS", 500_000_000, adapter=True), fake_generator("x"))] * 3
    try:
        Auro2BTriadSwarm(main_2b=main, specialists=wrong, mesie=FakeMesie())
    except ValueError as exc:
        assert "triad identities" in str(exc)
    else:
        raise AssertionError("duplicate triad identity was accepted")
