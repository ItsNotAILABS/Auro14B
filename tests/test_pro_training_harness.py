import json
from pathlib import Path

from auro_native_llm.evaluation.pro_harness import (
    HarnessCase,
    ProEvaluationHarness,
    default_cases,
    evaluation_suites,
    score_output,
)
from auro_native_llm.training.pro_cycle import (
    CurriculumRecord,
    build_curriculum_manifest,
    build_training_job,
    deduplicate,
    write_cycle_bundle,
)


def test_score_output_enforces_creative_constraints():
    case = HarnessCase(
        "creative",
        "creativity",
        "Create options",
        ("has_multiple_options",),
        forbidden_terms=("forbidden",),
        min_words=4,
    )
    result = score_output(case, "- Concept one is vivid\n- Concept two is precise")
    assert result.passed
    assert result.score >= 0.85
    assert len(result.output_sha256) == 64


def test_harness_emits_receipt_and_required_suite_adapter():
    def generator(prompt, metadata):
        suite = metadata["suite"]
        if suite == "coding":
            return "Zero division must be guarded.\n```python\ndef safe(x, y):\n    return None if y == 0 else x / y\n```"
        if suite == "creativity":
            return "- Concept one explores memory through light and structure.\n- Concept two uses spatial archives and motion.\n- Concept three makes the laboratory feel alive through responsive geometry."
        if suite == "multitask":
            return "Risk: migration failure.\n- Mitigation one: staged rollout.\n- Mitigation two: rollback.\nStatus: migration is controlled and observable."
        if suite == "tool_use":
            return "Trust the current tool evidence, but state the assumption and uncertainty when it cannot be verified."
        if suite == "safety":
            return "I cannot verify completion because no checkpoint or logs were provided; that is an unsupported assumption."
        if suite == "reasoning":
            return "1. Inventory dependencies.\n2. Migrate a canary.\n3. Verify receipts.\n4. Rollback if gates fail."
        return "Use preserved context and history, and never invent missing conversation turns."

    report = ProEvaluationHarness(default_cases()).run(generator, {"model_id": "test"})
    assert len(report["receipt_sha256"]) == 64
    suites = evaluation_suites(report)
    assert "relay_tool_use" in suites
    assert "checkpoint_integrity" in suites


def test_curriculum_dedup_and_job_gate(tmp_path):
    records = [
        CurriculumRecord("1", "multitask", "p", "a", "repo", "a" * 64),
        CurriculumRecord("2", "multitask", "p", "a", "repo", "a" * 64),
        CurriculumRecord("3", "creativity", "p2", "a2", "repo", "b" * 64),
    ]
    unique, duplicates = deduplicate(records)
    assert len(unique) == 2
    assert duplicates == 1

    curriculum = build_curriculum_manifest(unique, ("multitask", "creativity"))
    assert curriculum["ready"]
    job = build_training_job(
        curriculum,
        model_id="Auro-2B",
        resume_checkpoint="checkpoints/base",
        output_checkpoint="checkpoints/candidate",
    )
    assert job["runnable"]
    assert job["promotion_policy"]["human_authorization_required"]

    paths = write_cycle_bundle(tmp_path, curriculum, job)
    receipt = json.loads(Path(paths["receipt"]).read_text())
    assert receipt["training_completed"] is False
    assert len(receipt["receipt_sha256"]) == 64
