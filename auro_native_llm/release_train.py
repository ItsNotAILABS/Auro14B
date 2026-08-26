"""Evidence-driven AURO release train from 156K through 2B.

The planner translates checkpoint inventory state into explicit next actions. It
does not execute training, promote checkpoints, or treat architecture targets as
weights. Every job remains unapproved until an operator supplies the required
corpus, tokenizer, compute, checkpoint, evaluation, and promotion evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "auro.sub2b.release-train.v1"
TRAINING_JOB_SCHEMA = "auro.family-training-job.v1"
RELEASE_SEQUENCE = (
    "Auro-156K",
    "Auro-250M",
    "Auro-500M",
    "Auro-500M-SENSUS",
    "Auro-500M-PRAXIS",
    "Auro-500M-VERBUM",
    "Auro-2B",
)
BASE_LANES = ("Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B")
TRIAD_LANES = (
    "Auro-500M-SENSUS",
    "Auro-500M-PRAXIS",
    "Auro-500M-VERBUM",
)
PARAMETER_TARGETS = {
    "Auro-156K": 156_000,
    "Auro-250M": 250_000_000,
    "Auro-500M": 500_000_000,
    "Auro-500M-SENSUS": 500_000_000,
    "Auro-500M-PRAXIS": 500_000_000,
    "Auro-500M-VERBUM": 500_000_000,
    "Auro-2B": 2_000_000_000,
}
DEPLOY_PROFILES = {
    "Auro-156K": ("wasm", "embedded", "high-multiplicity-swarm"),
    "Auro-250M": ("phone", "browser-wasm", "cpu", "embedded-expert"),
    "Auro-500M": ("phone-high-memory", "laptop", "edge-gpu", "embedded-expert"),
    "Auro-500M-SENSUS": ("embedded-specialist", "evidence-and-perception"),
    "Auro-500M-PRAXIS": ("embedded-specialist", "code-and-execution"),
    "Auro-500M-VERBUM": ("embedded-specialist", "language-and-expression"),
    "Auro-2B": ("laptop", "private-edge-server", "atomic-swarm-parent"),
}
REQUIRED_EVIDENCE = (
    "tokenizer_audit",
    "corpus_manifest",
    "training_report",
    "training_execution_receipt",
    "checkpoint_manifest",
    "official_benchmarks",
    "coding_execution",
    "governed_execution",
    "api_chat_smoke",
    "browser_chat_smoke",
    "clean_install",
    "model_card",
    "promotion_authorization",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class ReleaseLane:
    model_id: str
    parameter_target: int
    kind: str
    base_model_id: str | None
    prerequisites: tuple[str, ...]
    deploy_profiles: tuple[str, ...]
    trainer: str | None
    mode: str
    tokenizer_policy: str
    checkpoint_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LANES = {
    "Auro-156K": ReleaseLane(
        "Auro-156K",
        PARAMETER_TARGETS["Auro-156K"],
        "atomic-base",
        None,
        (),
        DEPLOY_PROFILES["Auro-156K"],
        "python -m auro_native_llm.model.train",
        "full",
        "versioned-family-tokenizer-with-explicit-custody",
        "exact-checkpoint-evidence-required",
    ),
    "Auro-250M": ReleaseLane(
        "Auro-250M",
        PARAMETER_TARGETS["Auro-250M"],
        "atomic-base",
        None,
        (),
        DEPLOY_PROFILES["Auro-250M"],
        "python -m auro_native_llm.model.train",
        "full",
        "shared-family-tokenizer-or-explicit-compatible-tokenizer",
        "exact-mobile-and-browser-evaluation-required",
    ),
    "Auro-500M": ReleaseLane(
        "Auro-500M",
        PARAMETER_TARGETS["Auro-500M"],
        "atomic-base",
        None,
        (),
        DEPLOY_PROFILES["Auro-500M"],
        "python -m auro_native_llm.model.train",
        "full",
        "shared-family-tokenizer-or-explicit-compatible-tokenizer",
        "base-checkpoint-required-before-specialist-adapters",
    ),
    "Auro-500M-SENSUS": ReleaseLane(
        "Auro-500M-SENSUS",
        PARAMETER_TARGETS["Auro-500M-SENSUS"],
        "specialist-adapter-or-checkpoint",
        "Auro-500M",
        ("Auro-500M",),
        DEPLOY_PROFILES["Auro-500M-SENSUS"],
        None,
        "adapter-or-distinct-checkpoint",
        "must-preserve-base-tokenizer-ids",
        "distinct-checkpoint-or-adapter-hash-required",
    ),
    "Auro-500M-PRAXIS": ReleaseLane(
        "Auro-500M-PRAXIS",
        PARAMETER_TARGETS["Auro-500M-PRAXIS"],
        "specialist-adapter-or-checkpoint",
        "Auro-500M",
        ("Auro-500M",),
        DEPLOY_PROFILES["Auro-500M-PRAXIS"],
        None,
        "adapter-or-distinct-checkpoint",
        "must-preserve-base-tokenizer-ids",
        "distinct-checkpoint-or-adapter-hash-required",
    ),
    "Auro-500M-VERBUM": ReleaseLane(
        "Auro-500M-VERBUM",
        PARAMETER_TARGETS["Auro-500M-VERBUM"],
        "specialist-adapter-or-checkpoint",
        "Auro-500M",
        ("Auro-500M",),
        DEPLOY_PROFILES["Auro-500M-VERBUM"],
        None,
        "adapter-or-distinct-checkpoint",
        "must-preserve-base-tokenizer-ids",
        "distinct-checkpoint-or-adapter-hash-required",
    ),
    "Auro-2B": ReleaseLane(
        "Auro-2B",
        PARAMETER_TARGETS["Auro-2B"],
        "micro-parent",
        None,
        ("Auro-156K", "Auro-250M", "Auro-500M", *TRIAD_LANES),
        DEPLOY_PROFILES["Auro-2B"],
        "python -m auro_native_llm.model.train",
        "full",
        "shared-family-tokenizer-with-custody-and-compatibility-report",
        "parent-and-composed-council-evaluation-required",
    ),
}


@dataclass(frozen=True)
class InputArtifact:
    name: str
    path: str | None
    present: bool
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_input(name: str, raw_path: str | Path | None) -> InputArtifact:
    if raw_path is None or not str(raw_path).strip():
        return InputArtifact(name, None, False, None)
    path = Path(raw_path).expanduser().resolve()
    return InputArtifact(name, str(path), path.is_file(), sha256_file(path) if path.is_file() else None)


def _summary_for(inventory: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    table = inventory.get("triad") if model_id in TRIAD_LANES else inventory.get("through_2b")
    if isinstance(table, Mapping):
        value = table.get(model_id)
        if isinstance(value, Mapping):
            return value
    return {
        "model_id": model_id,
        "candidate_count": 0,
        "artifact_present": False,
        "integrity_verified": False,
        "training_provenance_verified": False,
        "evaluation_verified": False,
        "promotion_ready": False,
        "best_candidate": None,
        "blockers": ["no checkpoint candidate"],
    }


def _state(summary: Mapping[str, Any]) -> str:
    if summary.get("promotion_ready"):
        return "promoted"
    if summary.get("evaluation_verified"):
        return "evaluated-awaiting-promotion"
    if summary.get("training_provenance_verified"):
        return "trained-awaiting-evaluation"
    if summary.get("integrity_verified"):
        return "integrity-verified-awaiting-training-provenance"
    if summary.get("artifact_present"):
        return "artifact-present-integrity-incomplete"
    return "missing"


def _next_action(summary: Mapping[str, Any], lane: ReleaseLane) -> str:
    state = _state(summary)
    if state == "promoted":
        return "retain-and-regression-test"
    if state == "evaluated-awaiting-promotion":
        return "run-human-authorized-checkpoint-promotion-gate"
    if state == "trained-awaiting-evaluation":
        return "run-exact-checkpoint-evaluation-and-product-smokes"
    if state == "integrity-verified-awaiting-training-provenance":
        return "recover-or-regenerate-training-provenance-pinned-to-checkpoint"
    if state == "artifact-present-integrity-incomplete":
        return "repair-checkpoint-custody-hashes-tokenizer-geometry-and-identity"
    if lane.kind == "specialist-adapter-or-checkpoint":
        return "implement-and-run-versioned-specialist-adapter-training"
    return "run-bounded-family-training-job"


def _training_command(lane: ReleaseLane, corpus: InputArtifact, output_root: str) -> list[str] | None:
    if lane.trainer is None:
        return None
    return [
        "python",
        "-m",
        "auro_native_llm.model.train",
        "--model",
        lane.model_id,
        "--mode",
        lane.mode,
        "--steps",
        "REPLACE_WITH_APPROVED_STEPS",
        "--batch-size",
        "REPLACE_WITH_APPROVED_BATCH",
        "--seq-len",
        "REPLACE_WITH_APPROVED_SEQUENCE_LENGTH",
        "--vocab-size",
        "REPLACE_WITH_TOKENIZER_VOCAB_SIZE",
        "--output-dir",
        output_root,
        *(["--corpus-root", str(Path(corpus.path).parent)] if corpus.present and corpus.path else []),
    ]


def build_release_train(
    inventory: Mapping[str, Any],
    *,
    corpus_manifest: str | Path | None = None,
    tokenizer_manifest: str | Path | None = None,
    output_root: str = "checkpoints/auro_release_candidates",
) -> dict[str, Any]:
    corpus = inspect_input("corpus_manifest", corpus_manifest)
    tokenizer = inspect_input("tokenizer_manifest", tokenizer_manifest)
    inputs = {item.name: item.to_dict() for item in (corpus, tokenizer)}

    lane_reports: list[dict[str, Any]] = []
    promoted: set[str] = {
        model_id
        for model_id in RELEASE_SEQUENCE
        if _summary_for(inventory, model_id).get("promotion_ready")
    }
    for model_id in RELEASE_SEQUENCE:
        lane = LANES[model_id]
        summary = dict(_summary_for(inventory, model_id))
        missing_prerequisites = [item for item in lane.prerequisites if item not in promoted]
        execution_blockers = list(summary.get("blockers") or [])
        if not corpus.present and not summary.get("promotion_ready"):
            execution_blockers.append("corpus manifest is not supplied or readable")
        if not tokenizer.present and not summary.get("promotion_ready"):
            execution_blockers.append("tokenizer manifest is not supplied or readable")
        if missing_prerequisites:
            execution_blockers.append(
                "unpromoted prerequisites: " + ", ".join(missing_prerequisites)
            )
        if lane.kind == "specialist-adapter-or-checkpoint" and lane.trainer is None:
            execution_blockers.append(
                "no specialist adapter trainer is currently registered in the allowlisted training queue"
            )
        if lane.trainer and not summary.get("promotion_ready"):
            execution_blockers.append(
                "the current generic family trainer starts a new run and does not yet prove resume or distillation lineage"
            )

        command = _training_command(lane, corpus, output_root)
        job = {
            "schema": TRAINING_JOB_SCHEMA,
            "model_id": lane.model_id,
            "parameter_target": lane.parameter_target,
            "kind": lane.kind,
            "base_model_id": lane.base_model_id,
            "mode": lane.mode,
            "entrypoint": lane.trainer,
            "command": command,
            "output_root": output_root,
            "corpus_manifest_sha256": corpus.sha256,
            "tokenizer_manifest_sha256": tokenizer.sha256,
            "approved": False,
            "operator_authorization_required": True,
            "execution_ready": bool(
                command
                and corpus.present
                and tokenizer.present
                and not missing_prerequisites
                and not summary.get("promotion_ready")
                and lane.kind != "specialist-adapter-or-checkpoint"
            ),
            "required_outputs": list(REQUIRED_EVIDENCE),
            "claim_boundary": (
                "a generated command or completed process is not a promoted checkpoint; "
                "the exact output must pass checkpoint inventory and the promotion gate"
            ),
        }
        job["job_sha256"] = digest(job)
        lane_reports.append(
            {
                "lane": lane.to_dict(),
                "inventory": summary,
                "state": _state(summary),
                "next_action": _next_action(summary, lane),
                "missing_prerequisites": missing_prerequisites,
                "execution_blockers": sorted(set(execution_blockers)),
                "training_job": job,
            }
        )

    group_ready = all(
        _summary_for(inventory, model_id).get("promotion_ready")
        for model_id in RELEASE_SEQUENCE
    )
    phases = [
        {
            "phase": 0,
            "name": "inventory-and-custody",
            "gate": "all local candidates audited without trusting names",
        },
        {
            "phase": 1,
            "name": "corpus-and-tokenizer-custody",
            "gate": "source, license, deduplication, tokenizer IDs and hashes are fixed",
        },
        {
            "phase": 2,
            "name": "base-lane-training",
            "gate": "156K, 250M, 500M and 2B exact outputs have signed training provenance",
        },
        {
            "phase": 3,
            "name": "specialist-adaptation",
            "gate": "SENSUS, PRAXIS and VERBUM each have distinct checkpoint or adapter evidence",
        },
        {
            "phase": 4,
            "name": "exact-evaluation",
            "gate": "model-only, council, coding, conversation, mobile and product smoke evidence is pinned",
        },
        {
            "phase": 5,
            "name": "human-authorized-promotion",
            "gate": "every lane passes the constitutional promotion gate with rollback evidence",
        },
        {
            "phase": 6,
            "name": "ship-through-2b",
            "gate": "all seven release identities are promoted and packaged with checksums and model cards",
        },
    ]
    plan = {
        "schema": PLAN_SCHEMA,
        "version": "1.0.0",
        "release_group": "AURO through 2B",
        "release_sequence": list(RELEASE_SEQUENCE),
        "inputs": inputs,
        "inventory_schema": inventory.get("schema"),
        "inventory_root": inventory.get("root"),
        "inventory_claim_boundary": inventory.get("claim_boundary"),
        "lanes": lane_reports,
        "phases": phases,
        "release_ready": bool(group_ready),
        "training_executed": False,
        "checkpoints_created": False,
        "checkpoints_promoted": bool(group_ready),
        "known_platform_gaps": [
            "generic AURO family trainer has no resume/distillation argument",
            "specialist adapter trainer is not registered",
            "GitHub source cannot prove local private checkpoint presence",
            "physical WebGPU execution requires browser worker receipts",
            "promotion requires operator-held signing keys and external release evidence",
        ],
        "operator_commands": {
            "inventory": [
                "python",
                "scripts/inventory_auro_checkpoints.py",
                "--root",
                str(inventory.get("root") or "checkpoints/auro_minds"),
                "--output",
                "artifacts/sub2b-release/inventory.json",
            ],
            "promotion_gate": [
                "python",
                "scripts/checkpoint_promotion_gate.py",
                "CHECKPOINT_DIRECTORY",
                "RELEASE_EVIDENCE_JSON",
                "--output",
                "PROMOTION_RESULT_JSON",
            ],
        },
        "claim_boundary": (
            "this release train is an evidence-driven operator plan. It does not "
            "train, evaluate, promote, package, or publish a model by itself"
        ),
    }
    plan["plan_sha256"] = digest(plan)
    return plan
