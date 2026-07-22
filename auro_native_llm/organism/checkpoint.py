"""Full AuroMind checkpoint with constitutional identity continuity."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from auro_native_llm.model.checkpoint import load_checkpoint as load_lm_checkpoint
from auro_native_llm.model.checkpoint import save_checkpoint as save_lm_checkpoint
from auro_native_llm.organism.mind import AuroMind
from auro_native_llm.organism.self_train import Experience
from auro_native_llm.substrate import (
    build_constitutional_checkpoint,
    load_and_verify_constitutional_manifest,
    write_constitutional_manifest,
)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def save_mind(mind: AuroMind, directory: str | Path) -> Dict[str, Any]:
    """Persist a complete mind organism and seal continuity evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lm_meta = save_lm_checkpoint(mind.language, directory / "language")

    if mind.organs.memory is not None:
        mind.organs.memory.save(directory / "memory.json")

    trainer = mind.organs.trainer
    trainer_payload: Dict[str, Any] = {"stats": {}, "experiences": []}
    if trainer is not None:
        trainer_payload["stats"] = trainer.stats()
        trainer_payload["lr"] = trainer.lr
        trainer_payload["messy_mix"] = trainer.messy_mix
        trainer_payload["seq_len"] = trainer.seq_len
        trainer_payload["batch_size"] = trainer.batch_size
        trainer_payload["doctrine_seeds"] = list(trainer._doctrine_seeds)[:50]
        trainer_payload["experiences"] = [
            experience.to_dict() for experience in list(trainer.buffer)[-500:]
        ]
        trainer_payload["loss_history"] = list(trainer.loss_history)[-200:]
        trainer_payload["total_train_steps"] = trainer.total_train_steps
        trainer_payload["total_absorbs"] = trainer.total_absorbs
    (directory / "trainer.json").write_text(
        json.dumps(trainer_payload, indent=2), encoding="utf-8"
    )

    checkpoint_id = os.environ.get(
        "AURO_ORGANISM_CHECKPOINT_ID",
        f"{mind.model_id}-organism-{mind.act_count}-{int(time.time())}",
    )
    parent_id = os.environ.get("AURO_PARENT_ORGANISM_CHECKPOINT_ID") or None
    safe_id = os.environ.get("AURO_SAFE_ORGANISM_CHECKPOINT_ID") or parent_id
    meta = {
        "schema": "auro.mind.checkpoint.v2",
        "model_id": mind.model_id,
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": parent_id,
        "parameter_target": mind.config.parameter_target,
        "tier": mind.config.tier,
        "num_params_live": mind.language.num_params,
        "act_count": mind.act_count,
        "born_at": mind.born_at,
        "saved_at_unix": int(time.time()),
        "compute_plane": "MESIE",
        "always_training": True,
        "embedded_organs": mind.organs.manifest(),
        "canon_id": getattr(mind.organs.canon, "canon_id", None),
        "canon_sha256": getattr(mind.organs.canon, "content_sha256", None),
        "language_meta": lm_meta,
        "constitutional_manifest": "constitutional_manifest.json",
        "valuable": True,
    }
    (directory / "mind_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    files = ["trainer.json", "mind_meta.json"]
    if (directory / "memory.json").exists():
        files.append("memory.json")
    for relative in (
        "language/weights.npz",
        "language/layers.npz",
        "language/tokenizer.json",
        "language/config.json",
        "language/meta.json",
        "language/constitutional_manifest.json",
    ):
        if (directory / relative).exists():
            files.append(relative)

    constitutional = build_constitutional_checkpoint(
        root=directory,
        checkpoint_id=checkpoint_id,
        checkpoint_class="organism",
        model_id=mind.model_id,
        files=files,
        parent_checkpoint_id=parent_id,
        optimization={
            "total_train_steps": int(trainer_payload.get("total_train_steps", 0)),
            "loss_history_points": len(trainer_payload.get("loss_history", [])),
            "act_count": mind.act_count,
        },
        identity={
            "born_at": mind.born_at,
            "canon_id": meta["canon_id"],
            "canon_sha256": meta["canon_sha256"],
            "embedded_organs": meta["embedded_organs"],
        },
        rollback={"safe_checkpoint_id": safe_id},
        capabilities={
            "tool_registry_receipt": os.environ.get("AURO_TOOL_REGISTRY_RECEIPT"),
            "economic_interface_receipt": os.environ.get(
                "AURO_ECONOMIC_INTERFACE_RECEIPT"
            ),
        },
        evidence={
            "resume_state_present": trainer is not None,
            "identity_state_present": True,
            "continuity_parent_known": bool(parent_id),
            "rollback_target": safe_id,
            "rollback_verified": _truthy("AURO_ROLLBACK_VERIFIED"),
            "optimization_candidate": _truthy("AURO_OPTIMIZATION_CANDIDATE"),
            "matched_benchmark": _truthy("AURO_MATCHED_BENCHMARK_PASS"),
            "protected_capabilities_pass": _truthy(
                "AURO_PROTECTED_CAPABILITIES_PASS"
            ),
            "continual_learning": True,
            "replay_or_forgetting_eval": _truthy("AURO_FORGETTING_EVAL_PASS"),
            "architecture_changed": _truthy("AURO_ARCHITECTURE_CHANGED"),
            "reversible_module_boundary": _truthy("AURO_MODULE_ROLLBACK_PASS"),
            "tool_capabilities_present": _truthy(
                "AURO_TOOL_CAPABILITIES_PRESENT"
            ),
            "tool_registry_receipt": _truthy(
                "AURO_TOOL_REGISTRY_RECEIPT_PASS"
            ),
        },
        promotion_requested=_truthy("AURO_PROMOTION_REQUESTED"),
    )
    write_constitutional_manifest(directory, constitutional)
    meta["promotion_status"] = constitutional.promotion_status
    meta["constitutional_sha256"] = constitutional.manifest_sha256
    return meta


def load_mind(
    directory: str | Path,
    *,
    chrome_mock: bool = True,
    full_runtime: bool = False,
) -> AuroMind:
    """Restore a complete mind and verify constitutional custody when present."""
    directory = Path(directory)
    constitutional = None
    if (directory / "constitutional_manifest.json").exists():
        constitutional = load_and_verify_constitutional_manifest(directory)

    language = load_lm_checkpoint(directory / "language")
    mind = AuroMind(language, chrome_mock=chrome_mock)

    memory_path = directory / "memory.json"
    if memory_path.exists() and mind.organs.memory is not None:
        from auro_native_llm.scripture.memory import ScripturalMemory

        mind.organs.memory = ScripturalMemory.load(memory_path)

    trainer_path = directory / "trainer.json"
    if trainer_path.exists() and mind.organs.trainer is not None:
        data = json.loads(trainer_path.read_text(encoding="utf-8"))
        trainer = mind.organs.trainer
        trainer.lr = float(data.get("lr", trainer.lr))
        trainer.messy_mix = float(data.get("messy_mix", trainer.messy_mix))
        trainer.seq_len = int(data.get("seq_len", trainer.seq_len))
        trainer.batch_size = int(data.get("batch_size", trainer.batch_size))
        trainer.total_train_steps = int(data.get("total_train_steps", 0))
        trainer.total_absorbs = int(data.get("total_absorbs", 0))
        trainer.loss_history = list(data.get("loss_history", []))[-200:]
        seeds = data.get("doctrine_seeds") or []
        if seeds:
            trainer.seed_doctrine(seeds)
        for item in list(data.get("experiences") or [])[-200:]:
            trainer.buffer.append(
                Experience(
                    text=str(item.get("text", "")),
                    kind=str(item.get("kind", "restore")),
                    model_id=str(item.get("model_id", mind.model_id)),
                    reward=float(item.get("reward", 0.5)),
                    meta=dict(item.get("meta", {})),
                    ts=float(item.get("ts", time.time())),
                )
            )

    meta_path = directory / "mind_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mind.act_count = int(meta.get("act_count", 0))
        mind.born_at = float(meta.get("born_at", mind.born_at))

    try:
        from auro_native_llm.physics import get_physics_engine

        language.physics = get_physics_engine()
    except Exception:
        pass

    full = full_runtime or os.environ.get("AURO_FULL_RUNTIME", "").strip() in (
        "1",
        "true",
        "yes",
    )
    if full:
        try:
            from auro_native_llm.mesie_runtime import attach_mesie_runtime

            attach_mesie_runtime(mind, lite=True)
        except Exception:
            pass
        try:
            from auro_native_llm.ghost.supervisor import GhostSupervisor

            mind.ghost = GhostSupervisor(mind)  # type: ignore[attr-defined]
        except Exception:
            mind.ghost = None  # type: ignore[attr-defined]
        try:
            from auro_native_llm.gworkspace import get_envelope

            mind.gworkspace = get_envelope(mind, chrome_mock=chrome_mock)  # type: ignore[attr-defined]
        except Exception:
            mind.gworkspace = None  # type: ignore[attr-defined]
        try:
            from auro_native_llm.sdk_runtime.injector import inject_repo_sdks

            inject_repo_sdks(mind, max_packages=80)
        except Exception:
            pass
    else:
        mind.ghost = None  # type: ignore[attr-defined]
        mind.gworkspace = None  # type: ignore[attr-defined]
        mind._runtime_lazy = True  # type: ignore[attr-defined]

    if constitutional is not None:
        mind.constitutional_checkpoint = constitutional  # type: ignore[attr-defined]
        mind.identity_health = {  # type: ignore[attr-defined]
            "verified": True,
            "checkpoint_id": constitutional.get("checkpoint_id"),
            "promotion_status": constitutional.get("promotion_status"),
            "failed_protocols": constitutional.get("failed_protocols", []),
            "aiops_domains": constitutional.get("aiops_domains", []),
        }
    return mind
