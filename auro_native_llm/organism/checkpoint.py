"""Full AuroMind checkpoint with signed constitutional identity continuity."""
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
from auro_native_llm.substrate import build_constitutional_checkpoint, load_and_verify_constitutional_manifest, write_constitutional_manifest
from auro_native_llm.substrate.promotion_evidence import evidence_flags, load_signed_promotion_evidence


def _promotion_context() -> tuple[bool, Dict[str, Any], str | None, str | None]:
    requested = os.environ.get("AURO_PROMOTION_REQUESTED", "").strip().lower() in {"1", "true", "yes", "on"}
    signing_key = os.environ.get("AURO_CHECKPOINT_SIGNING_KEY") or None
    authorized_by = os.environ.get("AURO_CHECKPOINT_AUTHORIZER") or None
    if not requested:
        return False, {}, signing_key, authorized_by
    evidence_path = os.environ.get("AURO_PROMOTION_EVIDENCE")
    if not evidence_path:
        raise RuntimeError("AURO_PROMOTION_EVIDENCE is required for promotion")
    receipt = load_signed_promotion_evidence(evidence_path, signing_key or "")
    return True, {**evidence_flags(receipt), "promotion_receipt": receipt}, signing_key, authorized_by


def save_mind(mind: AuroMind, directory: str | Path) -> Dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lm_meta = save_lm_checkpoint(mind.language, directory / "language")
    if mind.organs.memory is not None:
        mind.organs.memory.save(directory / "memory.json")
    trainer = mind.organs.trainer
    trainer_payload: Dict[str, Any] = {"stats": {}, "experiences": []}
    if trainer is not None:
        trainer_payload.update({
            "stats": trainer.stats(), "lr": trainer.lr, "messy_mix": trainer.messy_mix,
            "seq_len": trainer.seq_len, "batch_size": trainer.batch_size,
            "doctrine_seeds": list(trainer._doctrine_seeds)[:50],
            "experiences": [item.to_dict() for item in list(trainer.buffer)[-500:]],
            "loss_history": list(trainer.loss_history)[-200:],
            "total_train_steps": trainer.total_train_steps, "total_absorbs": trainer.total_absorbs,
        })
    (directory / "trainer.json").write_text(json.dumps(trainer_payload, indent=2), encoding="utf-8")

    checkpoint_id = os.environ.get("AURO_ORGANISM_CHECKPOINT_ID", f"{mind.model_id}-organism-{mind.act_count}-{int(time.time())}")
    parent_id = os.environ.get("AURO_PARENT_ORGANISM_CHECKPOINT_ID") or None
    promotion_requested, signed_evidence, signing_key, authorized_by = _promotion_context()
    safe_id = signed_evidence.get("rollback_target") or os.environ.get("AURO_SAFE_ORGANISM_CHECKPOINT_ID") or parent_id
    meta = {
        "schema": "auro.mind.checkpoint.v3", "model_id": mind.model_id, "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": parent_id, "parameter_target": mind.config.parameter_target, "tier": mind.config.tier,
        "num_params_live": mind.language.num_params, "act_count": mind.act_count, "born_at": mind.born_at,
        "saved_at_unix": int(time.time()), "compute_plane": "MESIE", "always_training": True,
        "embedded_organs": mind.organs.manifest(), "canon_id": getattr(mind.organs.canon, "canon_id", None),
        "canon_sha256": getattr(mind.organs.canon, "content_sha256", None), "language_meta": lm_meta,
        "constitutional_manifest": "constitutional_manifest.json", "valuable": True,
    }
    # Persist the pre-manifest metadata so it can be included in the artifact inventory.
    (directory / "mind_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = ["trainer.json", "mind_meta.json"]
    if (directory / "memory.json").exists(): files.append("memory.json")
    for relative in ("language/weights.npz", "language/layers.npz", "language/tokenizer.json", "language/config.json", "language/meta.json", "language/constitutional_manifest.json"):
        if (directory / relative).exists(): files.append(relative)
    constitutional = build_constitutional_checkpoint(
        root=directory, checkpoint_id=checkpoint_id, checkpoint_class="organism", model_id=mind.model_id,
        files=files, parent_checkpoint_id=parent_id,
        optimization={"total_train_steps": int(trainer_payload.get("total_train_steps", 0)), "loss_history_points": len(trainer_payload.get("loss_history", [])), "act_count": mind.act_count},
        identity={"born_at": mind.born_at, "canon_id": meta["canon_id"], "canon_sha256": meta["canon_sha256"], "embedded_organs": meta["embedded_organs"]},
        rollback={"safe_checkpoint_id": safe_id},
        capabilities={"tool_registry_receipt": (signed_evidence.get("promotion_receipt") or {}).get("tools"), "economic_interface_receipt": os.environ.get("AURO_ECONOMIC_INTERFACE_RECEIPT")},
        evidence={**signed_evidence, "resume_state_present": trainer is not None, "identity_state_present": True,
            "continuity_parent_known": bool(parent_id), "rollback_target": safe_id},
        promotion_requested=promotion_requested, signing_key=signing_key, authorized_by=authorized_by,
    )
    write_constitutional_manifest(directory, constitutional)
    # Rewrite persisted metadata after status and authorization are known, then refresh its inventory hash and reseal.
    meta.update({"promotion_status": constitutional.promotion_status, "constitutional_sha256": constitutional.manifest_sha256,
        "authorization_hmac_sha256": constitutional.authorization_hmac_sha256, "authorized_by": constitutional.authorized_by})
    (directory / "mind_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    constitutional.files["mind_meta.json"] = {"sha256": __import__("hashlib").sha256((directory / "mind_meta.json").read_bytes()).hexdigest(), "bytes": (directory / "mind_meta.json").stat().st_size}
    constitutional.seal(signing_key=signing_key, authorized_by=authorized_by)
    write_constitutional_manifest(directory, constitutional)
    meta["constitutional_sha256"] = constitutional.manifest_sha256
    meta["authorization_hmac_sha256"] = constitutional.authorization_hmac_sha256
    (directory / "mind_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # The final metadata write changes its hash once more; update and seal one final deterministic time.
    constitutional.files["mind_meta.json"] = {"sha256": __import__("hashlib").sha256((directory / "mind_meta.json").read_bytes()).hexdigest(), "bytes": (directory / "mind_meta.json").stat().st_size}
    constitutional.seal(signing_key=signing_key, authorized_by=authorized_by)
    write_constitutional_manifest(directory, constitutional)
    return meta


def load_mind(directory: str | Path, *, chrome_mock: bool = True, full_runtime: bool = False, allow_quarantined: bool = False) -> AuroMind:
    directory = Path(directory)
    constitutional = None
    if (directory / "constitutional_manifest.json").exists():
        constitutional = load_and_verify_constitutional_manifest(directory, signing_key=os.environ.get("AURO_CHECKPOINT_SIGNING_KEY"), require_promoted=not allow_quarantined)
    elif not allow_quarantined:
        raise RuntimeError("constitutional manifest is required for organism load")
    language = load_lm_checkpoint(directory / "language", allow_quarantined=allow_quarantined)
    mind = AuroMind(language, chrome_mock=chrome_mock)
    memory_path = directory / "memory.json"
    if memory_path.exists() and mind.organs.memory is not None:
        from auro_native_llm.scripture.memory import ScripturalMemory
        mind.organs.memory = ScripturalMemory.load(memory_path)
    trainer_path = directory / "trainer.json"
    if trainer_path.exists() and mind.organs.trainer is not None:
        data = json.loads(trainer_path.read_text(encoding="utf-8")); trainer = mind.organs.trainer
        trainer.lr = float(data.get("lr", trainer.lr)); trainer.messy_mix = float(data.get("messy_mix", trainer.messy_mix))
        trainer.seq_len = int(data.get("seq_len", trainer.seq_len)); trainer.batch_size = int(data.get("batch_size", trainer.batch_size))
        trainer.total_train_steps = int(data.get("total_train_steps", 0)); trainer.total_absorbs = int(data.get("total_absorbs", 0))
        trainer.loss_history = list(data.get("loss_history", []))[-200:]
        if data.get("doctrine_seeds"): trainer.seed_doctrine(data["doctrine_seeds"])
        for item in list(data.get("experiences") or [])[-200:]:
            trainer.buffer.append(Experience(text=str(item.get("text", "")), kind=str(item.get("kind", "restore")), model_id=str(item.get("model_id", mind.model_id)), reward=float(item.get("reward", 0.5)), meta=dict(item.get("meta", {})), ts=float(item.get("ts", time.time()))))
    meta_path = directory / "mind_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")); mind.act_count = int(meta.get("act_count", 0)); mind.born_at = float(meta.get("born_at", mind.born_at))
    try:
        from auro_native_llm.physics import get_physics_engine
        language.physics = get_physics_engine()
    except Exception: pass
    full = full_runtime or os.environ.get("AURO_FULL_RUNTIME", "").strip().lower() in {"1", "true", "yes"}
    if full:
        try:
            from auro_native_llm.mesie_runtime import attach_mesie_runtime
            attach_mesie_runtime(mind, lite=True)
        except Exception: pass
        try:
            from auro_native_llm.ghost.supervisor import GhostSupervisor
            mind.ghost = GhostSupervisor(mind)  # type: ignore[attr-defined]
        except Exception: mind.ghost = None  # type: ignore[attr-defined]
        try:
            from auro_native_llm.gworkspace import get_envelope
            mind.gworkspace = get_envelope(mind, chrome_mock=chrome_mock)  # type: ignore[attr-defined]
        except Exception: mind.gworkspace = None  # type: ignore[attr-defined]
    else:
        mind.ghost = None  # type: ignore[attr-defined]
        mind.gworkspace = None  # type: ignore[attr-defined]
        mind._runtime_lazy = True  # type: ignore[attr-defined]
    if constitutional is not None:
        mind.constitutional_checkpoint = constitutional  # type: ignore[attr-defined]
        mind.identity_health = {"verified": True, "authorized": constitutional.get("authorized", False),
            "checkpoint_id": constitutional.get("checkpoint_id"), "promotion_status": constitutional.get("promotion_status"),
            "failed_protocols": constitutional.get("failed_protocols", []), "aiops_domains": constitutional.get("aiops_domains", [])}  # type: ignore[attr-defined]
    return mind
