"""Full AuroMind checkpoint — language weights + memory + trainer state."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.model.checkpoint import load_checkpoint as load_lm_checkpoint
from auro_native_llm.model.checkpoint import save_checkpoint as save_lm_checkpoint
from auro_native_llm.organism.mind import AuroMind
from auro_native_llm.organism.self_train import Experience


def save_mind(mind: AuroMind, directory: str | Path) -> Dict[str, Any]:
    """Persist a complete mind organism to disk."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    lm_meta = save_lm_checkpoint(mind.language, directory / "language")

    # Memory
    if mind.organs.memory is not None:
        mind.organs.memory.save(directory / "memory.json")

    # Trainer buffer (cap for size)
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
            e.to_dict() for e in list(trainer.buffer)[-500:]
        ]
        trainer_payload["loss_history"] = list(trainer.loss_history)[-200:]
        trainer_payload["total_train_steps"] = trainer.total_train_steps
        trainer_payload["total_absorbs"] = trainer.total_absorbs
    (directory / "trainer.json").write_text(
        json.dumps(trainer_payload, indent=2), encoding="utf-8"
    )

    meta = {
        "schema": "auro.mind.checkpoint.v1",
        "model_id": mind.model_id,
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
        "valuable": True,
    }
    (directory / "mind_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_mind(
    directory: str | Path,
    *,
    chrome_mock: bool = True,
) -> AuroMind:
    """Restore a complete mind from checkpoint."""
    directory = Path(directory)
    language = load_lm_checkpoint(directory / "language")
    mind = AuroMind(language, chrome_mock=chrome_mock)

    mem_path = directory / "memory.json"
    if mem_path.exists() and mind.organs.memory is not None:
        from auro_native_llm.scripture.memory import ScripturalMemory

        mind.organs.memory = ScripturalMemory.load(mem_path)

    tr_path = directory / "trainer.json"
    if tr_path.exists() and mind.organs.trainer is not None:
        data = json.loads(tr_path.read_text(encoding="utf-8"))
        tr = mind.organs.trainer
        tr.lr = float(data.get("lr", tr.lr))
        tr.messy_mix = float(data.get("messy_mix", tr.messy_mix))
        tr.seq_len = int(data.get("seq_len", tr.seq_len))
        tr.batch_size = int(data.get("batch_size", tr.batch_size))
        tr.total_train_steps = int(data.get("total_train_steps", 0))
        tr.total_absorbs = int(data.get("total_absorbs", 0))
        tr.loss_history = list(data.get("loss_history", []))
        seeds = data.get("doctrine_seeds") or []
        if seeds:
            tr.seed_doctrine(seeds)
        for ed in data.get("experiences", []):
            tr.buffer.append(
                Experience(
                    text=str(ed.get("text", "")),
                    kind=str(ed.get("kind", "restore")),
                    model_id=str(ed.get("model_id", mind.model_id)),
                    reward=float(ed.get("reward", 0.5)),
                    meta=dict(ed.get("meta", {})),
                    ts=float(ed.get("ts", time.time())),
                )
            )

    meta_path = directory / "mind_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mind.act_count = int(meta.get("act_count", 0))
        mind.born_at = float(meta.get("born_at", mind.born_at))

    # Bind installed mesie transformers / intelligence into restored mind
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

        inject_repo_sdks(mind)
    except Exception:
        pass

    return mind
