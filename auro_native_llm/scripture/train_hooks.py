"""Training hooks — doctrine-bound corpus + memory + receipts during pretrain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from auro_native_llm.scripture.substrate import ScripturalSubstrate


@dataclass
class ScripturalTrainConfig:
    model_id: str = "Auro-2B"
    steps: int = 20
    seed_texts: Optional[List[str]] = None


def doctrine_seed_texts(canon_principle: str) -> List[str]:
    return [
        canon_principle,
        "Auro compute plane is MESIE. Meaning engines latin sanskrit nahuatl construct residual memory.",
        "GATE_IDENTITY GATE_CAPABILITY GATE_PROOF GATE_CONTAINMENT GATE_MODEL_EVAL bind claims.",
        "Multi-embedded sub-agents host only allowed children under role matrix.",
        "No false checkpoint claims without receipts. Spectral memory is first class.",
        "Scriptural Systems Architecture: symbols construct behavior memory relationships possible world.",
    ]


def run_scriptural_training(
    *,
    model_id: str = "Auro-2B",
    steps: int = 20,
    substrate: Optional[ScripturalSubstrate] = None,
    texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run N doctrine-governed train steps with memory embedding."""
    sub = substrate or ScripturalSubstrate()
    seeds = texts or doctrine_seed_texts(sub.canon.principle)
    # include article texts as training law
    for a in sub.canon.articles:
        seeds.append(f"{a.id} {a.title}: {a.text}")

    history: List[Dict[str, Any]] = []
    for i in range(steps):
        text = seeds[i % len(seeds)]
        result = sub.train_step_governed(model_id, text)
        history.append(
            {
                "step": i + 1,
                "ok": result.ok,
                "output": result.output,
                "receipt": result.verdict.get("receipt_hash"),
            }
        )
        if not result.ok:
            break

    mem_path = sub.persist_memory("deliverables/auro_scripture/memory.json")
    rec_path = sub.save_receipts("deliverables/auro_scripture/receipts.jsonl")
    return {
        "schema": "auro.scripture.train_report.v1",
        "ok": all(h["ok"] for h in history) if history else False,
        "model_id": model_id,
        "steps_requested": steps,
        "steps_completed": len([h for h in history if h["ok"]]),
        "history": history,
        "memory_path": mem_path,
        "receipts_path": rec_path,
        "canon_id": sub.canon.canon_id,
        "memory_stats": sub.memory.stats(),
        "compute_plane": "MESIE",
        "scriptural": True,
    }
