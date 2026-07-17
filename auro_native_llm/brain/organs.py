"""Mini brains & hearts for coding, research, math — from BRAIN-AI + SOLUS.

BRAIN-AI- (FreddyCreates): NeuroEmergence Core, 873ms heartbeat, cognition houses.
SOLUS: MiniBrain + MiniHeart caretakers.
Auro: domain brains teach the SpectralGPT student via polyglot council.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.sdk.solus.constants import HEARTBEAT_MS, PHI
from mesie.sdk.solus.mini_brain import BrainThought, MiniBrain
from mesie.sdk.solus.mini_heart import MiniHeart, VitalsSnapshot


# Curriculum — teach full apps, deep research, math
CURRICULUM: Dict[str, List[str]] = {
    "code": [
        "Build a complete REST API with routes, validation, and SQLite persistence.",
        "Scaffold a React+TypeScript SPA with routing, state, and API client.",
        "Write a CLI tool with argparse, subcommands, and config file loading.",
        "Implement auth middleware, JWT issue/verify, and role gates.",
        "Create a polyglot service: Python API + Julia spectral worker + Haskell validator.",
        "Package a deployable app: Dockerfile, health endpoint, logging, env config.",
    ],
    "research": [
        "Survey MESIE spectral intelligence and cite multi-repo GitHub evidence.",
        "Compare MoE vs dense transformers for sovereign on-device inference.",
        "Deep dive: φ-heartbeat 873ms architecture from BRAIN-AI NeuroEmergence Core.",
        "Map doctrine laws PL-001..PL-010 to runtime enforcement points.",
        "Research CUDA alternatives on ARM64 and design ChaosCUDA blocked GEMM.",
        "Synthesize a lab note: autocycle SENSE→REASON→ACT→TRAIN with polyglot teachers.",
    ],
    "math": [
        "Prove or derive spectral energy identity via DFT magnitude sum.",
        "Compute φ powers and golden-ratio scaling of learning rates.",
        "Implement and verify blocked matrix multiply correctness.",
        "Estimate parameter count for SpectralGPT given h,L,ffn,experts.",
        "Softmax CE gradient derivation for tied embedding LM head.",
        "Kuramoto-style coherence metric for multi-lang engine parity.",
    ],
}


@dataclass
class DomainBrain:
    """Domain mini-brain with curriculum + SOLUS MiniBrain reasoner."""

    domain: str
    brain: MiniBrain
    curriculum: List[str] = field(default_factory=list)
    lesson_idx: int = 0
    thoughts: List[Dict[str, Any]] = field(default_factory=list)

    def next_lesson(self) -> str:
        if not self.curriculum:
            return f"{self.domain}: free exploration"
        lesson = self.curriculum[self.lesson_idx % len(self.curriculum)]
        self.lesson_idx += 1
        return lesson

    def think(self, score: float = 0.7, metric: float = 0.7, complexity: float = 0.4) -> BrainThought:
        t = self.brain.reason(
            {"score": score, "metric": metric, "complexity": complexity}
        )
        self.thoughts.append(
            {
                "conclusion": t.conclusion,
                "confidence": t.confidence,
                "evidence": t.evidence,
                "ts": time.time(),
            }
        )
        if len(self.thoughts) > 64:
            self.thoughts = self.thoughts[-64:]
        return t

    def teach_text(self) -> str:
        lesson = self.next_lesson()
        thought = self.think(score=0.75 + 0.1 * (self.lesson_idx % 3), complexity=0.5)
        return (
            f"[MINI_BRAIN domain={self.domain}]\n"
            f"LESSON: {lesson}\n"
            f"THOUGHT: {thought.conclusion} conf={thought.confidence}\n"
            f"EVIDENCE: {'; '.join(thought.evidence)}\n"
            f"HEARTBEAT_MS={HEARTBEAT_MS} PHI={PHI}\n"
            f"[/MINI_BRAIN]"
        )


@dataclass
class OrganismHeart:
    """Organism heart — BRAIN-AI 873ms lineage + SOLUS MiniHeart."""

    heart: MiniHeart
    last_vitals: Optional[VitalsSnapshot] = None

    def pulse(self, metric: float = 1.0) -> Dict[str, Any]:
        v = self.heart.pulse(sdk_metric=metric)
        self.last_vitals = v
        return {
            "bpm": v.bpm,
            "pulse_count": v.pulse_count,
            "coherence": v.coherence,
            "sdk_health": v.sdk_health,
            "caretaker": v.caretaker,
            "elapsed_ms": v.elapsed_ms,
            "heartbeat_ms": HEARTBEAT_MS,
            "lineage": ["BRAIN-AI- NeuroEmergence", "SOLUS MiniHeart", "Auro mind"],
        }


class CurriculumTeacher:
    """Teachers that feed code/research/math lessons into entangled train."""

    def __init__(self, cluster: "MiniBrainCluster") -> None:
        self.cluster = cluster

    def lesson_batch(self, n: int = 6) -> List[str]:
        out = []
        domains = list(self.cluster.brains.keys())
        for i in range(n):
            d = domains[i % len(domains)]
            out.append(self.cluster.brains[d].teach_text())
        return out

    def teach_and_train(self, mind: Any, *, steps_per_lesson: int = 1) -> Dict[str, Any]:
        lessons = self.lesson_batch(6)
        history = []
        for lesson in lessons:
            # polyglot entangled if available
            if hasattr(mind, "train_entangled"):
                history.append(
                    mind.train_entangled(lesson, steps=steps_per_lesson)
                )
            else:
                tok = mind.language.tokenizer
                ids = tok.encode(lesson[:800], max_length=64)
                arr = __import__("numpy").array([ids], dtype="int64")
                history.append(
                    mind.language.train_step(arr, arr, text_for_meaning=lesson[:200])
                )
            # heart pulse each lesson
            vitals = self.cluster.heart.pulse(metric=0.9)
            history[-1]["vitals"] = vitals
        return {
            "ok": True,
            "lessons": len(lessons),
            "domains": list(self.cluster.brains.keys()),
            "history_tail": history[-3:],
            "train_steps": mind.language.train_steps,
            "params": mind.language.num_params,
            "heart": self.cluster.heart.pulse(1.0),
        }


@dataclass
class MiniBrainCluster:
    """Code + research + math mini brains + one organism heart."""

    brains: Dict[str, DomainBrain]
    heart: OrganismHeart
    teacher: CurriculumTeacher = field(init=False)
    lineage: List[str] = field(
        default_factory=lambda: [
            "FreddyCreates/BRAIN-AI- NeuroEmergence Core",
            "mesie.sdk.solus MiniBrain/MiniHeart",
            "Auro polyglot engines/transformers/teachers",
            "ChaosCUDA local accelerator",
        ]
    )

    def __post_init__(self) -> None:
        self.teacher = CurriculumTeacher(self)

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.brain.cluster.v1",
            "domains": {
                k: {
                    "lessons": len(v.curriculum),
                    "lesson_idx": v.lesson_idx,
                    "thoughts": len(v.thoughts),
                }
                for k, v in self.brains.items()
            },
            "heart": self.heart.heart.to_dict(),
            "lineage": self.lineage,
            "curriculum_sizes": {k: len(v) for k, v in CURRICULUM.items()},
        }

    def pulse_all(self) -> Dict[str, Any]:
        thoughts = {k: v.think().conclusion for k, v in self.brains.items()}
        return {
            "thoughts": thoughts,
            "vitals": self.heart.pulse(0.95),
        }


def build_brain_cluster() -> MiniBrainCluster:
    brains = {
        "code": DomainBrain("code", MiniBrain("code"), list(CURRICULUM["code"])),
        "research": DomainBrain(
            "research", MiniBrain("research"), list(CURRICULUM["research"])
        ),
        "math": DomainBrain("math", MiniBrain("math"), list(CURRICULUM["math"])),
    }
    heart = OrganismHeart(MiniHeart("auro-organism-heart"))
    return MiniBrainCluster(brains=brains, heart=heart)
