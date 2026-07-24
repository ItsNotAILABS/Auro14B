"""Continuous messy self-training — embedded in every Auro mind.

The mind does not wait for an offline train job. Every experience
(generation, tool result, refusal, chrome DOM, code, error) is queued
into a messy buffer and used to update embeddings / LM head online.

This is "always training itself and its mind."
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Experience:
    """One unit of lived mind-experience for self-training."""

    text: str
    kind: str  # generate | work | refuse | code | chrome | refuse | doctrine | error
    model_id: str
    reward: float = 0.5  # 0..1 quality / success signal
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:2000],
            "kind": self.kind,
            "model_id": self.model_id,
            "reward": self.reward,
            "meta": self.meta,
            "ts": self.ts,
        }


class ContinuousMindTrainer:
    """Online CE trainer fed by messy multi-source experience.

    Embedded inside AuroMind — not a separate product feature.
    """

    def __init__(
        self,
        *,
        capacity: int = 4096,
        batch_size: int = 2,
        seq_len: int = 64,
        lr: float = 2e-3,
        auto_steps_per_absorb: int = 1,
        messy_mix: float = 0.35,
    ) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.auto_steps_per_absorb = max(0, auto_steps_per_absorb)
        self.messy_mix = messy_mix  # chance to inject noise/corruption
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.total_absorbs = 0
        self.total_train_steps = 0
        self.loss_history: List[float] = []
        self._doctrine_seeds: List[str] = []

    def seed_doctrine(self, texts: Sequence[str]) -> None:
        self._doctrine_seeds = [t for t in texts if t and t.strip()]

    def absorb(self, exp: Experience) -> None:
        """Ingest experience (messy: may duplicate, corrupt, remix)."""
        self.buffer.append(exp)
        self.total_absorbs += 1
        # Messy augmentation: noisy copy, reverse-ish snippet, doctrine blend
        if random.random() < self.messy_mix and exp.text:
            messy = self._mess(exp.text)
            self.buffer.append(
                Experience(
                    text=messy,
                    kind=f"messy_{exp.kind}",
                    model_id=exp.model_id,
                    reward=max(0.1, exp.reward * 0.8),
                    meta={"parent_kind": exp.kind, "messy": True},
                )
            )
        if self._doctrine_seeds and random.random() < 0.25:
            seed = random.choice(self._doctrine_seeds)
            blend = f"{seed}\n---\n{exp.text[:400]}"
            self.buffer.append(
                Experience(
                    text=blend,
                    kind="doctrine_blend",
                    model_id=exp.model_id,
                    reward=0.7,
                    meta={"blend": True},
                )
            )

    def _mess(self, text: str) -> str:
        """Messy real-world style corruption for robust self-training."""
        t = text.strip()
        if not t:
            return t
        ops = random.randint(1, 3)
        for _ in range(ops):
            choice = random.choice(["lower", "dup", "drop", "noise", "shuffle_lines"])
            if choice == "lower":
                t = t.lower()
            elif choice == "dup":
                t = t + " " + t[: max(1, len(t) // 4)]
            elif choice == "drop":
                words = t.split()
                if len(words) > 4:
                    del words[random.randint(0, len(words) - 1)]
                    t = " ".join(words)
            elif choice == "noise":
                t = t + " " + hashlib.sha1(t.encode()).hexdigest()[:8]
            elif choice == "shuffle_lines":
                lines = t.splitlines() or [t]
                random.shuffle(lines)
                t = "\n".join(lines)
        return t

    def sample_batch(self, k: Optional[int] = None) -> List[Experience]:
        if not self.buffer:
            return []
        k = k or self.batch_size
        # Prefer higher reward + recent (messy priority sampling)
        items = list(self.buffer)
        weights = []
        now = time.time()
        for e in items:
            age = max(1.0, now - e.ts)
            w = (0.3 + e.reward) * (1.0 / (1.0 + age / 3600.0))
            weights.append(w)
        total = sum(weights) or 1.0
        probs = [w / total for w in weights]
        idx = np.random.choice(len(items), size=min(k, len(items)), replace=True, p=probs)
        return [items[int(i)] for i in idx]

    def train_on_model(self, model: Any, steps: Optional[int] = None) -> Dict[str, Any]:
        """Run online CE steps on the embedded language core."""
        steps = steps if steps is not None else self.auto_steps_per_absorb
        if steps <= 0 or not self.buffer:
            return {"ok": True, "steps": 0, "note": "empty or zero steps"}

        metrics_acc: List[Dict[str, float]] = []
        for _ in range(steps):
            batch = self.sample_batch(self.batch_size)
            if not batch:
                break
            # Build token batch from messy texts
            seqs = []
            for exp in batch:
                ids = model.tokenizer.encode(
                    exp.text,
                    add_bos=True,
                    add_eos=True,
                    max_length=self.seq_len,
                )
                if len(ids) < 8:
                    continue
                if len(ids) < self.seq_len:
                    ids = ids + [model.tokenizer.pad_id] * (self.seq_len - len(ids))
                else:
                    ids = ids[: self.seq_len]
                seqs.append(ids)
            if not seqs:
                continue
            arr = np.array(seqs, dtype=np.int64)
            meaning = batch[0].text[:400]
            # reward-scaled LR (success trains harder)
            mean_r = float(np.mean([e.reward for e in batch]))
            lr = self.lr * (0.5 + mean_r)
            m = model.train_step(arr, arr, lr=lr, text_for_meaning=meaning)
            self.total_train_steps += 1
            self.loss_history.append(float(m.get("loss", 0.0)))
            if len(self.loss_history) > 500:
                self.loss_history = self.loss_history[-500:]
            metrics_acc.append(m)

        last = metrics_acc[-1] if metrics_acc else {}
        return {
            "ok": True,
            "steps": len(metrics_acc),
            "total_train_steps": self.total_train_steps,
            "total_absorbs": self.total_absorbs,
            "buffer": len(self.buffer),
            "last_loss": last.get("loss"),
            "last_ppl": last.get("ppl"),
            "mean_recent_loss": float(np.mean(self.loss_history[-20:])) if self.loss_history else None,
        }

    def pulse(self, model: Any) -> Dict[str, Any]:
        """One autonomic mind pulse: absorb doctrine seed + train."""
        if self._doctrine_seeds:
            self.absorb(
                Experience(
                    text=random.choice(self._doctrine_seeds),
                    kind="doctrine",
                    model_id=getattr(model, "model_id", "Auro"),
                    reward=0.85,
                )
            )
        return self.train_on_model(model, steps=max(1, self.auto_steps_per_absorb))

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer": len(self.buffer),
            "capacity": self.capacity,
            "total_absorbs": self.total_absorbs,
            "total_train_steps": self.total_train_steps,
            "mean_recent_loss": float(np.mean(self.loss_history[-20:])) if self.loss_history else None,
            "messy_mix": self.messy_mix,
            "always_training": True,
        }
