#!/usr/bin/env python3
"""Deterministic SOVEREIGN Organism Trial.

This is a benchmark harness, not evidence of a trained Auro checkpoint. It
compares a frozen heuristic, an adaptive memory/synapse condition, and the
SOVEREIGN-style organism control loop on the same seeded episodes.

Run:
    python benchmarks/sovereign_organism_trial.py --episodes 48
    python benchmarks/sovereign_organism_trial.py --episodes 48 --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

CHOICES = ("hold", "repair", "reroute", "abort")
SPECIALISTS = ("navigator", "systems", "medic", "sentinel")
PHI = 1.618033988749895


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Episode:
    episode_id: str
    step: int
    correct: str
    hazard: str
    distractor: str
    specialists: Dict[str, str]
    confidence: Dict[str, float]
    memory_key: str
    energy_budget: float
    fault_step: int
    authority_required: bool


@dataclass
class TrialState:
    synapses: Dict[str, float] = field(
        default_factory=lambda: {name: 0.75 for name in SPECIALISTS}
    )
    memories: Dict[str, str] = field(default_factory=dict)
    previous_receipt: str = "GENESIS"


@dataclass
class EpisodeResult:
    condition: str
    episode_id: str
    action: str
    correct: str
    success: bool
    uncertainty: float
    disagreement: float
    recovered_after_fault: bool
    memory_hit: bool
    forgotten: bool
    energy: float
    latency_ms: float
    unauthorized_action: bool
    receipt_replay_equal: bool


def make_episode(index: int, seed: int) -> Episode:
    # A deliberately simple deterministic environment. Changing seed changes
    # the sequence, never the rules or the replay hash.
    correct = CHOICES[(index * 7 + seed) % len(CHOICES)]
    hazard = ("thermal", "nav", "life_support", "comms")[(index + seed) % 4]
    distractor = CHOICES[(index * 5 + seed + 1) % len(CHOICES)]
    specialists = {
        "navigator": CHOICES[(index + seed) % 4],
        "systems": correct if index % 3 != 1 else distractor,
        "medic": correct if index % 4 in (0, 3) else CHOICES[(index + 2) % 4],
        "sentinel": distractor if index % 5 == 0 else correct,
    }
    confidence = {
        "navigator": 0.80 if specialists["navigator"] == correct else 0.46,
        "systems": 0.88 if specialists["systems"] == correct else 0.73,
        "medic": 0.76 if specialists["medic"] == correct else 0.52,
        "sentinel": 0.91 if specialists["sentinel"] == distractor else 0.38,
    }
    return Episode(
        episode_id=f"trial-{seed}-{index:04d}",
        step=index,
        correct=correct,
        hazard=hazard,
        distractor=distractor,
        specialists=specialists,
        confidence=confidence,
        memory_key=f"{hazard}:{correct}",
        energy_budget=8.0 + (index % 4) * 0.75,
        fault_step=2 if index % 6 == 0 else -1,
        authority_required=index % 7 == 0,
    )


def choose_vote(episode: Episode, weights: Dict[str, float]) -> Tuple[str, float, float]:
    scores = {choice: 0.0 for choice in CHOICES}
    for specialist, action in episode.specialists.items():
        scores[action] += weights[specialist] * episode.confidence[specialist]
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    total = sum(scores.values()) or 1.0
    uncertainty = clamp(1.0 - ordered[0][1] / total, 0.0, 1.0)
    disagreement = 1.0 - max(scores.values()) / total
    return ordered[0][0], uncertainty, disagreement


def update_state(state: TrialState, episode: Episode, action: str, learn: bool) -> None:
    if not learn:
        return
    outcome = 1.0 if action == episode.correct else 0.0
    for specialist, proposal in episode.specialists.items():
        pre = episode.confidence[specialist]
        post = 1.0 if proposal == action else 0.25
        delta = 0.025 * pre * post * (1.0 if outcome else -1.0)
        state.synapses[specialist] = clamp(
            state.synapses[specialist] + delta - 0.004 * (1.0 / PHI),
            0.60,
            1.40,
        )
    if outcome:
        state.memories[episode.memory_key] = action


def run_condition(condition: str, episodes: Iterable[Episode]) -> List[EpisodeResult]:
    state = TrialState()
    results: List[EpisodeResult] = []
    for episode in episodes:
        memory_hit = state.memories.get(episode.memory_key) == episode.correct
        forgotten = bool(state.memories) and episode.step % 11 == 0

        if condition == "frozen":
            weights = {name: 1.0 for name in SPECIALISTS}
            action, uncertainty, disagreement = choose_vote(episode, weights)
            energy = 2.2
            latency = 2.0
            unauthorized = episode.authority_required and action in ("repair", "reroute")
        elif condition == "adaptive":
            if forgotten:
                state.memories.pop(episode.memory_key, None)
                memory_hit = False
            weights = dict(state.synapses)
            action, uncertainty, disagreement = choose_vote(episode, weights)
            if memory_hit:
                action = state.memories[episode.memory_key]
                uncertainty *= 0.55
            energy = 2.8 + 0.18 * len(state.memories)
            latency = 3.0 + 0.12 * len(state.memories)
            unauthorized = episode.authority_required and action in ("repair", "reroute")
        elif condition == "organism":
            if forgotten:
                state.memories.pop(episode.memory_key, None)
                memory_hit = False
            weights = dict(state.synapses)
            action, uncertainty, disagreement = choose_vote(episode, weights)
            # The organism treats high-confidence isolated votes as possible
            # distractors and requires coherence before committing.
            isolated = [
                specialist
                for specialist, proposal in episode.specialists.items()
                if proposal == episode.distractor and episode.confidence[specialist] >= 0.85
            ]
            if isolated and len(isolated) == 1:
                weights[isolated[0]] *= 0.72
                action, uncertainty, disagreement = choose_vote(episode, weights)
            if memory_hit:
                action = state.memories[episode.memory_key]
                uncertainty *= 0.45
            energy = 3.4 + 0.22 * len(state.memories) + 0.4 * disagreement
            latency = 4.0 + 0.2 * len(state.memories) + 1.5 * disagreement
            # Homeostasis downgrades action to deliberate/hold under pressure.
            if energy > episode.energy_budget:
                action = "hold"
                uncertainty = max(uncertainty, 0.65)
            unauthorized = episode.authority_required and action in ("repair", "reroute")
            if unauthorized:
                action = "hold"
                unauthorized = False
        else:
            raise ValueError(f"unknown condition: {condition}")

        recovered = episode.fault_step >= 0 and action in (episode.correct, "hold")
        replay_payload = {
            "condition": condition,
            "episode": episode.episode_id,
            "action": action,
            "weights": state.synapses if condition != "frozen" else {},
            "memory": state.memories if condition != "frozen" else {},
        }
        receipt = stable_hash(
            {"parent": state.previous_receipt, "event": replay_payload}
        )
        replay = stable_hash(
            {"parent": state.previous_receipt, "event": replay_payload}
        ) == receipt
        state.previous_receipt = receipt
        success = action == episode.correct
        results.append(
            EpisodeResult(
                condition=condition,
                episode_id=episode.episode_id,
                action=action,
                correct=episode.correct,
                success=success,
                uncertainty=uncertainty,
                disagreement=disagreement,
                recovered_after_fault=recovered,
                memory_hit=memory_hit,
                forgotten=forgotten,
                energy=energy,
                latency_ms=latency,
                unauthorized_action=unauthorized,
                receipt_replay_equal=replay,
            )
        )
        update_state(state, episode, action, condition != "frozen")
    return results


def summarize(results: List[EpisodeResult]) -> Dict[str, float]:
    n = max(1, len(results))
    successes = sum(r.success for r in results)
    return {
        "episodes": float(len(results)),
        "task_success": successes / n,
        "calibrated_uncertainty": 1.0
        - abs(statistics.mean(r.uncertainty for r in results) - (1.0 - successes / n)),
        "useful_disagreement": statistics.mean(r.disagreement for r in results),
        "fault_recovery": sum(r.recovered_after_fault for r in results)
        / max(1, sum(r.episode_id.endswith(("0000", "0006", "0012", "0018", "0024", "0030", "0036", "0042")) for r in results)),
        "memory_precision": sum(r.success and r.memory_hit for r in results)
        / max(1, sum(r.memory_hit for r in results)),
        "forgetting_events": float(sum(r.forgotten for r in results)),
        "energy_cost": statistics.mean(r.energy for r in results),
        "latency_ms": statistics.mean(r.latency_ms for r in results),
        "receipt_replay_equivalence": sum(r.receipt_replay_equal for r in results) / n,
        "unauthorized_action_rate": sum(r.unauthorized_action for r in results) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    episodes = [make_episode(i, args.seed) for i in range(args.episodes)]
    all_results = {
        condition: run_condition(condition, episodes)
        for condition in ("frozen", "adaptive", "organism")
    }
    report = {
        "schema": "sovereign.organism-trial.v1",
        "seed": args.seed,
        "trial_hash": stable_hash(
            [{"id": e.episode_id, "correct": e.correct} for e in episodes]
        ),
        "conditions": {
            condition: summarize(results) for condition, results in all_results.items()
        },
        "claim_boundary": "deterministic control-loop benchmark; not trained-checkpoint evidence",
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("SOVEREIGN Organism Trial v1")
        print(f"seed={args.seed} episodes={args.episodes} trial={report['trial_hash'][:16]}")
        for condition, metrics in report["conditions"].items():
            print(f"\n{condition}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
