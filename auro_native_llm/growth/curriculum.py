"""Domain curriculum for the Auro career ladder.

Five training domains, each a milestone with explicit done-criteria:

- general mathematics, physics, architecture, ancient geometry: synthetic
  problem generators (verifiable prompts with exact checkers). Synthetic is
  fine for *mechanics verification* -- proving the curriculum machinery
  works: problems generate, verifiers judge, gates open and close. Real
  corpora + a real tokenizer are the remaining work for real training.
- coding with execution: the tool-use milestone made concrete. Training is
  NOT static code text -- the loop embeds real sandboxed runtimes
  (``exec_sandbox``): the model completes whole modules, the sandbox runs
  them against tests, and the observed output is the feedback signal.

A rung may carry several active milestones at once (the "career with
multiple projects" shape); each milestone defines its own done-criteria
and the rung advances when its gates are met.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

SYNTHETIC_V1 = "synthetic-v1 (mechanics verification; real corpora are future work)"


# ----------------------------------------------------------------------------
# Problems + verifiers
# ----------------------------------------------------------------------------

@dataclass
class Problem:
    domain: str
    prompt: str            # text ending with "Answer:"
    answer: str            # canonical answer string
    difficulty: int        # 1..3
    verify: Callable[[str], bool]


def _extract_number(resp: str) -> Optional[float]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", resp.replace(",", ""))
    return float(nums[-1]) if nums else None


def int_verifier(answer: int) -> Callable[[str], bool]:
    def check(resp: str) -> bool:
        n = _extract_number(resp)
        return n is not None and abs(n - answer) < 1e-9
    return check


def float_verifier(answer: float, tol: float = 0.011) -> Callable[[str], bool]:
    def check(resp: str) -> bool:
        n = _extract_number(resp)
        return n is not None and abs(n - answer) <= tol
    return check


def choice_verifier(answer: str) -> Callable[[str], bool]:
    want = answer.strip().lower()
    def check(resp: str) -> bool:
        return want in resp.strip().lower()
    return check


# ----------------------------------------------------------------------------
# Synthetic problem generators (mechanics verification)
# ----------------------------------------------------------------------------

def gen_math_problem(rng: np.random.Generator, difficulty: int = 1) -> Problem:
    kind = rng.choice(["mod", "linear", "gcd", "prime"])
    if kind == "mod":
        a = int(rng.integers(2, 60)); x = int(rng.integers(0, 200))
        b = int(rng.integers(0, 60)); m = int(rng.integers(64, 512))
        ans = (a * x + b) % m
        return Problem("math", f"Compute ({a}*{x}+{b}) mod {m}.\nAnswer:",
                       str(ans), difficulty, int_verifier(ans))
    if kind == "linear":
        a, x = int(rng.integers(2, 12)), int(rng.integers(-20, 20))
        b = int(rng.integers(-30, 30))
        c = a * x + b
        return Problem("math", f"Solve for x: {a}x + ({b}) = {c}.\nAnswer:",
                       str(x), difficulty, int_verifier(x))
    if kind == "gcd":
        a, b = int(rng.integers(2, 200)), int(rng.integers(2, 200))
        ans = math.gcd(a, b)
        return Problem("math", f"gcd({a}, {b}) = ?\nAnswer:",
                       str(ans), difficulty, int_verifier(ans))
    # prime test (small, deterministic)
    p = int(rng.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                        4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]))
    is_p = all(p % d for d in range(2, int(math.isqrt(p)) + 1)) and p > 1
    ans = "yes" if is_p else "no"
    return Problem("math", f"Is {p} prime? Answer yes or no.\nAnswer:",
                   ans, difficulty, choice_verifier(ans))


def gen_physics_problem(rng: np.random.Generator, difficulty: int = 1) -> Problem:
    kind = rng.choice(["velocity", "force", "convert"])
    if kind == "velocity":
        u, a, t = (int(rng.integers(0, 30)), int(rng.integers(-5, 10)),
                   int(rng.integers(1, 12)))
        ans = u + a * t
        return Problem("physics",
                       f"A body moves with initial velocity {u} m/s and constant "
                       f"acceleration {a} m/s^2. Velocity after {t} s (v = u + at) = ?\nAnswer:",
                       str(ans), difficulty, int_verifier(ans))
    if kind == "force":
        m, a = int(rng.integers(1, 50)), int(rng.integers(1, 20))
        ans = m * a
        return Problem("physics",
                       f"A mass of {m} kg accelerates at {a} m/s^2. Force (F = ma) in newtons = ?\nAnswer:",
                       str(ans), difficulty, int_verifier(ans))
    v = round(float(rng.uniform(10, 120)), 1)
    ans = round(v * 1000 / 3600, 2)
    return Problem("physics", f"Convert {v} km/h to m/s (2 decimals).\nAnswer:",
                   f"{ans:.2f}", difficulty, float_verifier(ans))


PHI = (1 + math.sqrt(5)) / 2  # the repo's phi_math constant, centralized here


def gen_geometry_problem(rng: np.random.Generator, difficulty: int = 1) -> Problem:
    kind = rng.choice(["pythagoras", "circle", "golden", "triangle"])
    if kind == "pythagoras":
        k = int(rng.integers(1, 8))
        a, b, c = 3 * k, 4 * k, 5 * k
        return Problem("ancient-geometry",
                       f"A right triangle has legs {a} and {b}. Hypotenuse = ?\nAnswer:",
                       str(c), difficulty, int_verifier(c))
    if kind == "circle":
        r = int(rng.integers(1, 20))
        ans = round(2 * math.pi * r, 2)
        return Problem("ancient-geometry",
                       f"A circle has radius {r}. Circumference (2 decimals) = ?\nAnswer:",
                       f"{ans:.2f}", difficulty, float_verifier(ans))
    if kind == "golden":
        w = int(rng.integers(2, 30))
        ans = round(w * PHI, 2)
        return Problem("ancient-geometry",
                       f"A golden rectangle (length/width = phi = 1.618...) has width {w}. "
                       f"Length (2 decimals) = ?\nAnswer:",
                       f"{ans:.2f}", difficulty, float_verifier(ans))
    b, h = int(rng.integers(2, 40)), int(rng.integers(2, 40))
    ans = b * h / 2
    ok = ans == int(ans)
    return Problem("ancient-geometry",
                   f"Triangle with base {b} and height {h}. Area = ?\nAnswer:",
                   str(int(ans)) if ok else f"{ans}",
                   difficulty, int_verifier(int(ans)) if ok else float_verifier(ans))


def gen_architecture_problem(rng: np.random.Generator, difficulty: int = 1) -> Problem:
    kind = rng.choice(["plan", "bays", "arch", "steps"])
    if kind == "plan":
        w = int(rng.integers(4, 40))
        ans = round(w * PHI, 1)
        return Problem("architecture",
                       f"A hall follows golden-ratio proportion (length = width * 1.618). "
                       f"Width {w} m: length (1 decimal) = ?\nAnswer:",
                       f"{ans:.1f}", difficulty, float_verifier(ans, tol=0.06))
    if kind == "bays":
        n, bay = int(rng.integers(3, 10)), int(rng.integers(2, 8))
        total = n * bay
        return Problem("architecture",
                       f"A colonnade has {n} equal bays spanning {total} m. Bay width = ?\nAnswer:",
                       str(bay), difficulty, int_verifier(bay))
    if kind == "arch":
        s = int(rng.integers(2, 20)) * 2
        return Problem("architecture",
                       f"A semicircular arch spans {s} m. Its rise (half the span) = ?\nAnswer:",
                       str(s // 2), difficulty, int_verifier(s // 2))
    n_steps, riser = int(rng.integers(3, 12)), int(rng.integers(10, 20))
    total_rise = n_steps * riser
    return Problem("architecture",
                   f"A stair climbs {total_rise} cm in {n_steps} equal risers. Riser height = ?\nAnswer:",
                   str(riser), difficulty, int_verifier(riser))


GENERATORS: Dict[str, Callable] = {
    "math": gen_math_problem,
    "physics": gen_physics_problem,
    "ancient-geometry": gen_geometry_problem,
    "architecture": gen_architecture_problem,
}


# ----------------------------------------------------------------------------
# Milestones + done-criteria
# ----------------------------------------------------------------------------

@dataclass
class EvalGate:
    metric: str          # "accuracy" | "pass_rate"
    threshold: float     # in (0, 1)
    eval_set: str        # what is measured, and how much
    min_rungs: int = 1


@dataclass
class Milestone:
    id: str
    domain: str
    title: str
    description: str
    rung_start: int
    rung_end: int        # inclusive
    data_provenance: str
    done: EvalGate
    notes: str = ""

    def active_at(self, rung: int) -> bool:
        return self.rung_start <= rung <= self.rung_end


MILESTONES: List[Milestone] = [
    Milestone(
        id="math-foundations", domain="math",
        title="General mathematics: arithmetic and algebra",
        description="Synthetic arithmetic/algebra problems (modular arithmetic, "
                    "linear equations, gcd, primality) with exact verifiers.",
        rung_start=0, rung_end=4, data_provenance=SYNTHETIC_V1,
        done=EvalGate(metric="accuracy", threshold=0.85,
                      eval_set="1000 held-out synthetic problems", min_rungs=2),
        notes="Real training replaces the generator with a math corpus + tokenizer; "
              "the gate shape (accuracy on held-out, threshold, min rungs) stays.",
    ),
    Milestone(
        id="ancient-geometry", domain="ancient-geometry",
        title="Ancient geometry: Euclid and the golden ratio",
        description="Pythagorean triples, circle mensuration, golden-rectangle "
                    "proportions (phi = (1+sqrt(5))/2), triangle area.",
        rung_start=0, rung_end=5, data_provenance=SYNTHETIC_V1,
        done=EvalGate(metric="accuracy", threshold=0.85,
                      eval_set="1000 held-out synthetic problems", min_rungs=2),
        notes="Pairs with math-foundations in the early rungs; the phi thread "
              "connects to the repo's phi_math tradition.",
    ),
    Milestone(
        id="physics", domain="physics",
        title="Physics: kinematics, dynamics, units",
        description="v = u + at, F = ma, unit conversions with tolerance verifiers.",
        rung_start=3, rung_end=8, data_provenance=SYNTHETIC_V1,
        done=EvalGate(metric="accuracy", threshold=0.80,
                      eval_set="1000 held-out synthetic problems", min_rungs=2),
        notes="Starts once the math gate is plausibly passable; numeric tolerance "
              "in verifiers models measurement error.",
    ),
    Milestone(
        id="architecture", domain="architecture",
        title="Architecture: proportion, bays, arches, stairs",
        description="Golden-ratio plans, colonnade bays, semicircular arches, "
                    "stair risers -- numeric layout reasoning with verifiers.",
        rung_start=5, rung_end=10, data_provenance=SYNTHETIC_V1,
        done=EvalGate(metric="accuracy", threshold=0.80,
                      eval_set="1000 held-out synthetic problems", min_rungs=2),
        notes="Synthetic layout arithmetic now; real training uses plan corpora "
              "and constraint-checking, not just numbers.",
    ),
    Milestone(
        id="coding-execution", domain="coding",
        title="Coding with execution: whole modules in the sandbox",
        description=(
            "NOT static code text. The loop embeds real sandboxed Python and "
            "PowerShell runtimes (exec_sandbox): the model completes/extends/debugs "
            "WHOLE modules, the sandbox runs them against tests, and the observed "
            "stdout/stderr becomes the feedback signal for the next attempt. "
            "Reward = sandbox pass rate; the data is generated by interaction."
        ),
        rung_start=8, rung_end=14, data_provenance="sandbox-interactive-v1",
        done=EvalGate(metric="pass_rate", threshold=0.70,
                      eval_set="200 held-out whole-module tasks, sandbox-executed",
                      min_rungs=2),
        notes="Starts late: tool use needs a model big enough to write coherent "
              "modules. Process-level isolation here is loop mechanics, not a "
              "security boundary -- real training jails untrusted code properly.",
    ),
]


def milestones_for_rung(rung: int) -> List[Milestone]:
    return [m for m in MILESTONES if m.active_at(rung)]


def curriculum_summary() -> str:
    lines = []
    for m in MILESTONES:
        lines.append(
            f"[{m.id}] rungs {m.rung_start}-{m.rung_end}: {m.title}\n"
            f"    done when {m.done.metric} >= {m.done.threshold} on {m.done.eval_set} "
            f"(min {m.done.min_rungs} rungs) [{m.data_provenance}]"
        )
    return "\n".join(lines)
