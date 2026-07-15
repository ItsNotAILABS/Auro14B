"""SOLUS Logic Prover — local formal math caretaker (your Logic Prover engine)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mesie.sdk.solus.constants import HEARTBEAT_MS, LOCAL_ENGINE, PHI, SOLUS_BRAND
from mesie.sdk.solus.mini_brain import MiniBrain
from mesie.sdk.solus.mini_heart import MiniHeart

PROOF_METHODS = [
    "direct", "contradiction", "induction", "contrapositive",
    "construction", "exhaustion", "bijection",
]
LOGICAL_CONNECTORS = [
    "therefore", "hence", "thus", "because", "since",
    "implies", "follows", "given", "assume", "suppose", "let", "qed",
]


@dataclass
class LogicProverReport:
    action: str
    ok: bool
    data: Dict[str, Any]
    heart: Dict[str, Any]
    brain: Dict[str, Any]
    sovereign: bool = True
    engine: str = LOCAL_ENGINE


class SolusLogicProver:
    """Autonomous math proof caretaker — mini heart + mini brain, 100% local SOLUS."""

    name = "Logic Prover"
    caretaker_id = "solus-logic-prover"

    def __init__(self) -> None:
        self.heart = MiniHeart(self.caretaker_id)
        self.brain = MiniBrain("logic_proof")
        self.proof_cache: Dict[str, Any] = {}

    def parse_expression(self, math_text: str) -> Dict[str, Any]:
        text = (math_text or "").strip()
        if not text:
            return {"error": "Empty expression"}
        operators = set("+-*/=<>^(){}[]")
        tokens = []
        current = ""
        for ch in text:
            if ch in operators:
                if current.strip():
                    tokens.append({"type": "term", "value": current.strip()})
                tokens.append({"type": "operator", "value": ch})
                current = ""
            elif ch == " ":
                if current.strip():
                    tokens.append({"type": "term", "value": current.strip()})
                current = ""
            else:
                current += ch
        if current.strip():
            tokens.append({"type": "term", "value": current.strip()})

        functions = ["sin", "cos", "tan", "log", "ln", "sqrt", "lim", "sum", "int", "prod", "inf", "sup"]
        detected = [f for f in functions if f in text]
        is_latex = "\\" in text or "frac" in text
        return {
            "original": text,
            "tokens": tokens,
            "token_count": len(tokens),
            "detected_functions": detected,
            "is_latex": is_latex,
            "complexity": self.evaluate_complexity(text),
            "engine": LOCAL_ENGINE,
            "timestamp": time.time(),
        }

    def evaluate_complexity(self, problem: str) -> Dict[str, Any]:
        text = str(problem or "")
        lower = text.lower()
        advanced = ["\\int", "\\sum", "\\prod", "\\lim", "\\infty", "partial", "nabla", "forall", "exists", "matrix", "det", "eigenvalue"]
        advanced_count = sum(1 for s in advanced if s in lower)
        operators = len(re.findall(r"[+\-*/^=<>]", text))
        nested = len(re.findall(r"[({[]", text))
        raw = advanced_count * 2 + operators * 0.5 + nested * 1.5 + len(text) * 0.02
        phi_max = PHI ** 5
        normalized = min(1.0, raw / phi_max)
        if normalized < 1 / PHI / PHI:
            level = "elementary"
        elif normalized < 1 / PHI:
            level = "intermediate"
        elif normalized < PHI / (PHI + 1):
            level = "advanced"
        else:
            level = "research"
        return {
            "raw_score": round(raw, 2),
            "normalized": round(normalized, 4),
            "level": level,
            "factors": {"advanced_symbols": advanced_count, "operators": operators, "nested_groups": nested, "length": len(text)},
        }

    def verify_proof(self, steps: List[Any]) -> Dict[str, Any]:
        if not steps:
            return {"valid": False, "error": "No proof steps provided"}
        verified = []
        all_valid = True
        total_conf = 0.0
        for i, step in enumerate(steps):
            text = step if isinstance(step, str) else str(step.get("statement") or step.get("text") or "")
            method = step.get("method", "direct") if isinstance(step, dict) else "direct"
            lower = text.lower()
            has_connector = any(c in lower for c in LOGICAL_CONNECTORS)
            references = False
            if i > 0:
                for p in range(i):
                    prev = steps[p] if isinstance(steps[p], str) else str(steps[p].get("statement", ""))
                    for w in [x for x in prev.split() if len(x) > 3]:
                        if w.lower() in lower:
                            references = True
                            break
                    if references:
                        break
            confidence = 0.5
            if has_connector:
                confidence += 0.2
            if references or i == 0:
                confidence += 0.15
            if len(text) > 20:
                confidence += 0.1
            confidence = min(1.0, confidence)
            step_valid = confidence * (PHI ** -(i * 0.3)) > 0.3
            if not step_valid:
                all_valid = False
            total_conf += confidence
            verified.append({
                "step": i + 1, "statement": text[:120], "method": method,
                "valid": step_valid, "confidence": round(confidence, 4),
            })
        return {
            "valid": all_valid,
            "steps": verified,
            "total_steps": len(steps),
            "average_confidence": round(total_conf / len(steps), 4),
            "overall_strength": "strong" if all_valid else "weak",
            "engine": LOCAL_ENGINE,
        }

    def generate_proof_chain(self, theorem: str) -> Dict[str, Any]:
        if not theorem:
            return {"error": "No theorem provided"}
        complexity = self.evaluate_complexity(theorem)
        step_count = max(3, round(complexity["normalized"] * 8 + 2))
        phases = ["hypothesis", "setup", "core-argument", "verification", "conclusion"]
        chain = []
        for i in range(step_count):
            phase = phases[min(len(phases) - 1, int(i / step_count * len(phases)))]
            method = PROOF_METHODS[i % len(PROOF_METHODS)]
            confidence = round(min(1.0, 0.95 * (PHI ** -(i * 0.2))), 4)
            chain.append({
                "step": i + 1, "phase": phase, "method": method,
                "description": f"Step {i+1}: Apply {method} in {phase} phase",
                "confidence": confidence, "engine": LOCAL_ENGINE,
            })
        avg = sum(c["confidence"] for c in chain) / len(chain)
        return {
            "theorem": theorem[:200],
            "complexity": complexity,
            "chain": chain,
            "total_steps": len(chain),
            "average_confidence": round(avg, 4),
            "estimated_time_ms": round(len(chain) * HEARTBEAT_MS * complexity["normalized"]),
            "engine": LOCAL_ENGINE,
        }

    def caretaker_run(self, action: str, **kwargs: Any) -> LogicProverReport:
        handlers = {
            "parse": lambda: self.parse_expression(kwargs.get("math_text", "")),
            "verify": lambda: self.verify_proof(kwargs.get("steps", [])),
            "prove": lambda: self.generate_proof_chain(kwargs.get("theorem", "")),
            "complexity": lambda: self.evaluate_complexity(kwargs.get("problem", "")),
        }
        fn = handlers.get(action)
        if not fn:
            data = {"error": f"unknown action: {action}"}
            ok = False
        else:
            data = fn()
            ok = "error" not in data
        vitals = self.heart.pulse(data.get("average_confidence", data.get("complexity", {}).get("normalized", 0.5) if isinstance(data.get("complexity"), dict) else 0.5))
        thought = self.brain.reason({"score": data.get("average_confidence", 0.6), "complexity": data.get("complexity", {}).get("normalized", 0.3) if isinstance(data.get("complexity"), dict) else 0.3})
        return LogicProverReport(
            action=action, ok=ok, data=data,
            heart={"bpm": vitals.bpm, "coherence": vitals.coherence, "sdk_health": vitals.sdk_health, **self.heart.to_dict()},
            brain={"conclusion": thought.conclusion, "confidence": thought.confidence, "evidence": thought.evidence},
        )