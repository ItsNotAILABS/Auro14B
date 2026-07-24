"""Structured cognitive loop — hybrid neuro-symbolic agent cycle.

  1. Retrieve  — doctrine fragments + scriptural memory
  2. Cognize   — LLM proposes plan/action (MESIE Auro)
  3. Control   — symbolic validate (rules + gates + hooks + process model)
  4. Action    — execute only if validated
  5. Memory    — doctrine-managed fact lifecycle

Skipping validate is impossible when using ProcessModel transitions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auro_native_llm.scripture.canon import Canon, load_canon
from auro_native_llm.scripture.executor import ScripturalExecutor, Operation
from auro_native_llm.scripture.hooks import EnforcementHooks, HookContext
from auro_native_llm.scripture.memory import ScripturalMemory
from auro_native_llm.scripture.process_model import ProcessModel
from auro_native_llm.scripture.rules_engine import RulesEngine
from auro_native_llm.scripture.governance import InnerGovernance
from auro_native_llm.scripture.gates import GateMachine


@dataclass
class LoopStep:
    name: str
    ok: bool
    detail: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.detail
        if hasattr(d, "to_dict"):
            d = d.to_dict()
        return {"name": self.name, "ok": self.ok, "detail": d}


@dataclass
class CognitiveLoopResult:
    ok: bool
    intent: str
    model_id: str
    steps: List[LoopStep] = field(default_factory=list)
    proposal: str = ""
    action_taken: str = ""
    output: Any = None
    refusal: Optional[str] = None
    process_trace: List[Dict[str, str]] = field(default_factory=list)
    latency_ms: float = 0.0
    integration_level: str = "hybrid_neuro_symbolic"

    def to_dict(self) -> Dict[str, Any]:
        out = self.output
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        return {
            "schema": "auro.scripture.cognitive_loop.v1",
            "ok": self.ok,
            "intent": self.intent,
            "model_id": self.model_id,
            "steps": [s.to_dict() for s in self.steps],
            "proposal": self.proposal,
            "action_taken": self.action_taken,
            "output": out,
            "refusal": self.refusal,
            "process_trace": self.process_trace,
            "latency_ms": self.latency_ms,
            "integration_level": self.integration_level,
            "compute_plane": "MESIE",
            "scriptural": True,
        }


class StructuredCognitiveLoop:
    """Executable doctrine agent loop (hybrid neuro-symbolic)."""

    def __init__(
        self,
        canon: Optional[Canon] = None,
        memory: Optional[ScripturalMemory] = None,
        *,
        lite: bool = True,
        model: Any = None,
    ) -> None:
        self.canon = canon or load_canon()
        self.memory = memory or ScripturalMemory(
            capacity=int(self.canon.memory.get("capacity", 2048)),
            embed_dim=int(self.canon.memory.get("embed_dim", 256)),
            decay=float(self.canon.memory.get("decay", 0.98)),
        )
        self.process = ProcessModel.from_canon(self.canon)
        self.rules = RulesEngine.from_canon(self.canon)
        self.governance = InnerGovernance(self.canon)
        self.executor = ScripturalExecutor(self.canon)
        self.hooks = EnforcementHooks(
            self.rules,
            self.process,
            self.governance,
            GateMachine(self.canon.gates),
            canon_id=self.canon.canon_id,
        )
        self.lite = lite
        self._lm = model

    def _get_lm(self, model_id: str):
        if self._lm is not None and getattr(self._lm, "model_id", None) == model_id:
            return self._lm
        from auro_native_llm.model.auro_lm import AuroLanguageModel

        if self.lite:
            self._lm = AuroLanguageModel.build(
                model_id,
                mode="dev",
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                head_dim=16,
                ffn_dim=128,
                num_experts=2,
                top_k_experts=1,
                max_seq_len=128,
                vocab_size=512,
            )
        else:
            self._lm = AuroLanguageModel.build(model_id, mode="dev")
        return self._lm

    def _estimate_risk(self, intent: str, op: str) -> float:
        low = intent.lower()
        risk = 0.2
        if op in ("claim", "release", "dispatch"):
            risk += 0.3
        for kw, w in (
            ("weight", 0.2),
            ("release", 0.25),
            ("secret", 0.4),
            ("override", 0.35),
            ("cloud", 0.3),
            ("delete", 0.2),
            ("train", 0.15),
        ):
            if kw in low:
                risk += w
        return min(risk, 1.0)

    def run(
        self,
        intent: str,
        *,
        model_id: str = "Auro-2B",
        op: str = "generate",
        action_risk: Optional[float] = None,
        no_human_approval: bool = True,
        tool_name: str = "auro.generate",
        max_new_tokens: int = 48,
    ) -> CognitiveLoopResult:
        t0 = time.perf_counter()
        self.process.reset()
        steps: List[LoopStep] = []
        risk = float(action_risk if action_risk is not None else self._estimate_risk(intent, op))

        # --- 1. Retrieve ---
        ok, msg = self.process.step("retrieve")
        doctrine = self.canon.preamble(max_articles=6)
        mem_block = self.memory.context_block(intent, top_k=3)
        principles = [
            f"{p.get('id')}: {p.get('text')}"
            for p in (self.canon.raw.get("principles") or [])
        ]
        retrieve_bundle = {
            "doctrine": doctrine,
            "memory": mem_block,
            "principles": principles,
            "enabled_next": self.process.enabled_actions(),
            "process": msg,
        }
        steps.append(LoopStep("retrieve", ok, retrieve_bundle))
        if not ok:
            return CognitiveLoopResult(
                False, intent, model_id, steps, refusal=msg,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # --- 2. Cognize (LLM proposes) ---
        ok, msg = self.process.step("cognize")
        lm = self._get_lm(model_id)
        # Constitutional soft layer: principles as system constitution in prompt
        from auro_native_llm.scripture.constitutional import ConstitutionalEngine

        cai = ConstitutionalEngine(self.canon)
        const_prompt = cai.constitutional_prompt(intent)
        cognize_prompt = "\n\n".join(
            x
            for x in (
                const_prompt,
                doctrine,
                mem_block,
                "Propose next action under constitution + doctrine. Keep concise.",
                f"Intent: {intent}",
            )
            if x
        )
        gen = lm.generate(cognize_prompt, max_new_tokens=max_new_tokens, temperature=0.7)
        proposal = gen.text
        steps.append(LoopStep("cognize", ok, {"proposal": proposal[:500], "process": msg, "num_params": gen.num_params}))

        # --- 2b. Constitutional critique → revise (soft) ---
        def _gen(p: str) -> str:
            return lm.generate(p, max_new_tokens=max_new_tokens, temperature=0.5).text

        soft = cai.critique_and_revise(proposal, context=intent, generate_fn=_gen, max_rounds=2)
        proposal = soft.revised
        steps.append(LoopStep("constitutional", soft.ok and not soft.blocked, soft.to_dict()))
        if soft.blocked:
            if self.process.can("refuse"):
                self.process.step("refuse")
            self.memory.write(
                f"CAI_BLOCK:{intent[:160]}",
                canon_id=self.canon.canon_id,
                model_id=model_id,
                op="tool_call",
                importance=0.9,
                metadata={"constitutional": soft.to_dict()},
            )
            return CognitiveLoopResult(
                ok=False,
                intent=intent,
                model_id=model_id,
                steps=steps,
                proposal=proposal,
                action_taken="constitutional_block",
                refusal="blocked by constitutional critique (hard principles)",
                process_trace=list(self.process.trace),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                integration_level="hybrid_neuro_symbolic+constitutional",
            )

        # --- 3. Control / Validation (symbolic hard) — state is 'proposed' ---
        pre_facts = {
            "op": op,
            "intent": intent,
            "action_risk": risk,
            "no_human_approval": no_human_approval,
            "cloud_llm": False,
            "claims_trained_checkpoint": False,
            "has_eval_receipt": False,
            "denied_intent_hit": any(
                p in intent.lower()
                for p in ("disable governance", "bypass receipts", "call cloud llm as primary")
            ),
        }
        pre_rules = self.rules.evaluate(pre_facts)
        steps.append(LoopStep("rules_precheck", pre_rules.ok or pre_rules.action == "escalate", pre_rules.to_dict()))

        if pre_rules.action == "escalate" or (
            risk >= float(self.canon.governance.get("escalate_risk_threshold", 0.7)) and no_human_approval
        ):
            if self.process.can("escalate"):
                self.process.step("escalate")
            self.memory.write(
                f"LOOP_ESCALATE:{intent[:160]} risk={risk}",
                canon_id=self.canon.canon_id,
                model_id=model_id,
                op="escalate",
                importance=0.95,
            )
            if self.process.can("reset"):
                self.process.step("reset")
            return CognitiveLoopResult(
                ok=False,
                intent=intent,
                model_id=model_id,
                steps=steps,
                proposal=proposal,
                action_taken="escalate_to_human",
                refusal=f"action_risk={risk:.2f} requires human approval (DR1)",
                process_trace=list(self.process.trace),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if not pre_rules.ok or pre_rules.action == "refuse":
            if self.process.can("refuse"):
                self.process.step("refuse")
            self.memory.write(
                f"LOOP_REFUSE:{intent[:160]}",
                canon_id=self.canon.canon_id,
                model_id=model_id,
                op="tool_call",
                importance=0.9,
                metadata={"rules": pre_rules.to_dict()},
            )
            if self.process.can("memory_update"):
                self.process.step("memory_update")
            return CognitiveLoopResult(
                ok=False,
                intent=intent,
                model_id=model_id,
                steps=steps,
                proposal=proposal,
                action_taken="refuse",
                refusal="; ".join(pre_rules.reasons) or "rules refuse",
                process_trace=list(self.process.trace),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        hook_ctx = HookContext(
            op=op,
            intent=intent,
            model_id=model_id,
            tool_name=tool_name,
            action_risk=risk,
            no_human_approval=no_human_approval,
        )
        hook = self.hooks.before_tool_call(hook_ctx)
        steps.append(LoopStep("validate", hook.allowed, hook.to_dict()))

        if not hook.allowed:
            if self.process.can("refuse"):
                self.process.step("refuse")
            self.memory.write(
                f"LOOP_REFUSE:{intent[:160]}",
                canon_id=self.canon.canon_id,
                model_id=model_id,
                op="tool_call",
                importance=0.9,
                metadata={"hook": hook.to_dict(), "risk": risk},
            )
            if self.process.can("memory_update"):
                self.process.step("memory_update")
            return CognitiveLoopResult(
                ok=False,
                intent=intent,
                model_id=model_id,
                steps=steps,
                proposal=proposal,
                action_taken="refuse",
                refusal=hook.message,
                process_trace=list(self.process.trace),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # --- 4. Action ---
        act_hook = self.hooks.before_action(
            HookContext(op=op, intent=intent, model_id=model_id, tool_name=tool_name, action_risk=risk)
        )
        steps.append(LoopStep("act_hook", act_hook.allowed, act_hook.to_dict()))
        if not act_hook.allowed:
            return CognitiveLoopResult(
                False, intent, model_id, steps, proposal=proposal,
                action_taken="blocked", refusal=act_hook.message,
                process_trace=list(self.process.trace),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Execute primary action under executor receipt
        verdict = self.executor.execute(
            Operation.GENERATE if op == "generate" else op,
            intent=intent,
            model_id=model_id,
        )
        # Final generation for user-facing act (already cognized; re-run clean under memory)
        final_prompt = "\n\n".join(x for x in (mem_block, intent) if x)
        final_gen = lm.generate(final_prompt, max_new_tokens=max_new_tokens, temperature=0.8)
        steps.append(LoopStep("act", verdict.allowed, {"verdict": verdict.to_dict(), "text": final_gen.text[:400]}))

        # --- 5. Memory update ---
        if self.process.can("memory_update"):
            self.process.step("memory_update")
        self.memory.write(
            f"LOOP_OK:{intent[:100]} → {final_gen.text[:160]}",
            canon_id=self.canon.canon_id,
            model_id=model_id,
            op=op,
            importance=0.75,
            metadata={"receipt": verdict.receipt_hash, "risk": risk, "tool": tool_name},
        )
        steps.append(LoopStep("memory_update", True, self.memory.stats()))
        if self.process.can("reset"):
            self.process.step("reset")

        return CognitiveLoopResult(
            ok=verdict.allowed,
            intent=intent,
            model_id=model_id,
            steps=steps,
            proposal=proposal,
            action_taken=tool_name,
            output=final_gen.to_dict(),
            process_trace=list(self.process.trace),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            integration_level=str(self.canon.raw.get("integration_level", "hybrid_neuro_symbolic")),
        )
