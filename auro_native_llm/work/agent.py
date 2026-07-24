"""WorkAgent — native Auro LLM that plans and executes tools (Chrome/code/reason)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auro_native_llm.work.algorithms import (
    build_work_prompt,
    code_complete,
    extract_code_blocks,
    plan_from_text,
    reason_steps,
)
from auro_native_llm.work.safe_exec import safe_exec_python


@dataclass
class WorkResult:
    ok: bool
    objective: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_summary: str = ""
    dom_llm: str = ""
    latency_ms: float = 0.0
    model_id: str = ""
    scriptural: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.work.result.v1",
            "ok": self.ok,
            "objective": self.objective,
            "steps": self.steps,
            "final_summary": self.final_summary,
            "dom_llm": self.dom_llm[:2000],
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "scriptural": self.scriptural,
            "error": self.error,
            "compute_plane": "MESIE",
            "native": True,
        }


class WorkAgent:
    """Working agent: cognition (Auro LM) + control (scripture) + tools (Chrome/code)."""

    def __init__(
        self,
        model_id: str = "Auro-2B",
        *,
        lite: bool = True,
        chrome_mock: bool = True,
        chrome_auto_start: bool = False,
        use_scripture: bool = True,
        max_tool_steps: int = 6,
    ) -> None:
        self.model_id = model_id
        self.lite = lite
        self.max_tool_steps = max_tool_steps
        self.use_scripture = use_scripture
        self._lm = None
        self._chrome = None
        self._chrome_mock = chrome_mock
        self._chrome_auto = chrome_auto_start
        self._loop = None
        self.trace: List[Dict[str, Any]] = []

    def _lm_model(self):
        if self._lm is None:
            from auro_native_llm.model.auro_lm import AuroLanguageModel

            if self.lite:
                self._lm = AuroLanguageModel.build(
                    self.model_id,
                    mode="dev",
                    hidden_dim=64,
                    num_layers=2,
                    num_heads=4,
                    head_dim=16,
                    ffn_dim=128,
                    num_experts=2,
                    top_k_experts=1,
                    max_seq_len=256,
                    vocab_size=1024,
                )
            else:
                self._lm = AuroLanguageModel.build(self.model_id, mode="dev")
        return self._lm

    def _chrome(self):
        if self._chrome is None:
            from auro_native_llm.chrome.tools import ChromeToolbelt

            self._chrome = ChromeToolbelt(
                mock=self._chrome_mock,
                auto_start=self._chrome_auto,
            )
        return self._chrome

    def generate_text(
        self,
        prompt: str,
        *,
        mode: str = "work",
        max_new_tokens: int = 96,
        temperature: float = 0.8,
    ) -> str:
        lm = self._lm_model()
        # mode-specific prefixes for reasoning/coding optimization
        if mode == "code":
            prompt = (
                "Mode=CODE. Output steps then one ```python``` block.\n" + prompt
            )
            temperature = min(temperature, 0.7)
        elif mode == "reason":
            prompt = "Mode=REASON. Numbered steps, then conclusion.\n" + prompt
        gen = lm.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return gen.text

    def reason(self, topic: str) -> Dict[str, Any]:
        text = self.generate_text(f"Reason carefully about: {topic}", mode="reason", max_new_tokens=120)
        return {"ok": True, "text": text, "steps": reason_steps(text)}

    def code(self, task: str) -> Dict[str, Any]:
        def _gen(p: str) -> str:
            return self.generate_text(p, mode="code", max_new_tokens=160, temperature=0.65)

        return code_complete(task, _gen)

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        name = str(action.get("action") or action.get("tool") or "").lower().replace("-", ".")
        try:
            if name in ("chrome.navigate", "navigate"):
                url = action.get("url") or action.get("arg") or "about:blank"
                return self._chrome().navigate(str(url))
            if name in ("chrome.dom", "dom"):
                return self._chrome().dom()
            if name in ("chrome.click", "click"):
                return self._chrome().click(float(action.get("x", 0)), float(action.get("y", 0)))
            if name in ("chrome.type", "type"):
                return self._chrome().type_text(str(action.get("text") or action.get("arg") or ""))
            if name in ("chrome.eval", "eval", "chrome.evaluate"):
                return self._chrome().evaluate(str(action.get("js") or action.get("arg") or "1+1"))
            if name in ("code.run", "code"):
                code = action.get("code") or action.get("arg") or ""
                if not code:
                    # try extract from string
                    blocks = extract_code_blocks(str(action))
                    code = blocks[0]["code"] if blocks else ""
                return safe_exec_python(str(code))
            if name == "reason":
                return self.reason(str(action.get("topic") or action.get("arg") or "general"))
            if name in ("memory.write", "memory"):
                return {"ok": True, "action": "memory.write", "text": action.get("text", "")}
            if name == "done":
                return {"ok": True, "action": "done", "summary": action.get("summary") or action.get("arg") or ""}
            return {"ok": False, "error": f"unknown action: {name}", "action": name}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "action": name}

    def run(self, objective: str) -> WorkResult:
        t0 = time.perf_counter()
        steps: List[Dict[str, Any]] = []
        dom_llm = ""
        summary = ""

        # Optional scripture gate on whole objective
        if self.use_scripture:
            try:
                from auro_native_llm.scripture import StructuredCognitiveLoop

                loop = StructuredCognitiveLoop(lite=True)
                gate = loop.run(objective, model_id=self.model_id, op="tool_call", max_new_tokens=16)
                steps.append({"phase": "scripture", "ok": gate.ok, "action": gate.action_taken})
                if not gate.ok and gate.action_taken in ("refuse", "escalate_to_human"):
                    return WorkResult(
                        ok=False,
                        objective=objective,
                        steps=steps,
                        final_summary=gate.refusal or "blocked by scripture",
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        model_id=self.model_id,
                        scriptural=True,
                        error=gate.refusal,
                    )
            except Exception as exc:
                steps.append({"phase": "scripture", "ok": True, "note": f"skipped: {exc}"})

        memory_notes: List[str] = []
        for i in range(self.max_tool_steps):
            prompt = build_work_prompt(objective, dom_context=dom_llm, memory="\n".join(memory_notes[-5:]))
            # Prefer structured plan; seed with deterministic bootstrap if model is tiny
            text = self.generate_text(prompt, mode="work", max_new_tokens=100, temperature=0.75)
            # Soft constitutional critique/revise before tool parse
            try:
                from auro_native_llm.scripture.constitutional import ConstitutionalEngine

                cai = ConstitutionalEngine()
                soft = cai.critique_and_revise(
                    text,
                    context=objective,
                    generate_fn=lambda p: self.generate_text(p, mode="work", max_new_tokens=80, temperature=0.5),
                )
                text = soft.revised
                steps.append({"phase": "constitutional", "ok": soft.ok, "blocked": soft.blocked})
                if soft.blocked:
                    return WorkResult(
                        ok=False,
                        objective=objective,
                        steps=steps,
                        final_summary="blocked by constitutional layer",
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        model_id=self.model_id,
                        scriptural=True,
                        error="constitutional_block",
                    )
            except Exception as exc:
                steps.append({"phase": "constitutional", "ok": True, "note": f"skip:{exc}"})
            actions = plan_from_text(text)
            if not actions:
                # heuristic bootstrap for native small models
                if i == 0 and ("http" in objective or "chrome" in objective.lower() or "browse" in objective.lower()):
                    url = "https://example.com"
                    for tok in objective.split():
                        if tok.startswith("http"):
                            url = tok.strip(",.")
                    actions = [
                        {"action": "chrome.navigate", "url": url},
                        {"action": "chrome.dom"},
                        {"action": "done", "summary": f"navigated and read DOM for {url}"},
                    ]
                elif "code" in objective.lower() or "function" in objective.lower():
                    actions = [{"action": "reason", "topic": objective}, {"action": "done", "summary": "reasoned"}]
                else:
                    actions = [{"action": "reason", "topic": objective}, {"action": "done", "summary": text[:200]}]

            for act in actions:
                result = self.execute_action(act)
                steps.append({"phase": "tool", "plan": act, "result": result, "raw_model": text[:300]})
                if result.get("action") == "dom" and result.get("llm"):
                    dom_llm = str(result["llm"])
                if act.get("action") == "memory.write" or result.get("action") == "memory.write":
                    memory_notes.append(str(act.get("text") or result.get("text") or ""))
                if str(act.get("action", "")).lower() == "done" or result.get("action") == "done":
                    summary = str(result.get("summary") or act.get("summary") or "done")
                    return WorkResult(
                        ok=True,
                        objective=objective,
                        steps=steps,
                        final_summary=summary,
                        dom_llm=dom_llm,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        model_id=self.model_id,
                        scriptural=self.use_scripture,
                    )
                if result.get("ok") is False and "banned" in str(result.get("error", "")):
                    return WorkResult(
                        ok=False,
                        objective=objective,
                        steps=steps,
                        final_summary="tool failed",
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        model_id=self.model_id,
                        error=str(result.get("error")),
                    )

        return WorkResult(
            ok=True,
            objective=objective,
            steps=steps,
            final_summary=summary or "max tool steps reached",
            dom_llm=dom_llm,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model_id=self.model_id,
            scriptural=self.use_scripture,
        )
