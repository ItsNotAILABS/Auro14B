"""AuroMind — production embedded model organism.

Every capability AI systems need is *inside* the mind:
  natural language · reason · code · work · chrome DOM ·
  scripture · constitutional critique · memory · continuous training.

Each family size is an AuroMind with the same organs (scaled core only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel, AuroGenerateResult
from auro_native_llm.model.config import AuroLMConfig, family_config
from auro_native_llm.organism.modules import EmbeddedOrgans, build_organs
from auro_native_llm.organism.self_train import ContinuousMindTrainer, Experience
from auro_native_llm.work.algorithms import (
    code_complete,
    extract_code_blocks,
    plan_from_text,
    reason_steps,
)


@dataclass
class MindResult:
    ok: bool
    kind: str
    model_id: str
    output: Any = None
    train_pulse: Optional[Dict[str, Any]] = None
    memory_wrote: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    embedded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        out = self.output
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        return {
            "schema": "auro.mind.result.v1",
            "ok": self.ok,
            "kind": self.kind,
            "model_id": self.model_id,
            "output": out,
            "train_pulse": self.train_pulse,
            "memory_wrote": self.memory_wrote,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "embedded": True,
            "compute_plane": "MESIE",
            "always_training": True,
        }


class AuroMind:
    """Full native model mind — organs embedded, always self-training."""

    def __init__(
        self,
        language: AuroLanguageModel,
        *,
        chrome_mock: bool = True,
        absorb_every_act: bool = True,
        train_every_act: bool = True,
    ) -> None:
        self.language = language
        self.model_id = language.model_id
        self.config = language.config
        self.absorb_every_act = absorb_every_act
        self.train_every_act = train_every_act
        self.organs: EmbeddedOrgans = build_organs(
            language, chrome_mock=chrome_mock, lite_tools=True
        )
        self.act_count = 0
        self.born_at = time.time()

    # ---------------------------------------------------------------- factory
    @classmethod
    def build(
        cls,
        model_id: str = "Auro-2B",
        mode: str = "dev",
        *,
        lite: bool = True,
        chrome_mock: bool = True,
        **overrides: Any,
    ) -> "AuroMind":
        """Build a complete mind for any family member — full organs always."""
        if lite and mode == "dev":
            # Smoke floor = MESIE spectral_gpt_tiny geometry + full arsenal.
            # (Previously under-scaled to 64-dim / 2L — that ignored the stack.)
            from auro_native_llm.model.config import mesie_preset_dims

            tiny = mesie_preset_dims("spectral_gpt_tiny")
            for k, v in tiny.items():
                overrides.setdefault(k, v)
            overrides.setdefault("mesie_preset", "spectral_gpt_tiny")
            overrides.setdefault("use_cross_modal", True)
            overrides.setdefault("use_spectral_encoder", True)
            overrides.setdefault("use_moe", True)
            overrides.setdefault("positional_encoding", "rotary")
            overrides.setdefault("normalization", "rms_norm")
            overrides.setdefault("activation", "swiglu")
            overrides.setdefault("qk_norm", True)
            overrides.setdefault("num_modalities", 8)
            # Cap vocab/seq for fast smoke without abandoning MESIE width/depth
            overrides.setdefault("max_seq_len", min(int(tiny["max_seq_len"]), 512))
            overrides.setdefault("vocab_size", min(int(tiny["vocab_size"]), 4096))
        lang = AuroLanguageModel.build(model_id, mode=mode, **overrides)  # type: ignore[arg-type]
        mind = cls(lang, chrome_mock=chrome_mock)
        # Runtime: bind installed mesie transformers / intelligence / helix / connectome
        try:
            from auro_native_llm.mesie_runtime import attach_mesie_runtime

            attach_mesie_runtime(mind, lite=lite)
        except Exception:
            pass
        # GHOST supervisory plane (MESIE-first hybrid + receipt chain)
        try:
            from auro_native_llm.ghost.supervisor import GhostSupervisor

            mind.ghost = GhostSupervisor(mind)  # type: ignore[attr-defined]
        except Exception:
            mind.ghost = None  # type: ignore[attr-defined]
        # Google virtual envelope: AI sandbox + collab workspace
        try:
            from auro_native_llm.gworkspace import get_envelope

            mind.gworkspace = get_envelope(mind, chrome_mock=chrome_mock)  # type: ignore[attr-defined]
        except Exception:
            mind.gworkspace = None  # type: ignore[attr-defined]
        # Runtime: inject SDKs from all Medina / ItsNotAILABS / FreddyCreates repos
        try:
            from auro_native_llm.sdk_runtime.injector import inject_repo_sdks

            inject_repo_sdks(mind)
        except Exception:
            pass
        return mind

    # ---------------------------------------------------------------- absorb
    def _absorb(
        self,
        text: str,
        kind: str,
        *,
        reward: float = 0.5,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Every act feeds the mind. Training is part of living."""
        trainer: ContinuousMindTrainer = self.organs.trainer
        exp = Experience(
            text=text,
            kind=kind,
            model_id=self.model_id,
            reward=reward,
            meta=meta or {},
        )
        if self.absorb_every_act and trainer is not None:
            trainer.absorb(exp)
        mem_ok = False
        if self.organs.memory is not None and self.organs.canon is not None:
            try:
                self.organs.memory.write(
                    f"{kind}:{text[:300]}",
                    canon_id=self.organs.canon.canon_id,
                    model_id=self.model_id,
                    op=kind if kind in self.organs.canon.allowed_ops else "generate",
                    importance=reward,
                    metadata=meta or {},
                )
                mem_ok = True
            except Exception:
                mem_ok = False
        pulse = None
        if self.train_every_act and trainer is not None:
            pulse = trainer.train_on_model(self.language, steps=1)
        self.act_count += 1
        return {"memory_wrote": mem_ok, "train_pulse": pulse}

    def pulse(self) -> Dict[str, Any]:
        """Autonomic heartbeat — doctrine + messy train without external ask."""
        if self.organs.trainer is None:
            return {"ok": False}
        return self.organs.trainer.pulse(self.language)

    # ---------------------------------------------------------------- capabilities (all embedded)
    def generate(self, prompt: str, **kw: Any) -> MindResult:
        t0 = time.perf_counter()
        # Doctrine preamble embedded
        preamble = ""
        if self.organs.constitutional is not None:
            preamble = self.organs.constitutional.constitutional_prompt(prompt)[:1200]
        mem = ""
        if self.organs.memory is not None:
            mem = self.organs.memory.context_block(prompt, top_k=3)
        # symbolic compression programs
        try:
            from auro_native_llm.symbolic.compress import SymbolicCompressor

            sym = SymbolicCompressor().expand_context(prompt, top_k=3)
        except Exception:
            sym = ""
        full = "\n\n".join(x for x in (preamble, mem, sym, prompt) if x)

        # Soft constitutional on empty is skip; hard check intent
        if self.organs.governance is not None:
            dec = self.organs.governance.review("generate", prompt, model_id=self.model_id)
            if not dec.allowed:
                absorb = self._absorb(prompt, "refuse", reward=0.3, meta={"reasons": dec.reasons})
                return MindResult(
                    ok=False,
                    kind="generate",
                    model_id=self.model_id,
                    error="; ".join(dec.reasons),
                    train_pulse=absorb.get("train_pulse"),
                    memory_wrote=absorb.get("memory_wrote", False),
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

        gen: AuroGenerateResult = self.language.generate(full, **kw)
        # Soft revise
        text = gen.text
        if self.organs.constitutional is not None:
            soft = self.organs.constitutional.critique_and_revise(text, context=prompt)
            text = soft.revised
            gen.metadata["constitutional"] = soft.to_dict()
            gen.text = text

        absorb = self._absorb(
            f"PROMPT:{prompt}\nOUT:{text}",
            "generate",
            reward=0.6 + 0.2 * float(gen.confidence) if hasattr(gen, "confidence") else 0.7,
            meta={"num_params": gen.num_params, "neuro": gen.metadata.get("neuro_emergence")},
        )
        return MindResult(
            ok=True,
            kind="generate",
            model_id=self.model_id,
            output=gen.to_dict(),
            train_pulse=absorb.get("train_pulse"),
            memory_wrote=absorb.get("memory_wrote", False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def think_answer(self, prompt: str, **kw: Any) -> Dict[str, Any]:
        """Public usable API: THINK then ANSWER with NeuroEmergence + agents context."""
        t0 = time.perf_counter()
        if self.organs.governance is not None:
            dec = self.organs.governance.review("generate", prompt, model_id=self.model_id)
            if not dec.allowed:
                return {
                    "ok": False,
                    "error": "; ".join(dec.reasons),
                    "thinking": "",
                    "answer": "",
                }
        mem = ""
        if self.organs.memory is not None:
            mem = self.organs.memory.context_block(prompt, top_k=3)
        full = f"{mem}\n{prompt}" if mem else prompt
        # ensure neuro attached
        if getattr(self.language, "_neuro", None) is None:
            try:
                from auro_native_llm.neuro.emergence import NeuroBridge

                NeuroBridge(self.language)
            except Exception:
                pass
        result = self.language.think_answer(full, **kw)
        self._absorb(
            f"THINK:{result.get('thinking','')[:500]}\nANSWER:{result.get('answer','')[:500]}",
            "think_answer",
            reward=0.85,
            meta={"neuro": result.get("neuro")},
        )
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        result["mind_id"] = self.model_id
        return result

    def agents(self) -> Any:
        """Internal agent manager (spawn/run_team)."""
        if getattr(self.organs, "agent_manager", None) is None:
            from auro_native_llm.agents.manager import AgentManager

            self.organs.agent_manager = AgentManager(self)  # type: ignore[attr-defined]
        return self.organs.agent_manager

    def ghost_run(self, intent: str, **kw: Any) -> Dict[str, Any]:
        """GHOST supervisor: policy → MESIE Ghost Node → optional LLM → receipts."""
        if getattr(self, "ghost", None) is None:
            from auro_native_llm.ghost.supervisor import GhostSupervisor

            self.ghost = GhostSupervisor(self)  # type: ignore[attr-defined]
        return self.ghost.run(intent, **kw).to_dict()  # type: ignore[attr-defined]

    def google_envelope(self, *, chrome_mock: bool = True, force: bool = False) -> Any:
        """AI's sandboxed Google suite + collab link (Chrome, Gmail, Drive, Search…)."""
        from auro_native_llm.gworkspace import get_envelope

        env = get_envelope(self, chrome_mock=chrome_mock, force=force)
        self.gworkspace = env  # type: ignore[attr-defined]
        return env

    def google(self, surface: str, action: str = "list", **kw: Any) -> Dict[str, Any]:
        """Unified Google workspace act: mail/drive/chrome/search/collab/calendar/sites."""
        env = self.google_envelope()
        out = env.act(surface, action, **kw)
        # absorb for self-train
        try:
            self._absorb(
                f"GOOGLE[{surface}.{action}] {str(out)[:400]}",
                "google_workspace",
                reward=0.75 if out.get("ok") else 0.4,
                meta={"surface": surface, "action": action},
            )
        except Exception:
            pass
        return out

    def collab(self, text: str, **kw: Any) -> Dict[str, Any]:
        """Post into shared user+AI project workspace (AI replies in-thread)."""
        return self.google("collab", "post", text=text, author=kw.get("author", "user"))

    def dual_think(self, intent: str, *, steps: int = 3, n_cores: int = 0) -> Dict[str, Any]:
        """Python=AI, Julia=BRAIN with distributed virtual physics cores.

        Environment is not the ceiling. Julia runs physics cores; Python acts.
        """
        from auro_native_llm.dual import DualOrganism

        org = DualOrganism(self, n_cores=n_cores)
        self.dual = org  # type: ignore[attr-defined]
        return org.think(intent, steps=steps).to_dict()

    def hybrid(self, prompt: str, *, force_mesie_only: bool = False) -> Dict[str, Any]:
        """GHOST/MESIE killer path: deterministic work first; LLM only if justified."""
        from auro_native_llm.vproc.hybrid import HybridRuntime

        rt = HybridRuntime(self)
        self.vproc = rt  # type: ignore[attr-defined]
        return rt.execute(prompt, force_mesie_only=force_mesie_only, save=True)

    def power_stack(self, prompt: str, *, rounds: int = 5, physics_steps: int = 3) -> Dict[str, Any]:
        """Deep physics + economic engines + algorithms + transformers together."""
        from auro_native_llm.engines.orchestra import PowerStack

        stack = PowerStack(self)
        self.power_stack_engine = stack  # type: ignore[attr-defined]
        return stack.run(prompt, rounds=rounds, physics_steps=physics_steps)

    def ready(self, *, output_dir: str = "artifacts/auro-readiness") -> Dict[str, Any]:
        """NOVA promotion readiness — coding + reasoning measured before any claim."""
        from auro_native_llm.intelligence.promotion import run_readiness

        return run_readiness(self, output_dir=output_dir)

    def code_solve(self, prompt: str, tests: str = "", entrypoint: str = "solution") -> Dict[str, Any]:
        """Real coding orchestrator — execute tests, not vibes."""
        from auro_foundry.coding_harness import CodingTask
        from auro_native_llm.intelligence.coding import CodingOrchestrator

        task = CodingTask(
            task_id="adhoc",
            prompt=prompt,
            tests=tests or "assert solution is not None\n",
            entrypoint=entrypoint,
        )
        att = CodingOrchestrator(self).solve_task(task)
        return {
            "ok": att.passed,
            "passed": att.passed,
            "source": att.source,
            "method": att.method,
            "attempts": att.attempts,
            "stderr": att.stderr[-1000:],
        }

    def reason(self, topic: str) -> MindResult:
        t0 = time.perf_counter()
        r = self.generate(
            f"Mode=REASON. Numbered steps then conclusion.\nTopic: {topic}",
            max_new_tokens=96,
            temperature=0.75,
        )
        text = (r.output or {}).get("text", "") if isinstance(r.output, dict) else ""
        steps = reason_steps(text)
        absorb = self._absorb(f"REASON:{topic}\n{text}", "reason", reward=0.75)
        return MindResult(
            ok=r.ok,
            kind="reason",
            model_id=self.model_id,
            output={"text": text, "steps": steps},
            train_pulse=absorb.get("train_pulse"),
            memory_wrote=absorb.get("memory_wrote", False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=r.error,
        )

    def code(self, task: str) -> MindResult:
        t0 = time.perf_counter()

        def _g(p: str) -> str:
            return self.language.generate(p, max_new_tokens=120, temperature=0.65).text

        out = code_complete(task, _g)
        absorb = self._absorb(f"CODE:{task}\n{out.get('primary_code','')}", "code", reward=0.8)
        return MindResult(
            ok=True,
            kind="code",
            model_id=self.model_id,
            output=out,
            train_pulse=absorb.get("train_pulse"),
            memory_wrote=absorb.get("memory_wrote", False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def chrome(self, action: str, **kwargs: Any) -> MindResult:
        t0 = time.perf_counter()
        belt = self.organs.chrome
        if belt is None:
            return MindResult(False, "chrome", self.model_id, error="chrome organ missing")
        try:
            if action == "navigate":
                out = belt.navigate(kwargs.get("url", "about:blank"))
            elif action == "dom":
                out = belt.dom()
            elif action == "eval":
                out = belt.evaluate(kwargs.get("js", "document.title"))
            elif action == "type":
                out = belt.type_text(kwargs.get("text", ""))
            elif action == "click":
                out = belt.click(float(kwargs.get("x", 0)), float(kwargs.get("y", 0)))
            else:
                out = {"ok": False, "error": f"unknown {action}"}
            reward = 0.85 if out.get("ok", True) else 0.2
            text = str(out.get("llm") or out)[:800]
            absorb = self._absorb(f"CHROME:{action}:{text}", "chrome", reward=reward)
            return MindResult(
                ok=bool(out.get("ok", True)),
                kind="chrome",
                model_id=self.model_id,
                output=out,
                train_pulse=absorb.get("train_pulse"),
                memory_wrote=absorb.get("memory_wrote", False),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            absorb = self._absorb(f"CHROME_ERR:{action}:{exc}", "error", reward=0.1)
            return MindResult(
                False,
                "chrome",
                self.model_id,
                error=str(exc),
                train_pulse=absorb.get("train_pulse"),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def work(self, objective: str) -> MindResult:
        """Full work loop embedded — plan tools, execute, train on every step."""
        t0 = time.perf_counter()
        steps: List[Dict[str, Any]] = []

        # Hard governance on objective
        if self.organs.governance is not None:
            dec = self.organs.governance.review("tool_call", objective, model_id=self.model_id)
            if not dec.allowed:
                absorb = self._absorb(objective, "refuse", reward=0.25)
                return MindResult(
                    ok=False,
                    kind="work",
                    model_id=self.model_id,
                    error="; ".join(dec.reasons),
                    train_pulse=absorb.get("train_pulse"),
                    memory_wrote=absorb.get("memory_wrote", False),
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

        # Constitutional soft on objective
        if self.organs.constitutional is not None:
            soft = self.organs.constitutional.critique_and_revise(objective)
            if soft.blocked:
                absorb = self._absorb(objective, "refuse", reward=0.2)
                return MindResult(
                    False,
                    "work",
                    self.model_id,
                    error="constitutional_block",
                    train_pulse=absorb.get("train_pulse"),
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

        dom_ctx = ""
        # Bootstrap browse if URL-like
        actions: List[Dict[str, Any]] = []
        if "http" in objective or "browse" in objective.lower() or "chrome" in objective.lower():
            url = "https://example.com"
            for tok in objective.split():
                if tok.startswith("http"):
                    url = tok.strip(",.")
            actions = [
                {"action": "chrome.navigate", "url": url},
                {"action": "chrome.dom"},
                {"action": "done", "summary": f"worked DOM for {url}"},
            ]
        else:
            # Mind plans
            plan_text = self.language.generate(
                f"Emit tool plan JSON actions for: {objective}",
                max_new_tokens=80,
                temperature=0.7,
            ).text
            if self.organs.constitutional is not None:
                plan_text = self.organs.constitutional.critique_and_revise(
                    plan_text, context=objective
                ).revised
            actions = plan_from_text(plan_text) or [
                {"action": "reason", "topic": objective},
                {"action": "done", "summary": "reasoned"},
            ]

        summary = ""
        for act in actions:
            name = str(act.get("action", "")).lower()
            if name.startswith("chrome.") or name in ("navigate", "dom", "click", "type", "eval"):
                mapped = name if name.startswith("chrome.") else f"chrome.{name}"
                if mapped == "chrome.navigate":
                    mr = self.chrome("navigate", url=act.get("url") or act.get("arg"))
                elif mapped in ("chrome.dom",):
                    mr = self.chrome("dom")
                    if isinstance(mr.output, dict) and mr.output.get("llm"):
                        dom_ctx = str(mr.output["llm"])
                elif mapped == "chrome.eval":
                    mr = self.chrome("eval", js=act.get("js") or act.get("arg") or "1")
                else:
                    mr = self.chrome(mapped.split(".", 1)[-1], **act)
                steps.append({"act": act, "result": mr.to_dict()})
            elif name in ("reason",):
                mr = self.reason(str(act.get("topic") or objective))
                steps.append({"act": act, "result": mr.to_dict()})
            elif name in ("code", "code.run"):
                mr = self.code(str(act.get("code") or act.get("arg") or objective))
                steps.append({"act": act, "result": mr.to_dict()})
            elif name == "done":
                summary = str(act.get("summary") or "done")
                steps.append({"act": act, "result": {"ok": True}})
                break
            else:
                steps.append({"act": act, "result": {"ok": False, "error": "unknown"}})

        absorb = self._absorb(
            f"WORK:{objective}\nSUM:{summary}\nDOM:{dom_ctx[:400]}",
            "work",
            reward=0.85,
            meta={"steps": len(steps)},
        )
        return MindResult(
            ok=True,
            kind="work",
            model_id=self.model_id,
            output={
                "summary": summary or "completed",
                "steps": steps,
                "dom": dom_ctx[:2000],
            },
            train_pulse=absorb.get("train_pulse"),
            memory_wrote=absorb.get("memory_wrote", False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def think(self, prompt: str) -> MindResult:
        """Alias: natural language act that still trains the mind."""
        return self.generate(prompt)

    def monaco(self, action: str, **kwargs: Any) -> MindResult:
        t0 = time.perf_counter()
        organ = self.organs.monaco
        if organ is None:
            return MindResult(False, "monaco", self.model_id, error="monaco organ missing")
        try:
            if action == "create":
                out = organ.create(
                    kwargs.get("content", ""),
                    language=kwargs.get("language", "python"),
                    filename=kwargs.get("filename", "main.py"),
                )
            elif action == "get":
                out = organ.get(kwargs["session_id"])
            elif action == "set":
                out = organ.set_content(kwargs["session_id"], kwargs.get("content", ""))
            elif action == "append":
                out = organ.append(kwargs["session_id"], kwargs.get("text", ""))
            elif action == "replace":
                out = organ.replace(kwargs["session_id"], kwargs.get("old", ""), kwargs.get("new", ""))
            elif action == "insert":
                out = organ.insert(kwargs["session_id"], int(kwargs.get("offset", 0)), kwargs.get("text", ""))
            elif action == "list":
                out = organ.list_sessions()
            elif action == "ui":
                out = organ.ui_payload(kwargs["session_id"])
            else:
                out = {"ok": False, "error": f"unknown monaco action {action}"}
            absorb = self._absorb(f"MONACO:{action}:{out}", "monaco", reward=0.8 if out.get("ok") else 0.2)
            return MindResult(
                bool(out.get("ok")), "monaco", self.model_id, output=out,
                train_pulse=absorb.get("train_pulse"), memory_wrote=absorb.get("memory_wrote", False),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            absorb = self._absorb(f"MONACO_ERR:{exc}", "error", reward=0.1)
            return MindResult(False, "monaco", self.model_id, error=str(exc), train_pulse=absorb.get("train_pulse"),
                              latency_ms=(time.perf_counter() - t0) * 1000.0)

    def jupyter(self, action: str, **kwargs: Any) -> MindResult:
        t0 = time.perf_counter()
        organ = self.organs.jupyter
        if organ is None:
            return MindResult(False, "jupyter", self.model_id, error="jupyter organ missing")
        try:
            if action == "create":
                out = organ.create(kwargs.get("title", "Auro Notebook"))
            elif action == "get":
                out = organ.get(kwargs["notebook_id"])
            elif action == "add_cell":
                out = organ.add_cell(
                    kwargs["notebook_id"], kwargs.get("source", ""),
                    cell_type=kwargs.get("cell_type", "code"),
                )
            elif action == "execute":
                out = organ.execute_cell(kwargs["notebook_id"], kwargs["cell_id"])
            elif action == "execute_all":
                out = organ.execute_all(kwargs["notebook_id"])
            elif action == "export":
                out = organ.export_ipynb(kwargs["notebook_id"])
            elif action == "list":
                out = organ.list_notebooks()
            else:
                out = {"ok": False, "error": f"unknown jupyter action {action}"}
            absorb = self._absorb(f"JUPYTER:{action}:{out}", "jupyter", reward=0.85 if out.get("ok") else 0.2)
            return MindResult(
                bool(out.get("ok")), "jupyter", self.model_id, output=out,
                train_pulse=absorb.get("train_pulse"), memory_wrote=absorb.get("memory_wrote", False),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            absorb = self._absorb(f"JUPYTER_ERR:{exc}", "error", reward=0.1)
            return MindResult(False, "jupyter", self.model_id, error=str(exc), train_pulse=absorb.get("train_pulse"),
                              latency_ms=(time.perf_counter() - t0) * 1000.0)

    def search(self, query: str, **kwargs: Any) -> MindResult:
        t0 = time.perf_counter()
        organ = self.organs.search
        if organ is None:
            return MindResult(False, "search", self.model_id, error="search organ missing")
        try:
            out = organ.search(query, **kwargs)
            absorb = self._absorb(f"SEARCH:{query}:{out}", "search", reward=0.75 if out.get("ok") else 0.2)
            return MindResult(
                bool(out.get("ok")), "search", self.model_id, output=out,
                train_pulse=absorb.get("train_pulse"), memory_wrote=absorb.get("memory_wrote", False),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            absorb = self._absorb(f"SEARCH_ERR:{exc}", "error", reward=0.1)
            return MindResult(False, "search", self.model_id, error=str(exc), train_pulse=absorb.get("train_pulse"),
                              latency_ms=(time.perf_counter() - t0) * 1000.0)

    def mcp(self, action: str, **kwargs: Any) -> MindResult:
        t0 = time.perf_counter()
        organ = self.organs.mcp
        if organ is None:
            return MindResult(False, "mcp", self.model_id, error="mcp organ missing")
        try:
            if action == "spin_up":
                out = organ.spin_up()
            elif action == "list_tools":
                out = organ.list_tools()
            elif action == "call":
                out = organ.call(kwargs.get("tool", ""), kwargs.get("arguments") or {})
            elif action == "shutdown":
                out = organ.hub.shutdown()
            else:
                out = {"ok": False, "error": f"unknown mcp action {action}"}
            absorb = self._absorb(f"MCP:{action}:{out}", "mcp", reward=0.85 if out.get("ok") else 0.2)
            return MindResult(
                bool(out.get("ok")), "mcp", self.model_id, output=out,
                train_pulse=absorb.get("train_pulse"), memory_wrote=absorb.get("memory_wrote", False),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            absorb = self._absorb(f"MCP_ERR:{exc}", "error", reward=0.1)
            return MindResult(False, "mcp", self.model_id, error=str(exc), train_pulse=absorb.get("train_pulse"),
                              latency_ms=(time.perf_counter() - t0) * 1000.0)

    def teach(self, domains: Optional[List[str]] = None) -> MindResult:
        t0 = time.perf_counter()
        cur = self.organs.curriculum
        if cur is None:
            return MindResult(False, "teach", self.model_id, error="curriculum missing")
        out = cur.teach(self, domains)
        return MindResult(
            True, "teach", self.model_id, output=out,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            train_pulse=out.get("train_pulse"),
            memory_wrote=True,
        )

    # ---------------------------------------------------------------- succotash engines/models
    def route_engines(self, task: str) -> Dict[str, Any]:
        """Route a task using potential-succotash engines/models catalogue."""
        if self.organs.succotash is None:
            return {"ok": False, "error": "succotash organ unavailable"}
        route = self.organs.succotash.route(task)
        out = route.to_dict() if hasattr(route, "to_dict") else dict(route)
        out["ok"] = True
        # absorb routing experience
        self._absorb(
            f"ROUTE task={task} → {out.get('rationale')}",
            kind="route",
            reward=0.7,
            meta=out,
        )
        return out

    def list_engines(self) -> List[Dict[str, Any]]:
        if self.organs.engines:
            return list(self.organs.engines)
        if self.organs.succotash:
            return self.organs.succotash.list_engines()
        return []

    def list_models(self) -> List[Dict[str, Any]]:
        if self.organs.model_catalogue:
            return list(self.organs.model_catalogue)
        if self.organs.succotash:
            return self.organs.succotash.list_models()
        return []

    def python(self, source: str, *, intent: str = "") -> MindResult:
        """Run sandboxed Python via the embedded organ; always absorbs experience."""
        t0 = time.perf_counter()
        if self.organs.python is None:
            return MindResult(
                ok=False,
                kind="python",
                model_id=self.model_id,
                error="python organ unavailable",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        if self.organs.governance is not None:
            dec = self.organs.governance.review(
                "python", intent or source[:200], model_id=self.model_id
            )
            if not dec.allowed:
                absorb = self._absorb(
                    f"PYTHON_REFUSE:{intent}\n{source[:500]}",
                    "refuse",
                    reward=0.3,
                    meta={"reasons": dec.reasons},
                )
                return MindResult(
                    ok=False,
                    kind="python",
                    model_id=self.model_id,
                    error="; ".join(dec.reasons),
                    train_pulse=absorb.get("train_pulse"),
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
        result = self.organs.python.run(source, intent=intent or "python")
        emb = None
        try:
            from auro_native_llm.corpus.embeddings import MaxEmbedder

            emb = MaxEmbedder().embed_text(result.training_text).tolist()
        except Exception:
            pass
        absorb = self._absorb(
            result.training_text,
            "python_run" if result.ok else "python_error",
            reward=0.9 if result.ok else 0.35,
            meta={"receipt": result.receipt, "ok": result.ok},
        )
        # attach embedding on last experience if trainer present
        if self.organs.trainer is not None and emb is not None and self.organs.trainer.buffer:
            try:
                self.organs.trainer.buffer[-1].embedding = emb
            except Exception:
                pass
        if self.organs.trainer is not None:
            pulse = self.organs.trainer.train_on_model(self.language, steps=1)
        else:
            pulse = absorb.get("train_pulse")
        return MindResult(
            ok=result.ok,
            kind="python",
            model_id=self.model_id,
            output=result.to_dict(),
            train_pulse=pulse if isinstance(pulse, dict) else absorb.get("train_pulse"),
            memory_wrote=absorb.get("memory_wrote", False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=result.error,
        )

    def autocycle(self, cycles: int = 3, **kw: Any) -> Dict[str, Any]:
        """Run the autonomous learn loop in-process (uses this mind if checkpoint matches)."""
        from auro_native_llm.organism.autocycle import AutocycleConfig, run_autocycle

        return run_autocycle(
            AutocycleConfig(
                model_id=self.model_id,
                cycles=cycles,
                show=bool(kw.get("show", True)),
                resume_checkpoint=kw.get("resume_checkpoint"),
                train_steps_per_cycle=int(kw.get("train_steps_per_cycle", 2)),
            )
        )

    def polyglot_roster(self) -> Dict[str, Any]:
        """Engines / transformers / orchestrators / teachers roster."""
        from auro_native_llm.polyglot.entangled import get_orchestrator

        return get_orchestrator().roster()

    def teach_domains(self, *, steps_per_lesson: int = 1) -> Dict[str, Any]:
        """Mini brains (code/research/math) + heart teach the student model now."""
        if self.organs.brains is None:
            from auro_native_llm.brain.organs import build_brain_cluster

            self.organs.brains = build_brain_cluster()
        report = self.organs.brains.teacher.teach_and_train(
            self, steps_per_lesson=steps_per_lesson
        )
        report["cuda"] = None
        try:
            from auro_native_llm.polyglot.cuda_plane import get_cuda_plane

            report["cuda"] = get_cuda_plane(refresh=True).info()
        except Exception:
            pass
        report["polyglot"] = (
            self.organs.polyglot.info() if self.organs.polyglot else None
        )
        report["brains"] = self.organs.brains.info()
        return report

    def heart_pulse(self) -> Dict[str, Any]:
        if self.organs.brains is None:
            from auro_native_llm.brain.organs import build_brain_cluster

            self.organs.brains = build_brain_cluster()
        return self.organs.brains.pulse_all()

    def portal_open(self, *, chrome_mock: bool = True) -> Dict[str, Any]:
        """Spin interior MCP portal with multi-site internet UI tools."""
        from auro_native_llm.embedded.portal import build_portal

        portal = build_portal(self, chrome_mock=chrome_mock)
        spun = portal.spin()
        self._absorb(
            f"PORTAL_OPEN tools={spun.get('tools') and len(spun.get('tools') or [])} url={spun.get('url')}",
            "portal",
            reward=0.9,
            meta={"portal": portal.manifest()},
        )
        return spun

    def multi_site(
        self,
        objective: str,
        urls: List[str],
        *,
        chrome_mock: bool = True,
    ) -> Dict[str, Any]:
        """Work many internet sites at once via fleet agents + absorb digest."""
        if self.organs.portal is None:
            self.portal_open(chrome_mock=chrome_mock)
        fleet = self.organs.fleet
        result = fleet.work_objective(objective, urls)
        # mind reason over digest
        digest = result.get("summary_for_llm") or ""
        gen = self.generate(
            f"Multi-site agent report. Objective={objective}\n{digest[:1800]}\nWrite findings + next actions.",
            max_new_tokens=80,
            temperature=0.7,
        )
        text = (gen.output or {}).get("text", "") if isinstance(gen.output, dict) else ""
        self._absorb(
            f"MULTI_SITE objective={objective}\n{digest[:1500]}\nFINDINGS:{text[:500]}",
            "multi_site",
            reward=0.92 if result.get("ok") else 0.5,
            meta={"n_urls": len(urls), "latency_ms": result.get("latency_ms")},
        )
        result["mind_findings"] = text
        result["generate_ok"] = gen.ok
        return result

    def train_entangled(
        self,
        text: str,
        *,
        steps: int = 1,
        lr: float = 2e-3,
    ) -> Dict[str, Any]:
        """Train student with polyglot council (engines+teachers) entangled."""
        from auro_native_llm.polyglot.entangled import get_orchestrator

        tok = self.language.tokenizer
        ids = tok.encode(text, add_bos=True, add_eos=True, max_length=min(96, self.config.max_seq_len))
        if len(ids) < 8:
            ids = ids + [tok.pad_id] * (8 - len(ids))
        arr = np.array([ids], dtype=np.int64)
        orch = get_orchestrator()
        history = []
        for _ in range(max(1, steps)):
            history.append(
                orch.council_train_step(
                    self.language, arr, arr, lr=lr, text_for_meaning=text[:400]
                )
            )
        # absorb teaching text
        self._absorb(
            f"ENTANGLED_TRAIN steps={steps} text={text[:500]} council={history[-1].get('council')}",
            "polyglot_train",
            reward=0.88,
            meta={"entangled": True, "n": len(history)},
        )
        return {
            "ok": True,
            "steps": len(history),
            "last": history[-1],
            "roster": orch.roster(),
            "train_steps": self.language.train_steps,
            "params": self.language.num_params,
        }

    def polyglot(self, action: str = "suite", **kw: Any) -> MindResult:
        """Run Julia + Haskell + Python + CUDA plane compute; absorb into trainer."""
        t0 = time.perf_counter()
        if self.organs.polyglot is None:
            return MindResult(
                ok=False,
                kind="polyglot",
                model_id=self.model_id,
                error="polyglot organ unavailable",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        organ = self.organs.polyglot
        if action == "health":
            res = organ.health_all()
        elif action == "spectral":
            res = organ.spectral_energy_all(kw.get("x"))
        elif action == "phi":
            res = organ.phi_powers_all(int(kw.get("n", 12)))
        elif action == "embed":
            res = organ.multi_embed_all(str(kw.get("text", "MESIE Auro")))
        elif action == "train_step":
            res = organ.accelerated_train_step(
                int(kw.get("dim", 64)), int(kw.get("batch", 8))
            )
        else:
            res = organ.suite()
        absorb = self._absorb(
            res.training_text,
            "polyglot",
            reward=0.9 if res.ok else 0.4,
            meta={"action": action, "cuda": res.meta},
        )
        if self.organs.trainer is not None and res.ok:
            pulse = self.organs.trainer.train_on_model(self.language, steps=1)
        else:
            pulse = absorb.get("train_pulse")
        return MindResult(
            ok=res.ok,
            kind="polyglot",
            model_id=self.model_id,
            output=res.to_dict(),
            train_pulse=pulse if isinstance(pulse, dict) else absorb.get("train_pulse"),
            memory_wrote=absorb.get("memory_wrote", False),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=None if res.ok else "polyglot suite partial failure",
        )

    # ---------------------------------------------------------------- identity
    def info(self) -> Dict[str, Any]:
        live = self.language.num_params
        target = self.config.parameter_target
        succ = None
        if self.organs.succotash is not None:
            try:
                succ = self.organs.succotash.registry.summary()
            except Exception:
                succ = {"present": True}
        return {
            "model_id": self.model_id,
            "tier": self.config.tier,
            "parameter_target": target,
            "num_params_live": live,
            "live_is_running_model": True,
            "target_is_architecture_label": True,
            "live_vs_target_ratio": (live / target) if target else None,
            "claim_boundary": (
                "Live params are the trained, running model. "
                "Family labels (2B/4B/8B/14B/100B) are architecture targets for scaled cores. "
                "Value is proven by CE drop + working tools + durable checkpoint, not marketing."
            ),
            "production_loop": "train → measure → save → load → work → keep learning",
            "compute_plane": "MESIE",
            "embedded_organs": self.organs.manifest(),
            "always_training": True,
            "trainer": self.organs.trainer.stats() if self.organs.trainer else {},
            "memory": self.organs.memory.stats() if self.organs.memory else {},
            "act_count": self.act_count,
            "canon_id": getattr(self.organs.canon, "canon_id", None),
            "integration": "full_embedded_organism",
            "succotash": succ,
            "engines_count": len(self.list_engines()),
            "models_count": len(self.list_models()),
            "engines_source": "https://github.com/FreddyCreates/potential-succotash",
            "python_organ": (
                self.organs.python.info() if self.organs.python is not None else None
            ),
            "capabilities": [
                "generate", "reason", "code", "work", "chrome",
                "monaco", "jupyter", "search", "mcp", "teach",
                "scripture", "constitutional", "self_train", "memory",
                "route_engines", "list_engines", "list_models", "succotash_corpus",
                "python", "autocycle", "polyglot", "julia", "haskell", "cuda_plane",
                "polyglot_roster", "train_entangled",
                "engines", "transformers", "orchestrators", "teachers",
                "teach_domains", "heart_pulse", "mini_brain", "mini_heart",
                "chaos_cuda", "code", "research", "math",
                "portal_open", "multi_site", "interior_mcp", "multi_site_agents",
                # installed mesie package (pip install mesie / mesie[ml] / mesie[intelligence])
                "mesie_spectral", "mesie_transformers", "mesie_embeddings",
                "mesie_helix", "mesie_intelligence", "mesie_connectome",
                "mesie_pretraining", "mesie_miniverse", "mesie_validation",
                "mesie_match", "mesie_psd_fas", "sdk_runtime_inject",
                # GHOST pillars + hybrid MESIE node + receipts
                "ghost", "ghost_agents", "ghost_mesie_node", "ghost_receipts",
                "ghost_policy", "ghost_hybrid_llm", "ghost_haunt_detector",
                # Google virtual envelope + collab
                "google_envelope", "google_chrome", "google_mail", "google_drive",
                "google_search", "google_calendar", "google_sites", "collab_workspace",
                # dual organism
                "python_ai", "julia_brain", "virtual_physics_cores", "distributed_think",
                "power_stack", "physics_engines", "economic_engines", "coupled_algorithms",
            ],
            "brains": (
                self.organs.brains.info() if self.organs.brains is not None else None
            ),
            "mesie_runtime": (
                self.mesie_runtime.health()  # type: ignore[attr-defined]
                if getattr(self, "mesie_runtime", None) is not None
                else None
            ),
            "ghost": bool(getattr(self, "ghost", None)),
        }
