"""MatureHIM: HIM with persistent lexical space and receipt-preserving revision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from auro_native_llm.him.being import HIM
from auro_native_llm.language.spatial_lexicon import LanguageEngine
from auro_native_llm.model.usable import is_usable_text


class MatureHIM(HIM):
    """Preserve the organism loop while strengthening language control."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.language = LanguageEngine()
        self.lexicon = self.language.lexicon
        self.composer = self.language.composer
        self._last_language_receipt: dict[str, Any] | None = None
        self.language.observe(self.whoami().get("doctrine", ""), source="identity", metadata={"session_id": self.session_id})

    def sense(self, goal: str) -> dict[str, Any]:
        base = super().sense(goal)
        context = self.colony.context.retrieve(goal, top_k=12, token_cap=8000)
        goal_receipt = self.language.observe(goal, source=f"goal:{self.session_id}", metadata={"phase": "sense"})
        context_receipt = self.language.observe(context, source=f"context:{self.session_id}", metadata={"phase": "sense"}) if context else None
        grab = self.language.grab(goal, top_k=10)
        base["language"] = {"goal_ingestion": goal_receipt, "context_ingestion": context_receipt, "associations": grab["associations"], "grabs": grab["spans"], "lexicon": self.lexicon.manifest()}
        base["context_preview"] = context[:2000]
        self._last_language_receipt = base["language"]
        return base

    def plan(self, goal: str, sense: dict[str, Any]) -> dict[str, Any]:
        plan = super().plan(goal, sense)
        plan["language_plan"] = self.language.plan(goal, branches=8)
        self._last_language_receipt = plan["language_plan"]
        return plan

    def reflect(self, goal: str, observations: list[dict[str, Any]], artifacts: list[str]) -> dict[str, Any]:
        base = super().reflect(goal, observations, artifacts)
        retrieved = self.colony.context.retrieve(goal, top_k=12, token_cap=10000)
        candidates = [str(base.get("answer") or ""), *[str(item) for item in artifacts if item]]
        require_uncertainty = any(marker in goal.casefold() for marker in ("cannot verify", "unknown", "production-ready", "evidence required", "beat kimi", "outperform", "superiority"))
        composed = self.language.compose(goal, candidates, evidence=[retrieved] if retrieved else [], require_uncertainty=require_uncertainty)
        text = composed["text"]
        if is_usable_text(text, min_len=40):
            base.update(answer=text, text=text, method="mature_him_language_engine_v1", language_receipt=composed)
            self._last_language_receipt = composed
            self.colony.context.ingest(f"MATURE RESPONSE:\n{text}", kind="artifact", meta={"method": base["method"], "lexicon_sha256": composed["lexicon"]["sha256"], "score": composed["score"]})
        return base

    def run(self, goal: str, *, max_actions: int = 5) -> dict[str, Any]:
        """Run HIM and preserve language evidence dropped by the parent report."""
        report = super().run(goal, max_actions=max_actions)
        report["language_receipt"] = self._last_language_receipt
        report["language_engine"] = self.language.manifest()
        saved = report.get("saved")
        if saved:
            try:
                Path(saved).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            except Exception:
                pass
        return report

    def develop(self, task: str, *, cycles: int = 4) -> dict[str, Any]:
        """Generate, read, red-team, rewrite, and preserve one trajectory."""
        prompts = (
            "Generate a substantial first artifact. Lead with the answer and preserve evidence boundaries.",
            "Read the draft as a strict editor. Identify omissions, repetition, unsupported claims, and doctrine drift.",
            "Red-team the draft. Produce counterclaims, failure modes, benchmark requirements, and minimum trust evidence.",
            "Rewrite using the draft, editorial readback, and red-team findings. Do not hide uncertainty.",
        )
        stages: list[dict[str, Any]] = []
        draft = ""
        critiques: list[str] = []
        for index, instruction in enumerate(prompts[:max(1, min(cycles, len(prompts)))], 1):
            material = draft
            if critiques:
                material += "\n\nCRITIQUES:\n" + "\n\n".join(critiques)
            prompt = f"AUTONOMOUS DEVELOPMENT STAGE {index}. {instruction}\n\nTASK:\n{task}\n\nCURRENT MATERIAL:\n{material or '[none yet]'}"
            report = self.run(prompt, max_actions=5)
            output = str(report.get("answer") or report.get("text") or "")
            self.language.observe(output, source=f"development:{index}:{self.session_id}", metadata={"stage": index})
            self.colony.context.ingest(output, kind="artifact", meta={"stage": index})
            if index == 1:
                draft = output
            elif index in (2, 3):
                critiques.append(output)
            else:
                draft = output or draft
            stages.append({"stage": index, "instruction": instruction, "input_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(), "report": report})
        revision = self.language.revise(task, draft, critiques, evidence=[self.colony.context.retrieve(task, top_k=16, token_cap=14000)], require_uncertainty=True)
        final = revision["text"] if is_usable_text(revision["text"], min_len=80) else draft
        self._last_language_receipt = revision
        return {"schema": "auro.him.development.v2", "task": task, "stages": stages, "final": final, "revision_receipt": revision, "lexicon": self.lexicon.manifest(), "weight_update_performed": False}


def awaken_mature_him(mind: Any = None, *, n_germs: int = 40, context_tokens: int = 500_000) -> MatureHIM:
    return MatureHIM(mind, n_germs=n_germs, context_tokens=context_tokens)
