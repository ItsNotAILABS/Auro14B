"""MatureHIM: preserves HIM's organism loop while strengthening language control."""
from __future__ import annotations

from typing import Any

from auro_native_llm.him.being import HIM
from auro_native_llm.language.spatial_lexicon import ExecutiveComposer, SpatialLexicon
from auro_native_llm.model.usable import is_usable_text


class MatureHIM(HIM):
    """HIM with lexical memory, evidence-aware composition, and self-revision."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lexicon = SpatialLexicon()
        self.composer = ExecutiveComposer(self.lexicon)
        self.lexicon.ingest(self.whoami().get("doctrine", ""), source="identity")

    def sense(self, goal: str) -> dict[str, Any]:
        base = super().sense(goal)
        context = self.colony.context.retrieve(goal, top_k=12, token_cap=8000)
        self.lexicon.ingest(goal, source="goal")
        if context:
            self.lexicon.ingest(context, source="retrieved-context")
        base["spatial_associations"] = self.lexicon.associations(goal, top_k=16)
        base["lexicon"] = self.lexicon.manifest()
        base["context_preview"] = context[:1500]
        return base

    def reflect(self, goal: str, observations: list[dict[str, Any]], artifacts: list[str]) -> dict[str, Any]:
        base = super().reflect(goal, observations, artifacts)
        candidates = [str(base.get("answer") or ""), *[str(x) for x in artifacts if x]]
        evidence = [self.colony.context.retrieve(goal, top_k=12, token_cap=10000)]
        require_uncertainty = any(
            marker in goal.casefold()
            for marker in ("cannot inspect", "cannot verify", "unknown", "production-ready", "evidence required")
        )
        composed = self.composer.compose(
            goal,
            candidates,
            evidence=evidence,
            require_uncertainty=require_uncertainty,
        )
        text = composed["text"]
        if is_usable_text(text, min_len=40):
            base["answer"] = text
            base["text"] = text
            base["method"] = "mature_him_executive_compose"
            base["language_receipt"] = composed
            self.colony.context.ingest(
                f"MATURE RESPONSE:\n{text}",
                kind="artifact",
                meta={"method": base["method"], "score": composed["score"]},
            )
        return base

    def develop(self, task: str, *, cycles: int = 4) -> dict[str, Any]:
        """Generate, read back, challenge, revise, and preserve one work artifact."""
        stages = []
        current = task
        prompts = (
            "Generate a substantial first artifact. Preserve evidence boundaries and unresolved questions.",
            "Read the artifact you just generated. Identify omissions, repetition, unsupported claims, and weak structure.",
            "Red-team the artifact. Find counterclaims, failure modes, and the minimum evidence needed to trust it.",
            "Revise the artifact using the readback and red-team findings. Produce the strongest final version without hiding uncertainty.",
        )
        for index, instruction in enumerate(prompts[:max(1, cycles)], 1):
            report = self.run(f"AUTONOMOUS DEVELOPMENT STAGE {index}. {instruction}\n\nTASK:\n{task}\n\nCURRENT MATERIAL:\n{current}", max_actions=5)
            current = str(report.get("answer") or report.get("text") or current)
            self.lexicon.ingest(current, source=f"development-stage-{index}")
            self.colony.context.ingest(current, kind="artifact", meta={"stage": index})
            stages.append({"stage": index, "instruction": instruction, "report": report})
        return {
            "schema": "auro.him.development.v1",
            "task": task,
            "stages": stages,
            "final": current,
            "lexicon": self.lexicon.manifest(),
        }


def awaken_mature_him(mind: Any = None, *, n_germs: int = 40, context_tokens: int = 500_000) -> MatureHIM:
    return MatureHIM(mind, n_germs=n_germs, context_tokens=context_tokens)
