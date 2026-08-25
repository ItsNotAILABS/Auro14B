"""Self-written, versioned skill artifacts distilled from harness outcomes.

Skills are data/procedure artifacts, never silently executable Python. They are
selected by objective similarity and outcome score, and may be revised or
retired by later harnesses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import time

from .harness import HarnessState, HarnessStore


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:64] or "skill"


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass
class SkillArtifact:
    id: str
    name: str
    version: int
    objective_pattern: str
    procedure: list[str]
    completion_checks: list[str]
    evidence: list[str]
    source_harness_id: str
    parent_skill_id: str | None = None
    outcome_score: float = 0.0
    uses: int = 0
    successes: int = 0
    state: str = "candidate"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = "auro.harness.skill.v1"
        value["sha256"] = _hash(value)
        return value


class HarnessSkillForge:
    def __init__(self, store: HarnessStore, root: str | Path | None = None) -> None:
        self.store = store
        self.root = Path(root) if root is not None else store.root / "_skills"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, skill_id: str) -> Path:
        return self.root / f"{skill_id}.json"

    def list(self, *, include_retired: bool = False) -> list[SkillArtifact]:
        skills = []
        for path in self.root.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                fields = SkillArtifact.__dataclass_fields__
                skill = SkillArtifact(**{k: v for k, v in raw.items() if k in fields})
                if include_retired or skill.state != "retired":
                    skills.append(skill)
            except Exception:
                continue
        return sorted(skills, key=lambda x: (x.outcome_score, x.version, x.updated_at), reverse=True)

    def save(self, skill: SkillArtifact) -> SkillArtifact:
        skill.updated_at = time.time()
        self._path(skill.id).write_text(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return skill

    def distill(self, harness_id: str) -> SkillArtifact | None:
        state = self.store.load(harness_id)
        completed = [task for task in state.tasks.values() if task.state == "completed" and task.result]
        if not completed:
            return None
        procedure = []
        evidence = []
        checks = []
        for task in completed:
            procedure.append(f"Execute work package: {task.objective}")
            result = task.result or {}
            summary = str(result.get("final_summary") or result.get("summary") or "")
            if summary:
                evidence.append(summary[:800])
            checks.append(f"Confirm task {task.id} completed with an explicit result receipt.")
        child_states = [self.store.load(cid) for cid in state.child_ids if self.store.exists(cid)]
        total = max(1, len(completed) + len(child_states))
        successful_children = sum(child.state == "completed" for child in child_states)
        score = min(1.0, (len(completed) + successful_children) / total)
        name = _slug(state.objective)
        prior = [skill for skill in self.list(include_retired=True) if skill.name == name]
        parent = prior[0] if prior else None
        version = (parent.version + 1) if parent else 1
        skill_id = f"skill_{name}_v{version}"
        skill = SkillArtifact(
            id=skill_id,
            name=name,
            version=version,
            objective_pattern=state.objective,
            procedure=procedure,
            completion_checks=checks,
            evidence=evidence,
            source_harness_id=state.id,
            parent_skill_id=parent.id if parent else None,
            outcome_score=score,
            state="active" if score >= 0.75 else "candidate",
        )
        if parent and skill.outcome_score >= parent.outcome_score:
            parent.state = "retired"
            self.save(parent)
        self.save(skill)
        self.store.append_event(harness_id, "skill_distilled", {"skill_id": skill.id, "version": skill.version, "outcome_score": score})
        return skill

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {x for x in re.findall(r"[a-z0-9_\-]{3,}", text.lower())}

    def select(self, objective: str, *, limit: int = 4) -> list[SkillArtifact]:
        query = self._terms(objective)
        ranked = []
        for skill in self.list():
            terms = self._terms(skill.objective_pattern)
            overlap = len(query & terms) / max(1, len(query | terms))
            score = 0.65 * overlap + 0.35 * skill.outcome_score
            ranked.append((score, skill))
        return [skill for score, skill in sorted(ranked, key=lambda x: x[0], reverse=True)[: max(1, int(limit))] if score > 0.05]

    def record_use(self, skill_id: str, *, success: bool) -> SkillArtifact:
        raw = json.loads(self._path(skill_id).read_text(encoding="utf-8"))
        fields = SkillArtifact.__dataclass_fields__
        skill = SkillArtifact(**{k: v for k, v in raw.items() if k in fields})
        skill.uses += 1
        skill.successes += int(bool(success))
        observed = skill.successes / max(1, skill.uses)
        skill.outcome_score = 0.7 * skill.outcome_score + 0.3 * observed
        if skill.uses >= 5 and skill.outcome_score < 0.35:
            skill.state = "retired"
        return self.save(skill)


__all__ = ["SkillArtifact", "HarnessSkillForge"]
