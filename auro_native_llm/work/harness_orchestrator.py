"""Recursive orchestration for AURO independent harnesses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from .harness import HarnessState, HarnessTask, IndependentHarnessFabric


DEFAULT_ROLE_INSTRUCTIONS = {
    "planner": "Decompose objectives, sequence dependencies, and identify completion criteria.",
    "researcher": "Gather evidence and context; report sources, uncertainty, and open questions.",
    "coder": "Implement, inspect, test, and repair code or technical artifacts.",
    "reviewer": "Review outputs for correctness, regressions, unsupported claims, and missing work.",
    "operator": "Execute bounded tool work and produce concrete completion receipts.",
}


@dataclass(frozen=True)
class FanoutPlan:
    objective: str
    subproblems: list[dict[str, Any]]
    planner_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "auro.harness.fanout-plan.v3",
            "objective": self.objective,
            "subproblems": self.subproblems,
            "planner_text": self.planner_text,
        }


class HarnessOrchestrator:
    """Plan, fan out, run, and rejoin complete harness instances."""

    def __init__(self, fabric: IndependentHarnessFabric | None = None) -> None:
        self.fabric = fabric or IndependentHarnessFabric()

    def plan(self, objective: str, *, model_id: str = "Auro-2B", max_children: int = 6) -> FanoutPlan:
        max_children = max(1, min(int(max_children), self.fabric.max_children))
        text = ""
        try:
            from .agent import WorkAgent
            agent = WorkAgent(model_id=model_id, lite=True, use_scripture=True, max_tool_steps=2)
            prompt = (
                "Decompose the following objective into independent work packages that can run in parallel. "
                f"Return JSON only with key subproblems, max {max_children}. Each item must have objective, role, and completion_criteria.\nOBJECTIVE: {objective}"
            )
            text = agent.generate_text(prompt, mode="reason", max_new_tokens=220, temperature=0.45)
        except Exception:
            text = ""
        items = self._parse_subproblems(text, objective, max_children)
        return FanoutPlan(objective=objective, subproblems=items, planner_text=text[:4000])

    def _parse_subproblems(self, text: str, objective: str, max_children: int) -> list[dict[str, Any]]:
        raw = text.strip()
        candidates = []
        if "{" in raw and "}" in raw:
            try:
                obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
                candidates = obj.get("subproblems", []) if isinstance(obj, dict) else []
            except Exception:
                candidates = []
        cleaned = []
        for item in candidates:
            if not isinstance(item, dict) or not str(item.get("objective", "")).strip():
                continue
            role = str(item.get("role") or self._infer_role(str(item["objective"])))
            cleaned.append({
                "objective": str(item["objective"]).strip(),
                "role": role,
                "completion_criteria": str(item.get("completion_criteria") or "Produce a concrete, reviewable result."),
            })
            if len(cleaned) >= max_children:
                break
        if cleaned:
            return cleaned
        clauses = [x.strip(" -\t") for x in re.split(r"[;\n]+", objective) if x.strip()]
        if len(clauses) <= 1:
            clauses = [
                f"Plan and define the work for: {objective}",
                f"Implement or execute the core deliverable for: {objective}",
                f"Review, test, and identify remaining issues for: {objective}",
            ]
        return [
            {"objective": clause, "role": self._infer_role(clause), "completion_criteria": "Produce a concrete, reviewable result."}
            for clause in clauses[:max_children]
        ]

    @staticmethod
    def _infer_role(objective: str) -> str:
        low = objective.lower()
        if any(x in low for x in ("code", "implement", "build", "fix", "test")):
            return "coder"
        if any(x in low for x in ("research", "find", "compare", "evidence")):
            return "researcher"
        if any(x in low for x in ("review", "audit", "verify", "critique")):
            return "reviewer"
        return "planner"

    def fan_out_plan(self, parent_id: str, plan: FanoutPlan) -> list[HarnessState]:
        parent = self.fabric.store.load(parent_id)
        children = []
        for item in plan.subproblems:
            role = str(item.get("role") or "planner")
            child = self.fabric.create_harness(
                str(item["objective"]),
                parent_id=parent_id,
                model_id=parent.model_id,
                agent_roster=[role, "reviewer"] if role != "reviewer" else ["reviewer"],
                tasks=[{
                    "objective": f"ROLE={role}. {DEFAULT_ROLE_INSTRUCTIONS.get(role, '')}\nOBJECTIVE={item['objective']}\nCOMPLETION={item.get('completion_criteria', '')}",
                    "max_attempts": 3,
                }],
            )
            for task in child.tasks.values():
                task.assigned_agent = role
            self.fabric.store.save(child)
            children.append(child)
        self.fabric.store.append_event(parent_id, "fanout_plan_applied", {"child_ids": [c.id for c in children], "plan": plan.to_dict()})
        return children

    def orchestrate(self, objective: str, *, model_id: str = "Auro-2B", max_children: int = 6) -> dict[str, Any]:
        parent = self.fabric.create_harness(objective, model_id=model_id, tasks=[])
        # Parent task is a join task; children do independent work first.
        parent.tasks.clear()
        plan = self.plan(objective, model_id=model_id, max_children=max_children)
        children = self.fan_out_plan(parent.id, plan)
        join = self.fabric.add_task(parent.id, "Aggregate and review all child harness results.")
        state = self.fabric.store.load(parent.id)
        join.state = "waiting_children"
        self.fabric.store.save(state)
        return {
            "schema": "auro.harness.orchestration.v3",
            "parent": self.fabric.store.load(parent.id).to_dict(),
            "plan": plan.to_dict(),
            "children": [c.to_dict() for c in children],
        }

    def advance_tree(self, parent_id: str, *, worker_id: str = "orchestrator", cycles_per_child: int = 8) -> dict[str, Any]:
        parent = self.fabric.store.load(parent_id)
        child_runs = []
        for child_id in parent.child_ids:
            child = self.fabric.store.load(child_id)
            if child.state == "active":
                child_runs.append(self.fabric.run_until_blocked(child_id, worker_id=f"{worker_id}:{child_id[-8:]}", max_cycles=cycles_per_child))
        parent = self.fabric.store.load(parent_id)
        all_terminal = all(self.fabric.store.load(cid).state in {"completed", "failed", "cancelled"} for cid in parent.child_ids)
        if all_terminal:
            for task in parent.tasks.values():
                if task.state == "waiting_children":
                    aggregate = self.fabric.aggregate(parent_id)
                    task.state = "completed"
                    task.result = {"ok": True, "aggregate": aggregate}
            parent.state = "completed"
            parent.completed_tasks = sum(t.state == "completed" for t in parent.tasks.values())
            parent.failed_tasks = sum(t.state == "failed" for t in parent.tasks.values())
            parent.final_summary = self.fabric.aggregate(parent_id)["summary"]
            self.fabric.store.save(parent)
            self.fabric.store.append_event(parent_id, "children_rejoined", {"summary": parent.final_summary})
        return {
            "schema": "auro.harness.tree-advance.v3",
            "parent": self.fabric.store.load(parent_id).to_dict(),
            "child_runs": child_runs,
            "aggregate": self.fabric.aggregate(parent_id),
        }


__all__ = ["FanoutPlan", "HarnessOrchestrator", "DEFAULT_ROLE_INSTRUCTIONS"]
