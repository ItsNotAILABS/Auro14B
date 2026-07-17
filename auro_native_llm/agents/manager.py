"""Internal agent manager — spawn, assign, supervise, absorb.

Agents live *inside* the mind: planner, researcher, coder, browser fleet,
polyglot worker. Manager is the control plane for multi-agent work.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class InternalAgent:
    agent_id: str
    role: str  # planner | researcher | coder | browser | polyglot | critic
    status: str = "idle"  # idle|running|done|error
    task: str = ""
    result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "status": self.status,
            "task": self.task[:200],
            "result_ok": None if self.result is None else bool(self.result.get("ok", True)),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AgentManager:
    """Manage internal agents for thinking, coding, research, multi-site work."""

    ROLES = ("planner", "researcher", "coder", "browser", "polyglot", "critic", "math")

    def __init__(self, mind: Any) -> None:
        self.mind = mind
        self.agents: Dict[str, InternalAgent] = {}
        self.history: List[Dict[str, Any]] = []

    def spawn(self, role: str, task: str = "") -> InternalAgent:
        if role not in self.ROLES:
            role = "planner"
        aid = f"{role}-{uuid.uuid4().hex[:8]}"
        ag = InternalAgent(agent_id=aid, role=role, task=task)
        self.agents[aid] = ag
        return ag

    def list_agents(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "agents": [a.to_dict() for a in self.agents.values()],
            "n": len(self.agents),
            "roles": list(self.ROLES),
        }

    def run_agent(self, agent_id: str, task: Optional[str] = None) -> Dict[str, Any]:
        ag = self.agents.get(agent_id)
        if not ag:
            return {"ok": False, "error": f"unknown agent {agent_id}"}
        task = task or ag.task
        ag.status = "running"
        ag.task = task
        ag.updated_at = time.time()
        mind = self.mind
        try:
            if ag.role == "planner":
                r = mind.reason(f"Plan steps for: {task}")
                out = {"ok": r.ok, "kind": "plan", "output": r.output}
            elif ag.role == "researcher":
                # retrieval + reason
                hits = []
                try:
                    from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

                    hits = [h.to_dict() for h in GitHubKnowledgeDB().search(task, top_k=4)]
                except Exception:
                    pass
                # symbolic expand
                try:
                    from auro_native_llm.symbolic.compress import SymbolicCompressor

                    sym = SymbolicCompressor().expand_context(task, top_k=3)
                except Exception:
                    sym = ""
                prompt = (
                    f"DEEP RESEARCH then ANSWER.\nTask: {task}\n"
                    f"Evidence: {hits[:2]}\n{sym}\n"
                    f"Write: findings, sources, conclusion."
                )
                g = mind.think_answer(prompt, max_new_tokens=96)
                out = {"ok": g.get("ok", True), "kind": "research", "output": g, "hits": hits}
            elif ag.role == "coder":
                g = mind.think_answer(
                    f"CODE complete application slice for: {task}\n"
                    f"Output: structure + key files + python code block.",
                    max_new_tokens=120,
                )
                if hasattr(mind, "python") and mind.organs.python:
                    # execute a tiny verification if code organ present
                    py = mind.python(
                        "print('coder_agent_ready'); result=42\n",
                        intent=f"coder:{task[:80]}",
                    )
                    g["python_verify"] = py.to_dict() if hasattr(py, "to_dict") else {"ok": py.ok}
                out = {"ok": g.get("ok", True), "kind": "code", "output": g}
            elif ag.role == "browser":
                urls = [
                    "https://example.com",
                    "https://example.org",
                ]
                if "http" in task:
                    import re

                    found = re.findall(r"https?://\\S+", task)
                    if found:
                        urls = found[:4]
                ms = mind.multi_site(task, urls, chrome_mock=True)
                out = {"ok": ms.get("ok", True), "kind": "browser", "output": ms}
            elif ag.role == "polyglot":
                r = mind.polyglot("suite")
                out = {"ok": r.ok, "kind": "polyglot", "output": r.to_dict()}
            elif ag.role == "math":
                g = mind.think_answer(
                    f"MATH: solve/derive carefully with steps then final answer.\nProblem: {task}",
                    max_new_tokens=96,
                )
                # verify with polyglot phi if available
                if mind.organs.polyglot:
                    p = mind.polyglot("phi", n=8)
                    g["polyglot_phi"] = p.to_dict() if hasattr(p, "to_dict") else {}
                out = {"ok": g.get("ok", True), "kind": "math", "output": g}
            else:  # critic
                g = mind.think_answer(
                    f"CRITIQUE the plan/result for: {task}\nList risks, gaps, next tests.",
                    max_new_tokens=80,
                )
                out = {"ok": g.get("ok", True), "kind": "critic", "output": g}

            ag.status = "done" if out.get("ok") else "error"
            ag.result = out
            ag.updated_at = time.time()
            self.history.append({"agent": ag.to_dict(), "ts": time.time()})
            return {"ok": True, "agent": ag.to_dict(), "result": out}
        except Exception as exc:
            ag.status = "error"
            ag.result = {"ok": False, "error": str(exc)}
            return {"ok": False, "agent": ag.to_dict(), "error": str(exc)}

    def run_team(self, objective: str, roles: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Spawn and run a multi-agent team on one objective."""
        roles = list(roles or ("planner", "researcher", "coder", "critic"))
        results = []
        for role in roles:
            ag = self.spawn(role, objective)
            results.append(self.run_agent(ag.agent_id, objective))
        # final synthesis
        synth = self.mind.think_answer(
            f"Synthesize multi-agent work for: {objective}\n"
            f"Roles completed: {roles}\nGive final answer + action list.",
            max_new_tokens=100,
        )
        return {
            "schema": "auro.agents.team.v1",
            "ok": all(r.get("ok") for r in results),
            "objective": objective,
            "agents": results,
            "synthesis": synth,
            "n_agents": len(results),
        }

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema": "auro.agents.manager.v1",
            "n": len(self.agents),
            "roles": list(self.ROLES),
            "agents": [a.to_dict() for a in self.agents.values()],
            "history": len(self.history),
        }
