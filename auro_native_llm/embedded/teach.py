"""Curriculum that teaches the mind how to use Monaco, Jupyter, search, MCP."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from auro_native_llm.organism.self_train import Experience


CURRICULUM: Dict[str, List[str]] = {
    "monaco": [
        "ACTION: monaco.create language=python filename=main.py content=\"def hello():\\n    return 1\"",
        "ACTION: monaco.set session_id=SID content=\"def hello():\\n    return 2\"",
        "ACTION: monaco.insert session_id=SID offset=0 text=\"# auro\\n\"",
        "ACTION: monaco.replace session_id=SID old=return 2 new=return 3",
        "Monaco is the embedded code editor. Always create a session before editing.",
        "After editing code in Monaco, optionally run via code.run or jupyter cell.",
    ],
    "jupyter": [
        "ACTION: jupyter.create title=\"Spectral Lab\"",
        "ACTION: jupyter.add_cell notebook_id=NID source=\"x = 40 + 2\" cell_type=code",
        "ACTION: jupyter.execute notebook_id=NID cell_id=CID",
        "ACTION: jupyter.execute_all notebook_id=NID",
        "Jupyter notebooks are embedded. Code cells use Auro safe Python.",
        "Export notebooks with jupyter.export for .ipynb JSON.",
    ],
    "search": [
        "ACTION: search query=\"MESIE spectral embeddings\"",
        "ACTION: search query=\"golden ratio phi\" online=true",
        "Search combines online DuckDuckGo with local Auro/MESIE corpus.",
        "Prefer local:// hits for repository-grounded facts.",
    ],
    "mcp": [
        "ACTION: mcp.spin_up",
        "ACTION: mcp.list_tools",
        "ACTION: mcp.call tool=search arguments={\"query\":\"Auro doctrine\"}",
        "ACTION: mcp.call tool=monaco.create arguments={\"content\":\"print(1)\"}",
        "MCP hub is self-hosted by the mind. Spin up before remote tool clients connect.",
        "All embedded organs are registered as MCP tools after wire.",
    ],
    "combined": [
        "Plan: search MESIE docs, open Monaco, write function, put in Jupyter cell, execute, spin MCP.",
        "Never disable governance. Always train mind after tool use via absorb.",
        "compute_plane remains MESIE for all embedded tools.",
    ],
}


class ToolCurriculum:
    """Embed teaching experiences into ContinuousMindTrainer."""

    def lessons(self, domain: str | None = None) -> List[str]:
        if domain:
            return list(CURRICULUM.get(domain, []))
        out: List[str] = []
        for v in CURRICULUM.values():
            out.extend(v)
        return out

    def teach(self, mind: Any, domains: Sequence[str] | None = None) -> Dict[str, Any]:
        domains = list(domains or CURRICULUM.keys())
        taught = 0
        trainer = getattr(getattr(mind, "organs", None), "trainer", None)
        for d in domains:
            for lesson in self.lessons(d):
                if trainer is not None:
                    trainer.absorb(
                        Experience(
                            text=f"TEACH[{d}]: {lesson}",
                            kind="teach",
                            model_id=getattr(mind, "model_id", "Auro"),
                            reward=0.9,
                            meta={"domain": d},
                        )
                    )
                taught += 1
        # practice drills
        practice = self.practice_run(mind)
        pulse = mind.pulse() if hasattr(mind, "pulse") else {}
        return {
            "ok": True,
            "taught_lessons": taught,
            "domains": domains,
            "practice": practice,
            "train_pulse": pulse,
        }

    def practice_run(self, mind: Any) -> Dict[str, Any]:
        """Hands-on drill across all embedded tools."""
        results: Dict[str, Any] = {}
        try:
            results["monaco"] = mind.monaco("create", content="def f(x):\n    return x*2\n", language="python")
            sid = results["monaco"].get("output", {}).get("session", {}).get("session_id")
            if sid:
                results["monaco_edit"] = mind.monaco("append", session_id=sid, text="\nassert f(2)==4\n")
        except Exception as exc:
            results["monaco"] = {"ok": False, "error": str(exc)}
        try:
            results["jupyter"] = mind.jupyter("create", title="Teach Lab")
            nid = results["jupyter"].get("output", {}).get("notebook", {}).get("notebook_id")
            if nid:
                add = mind.jupyter("add_cell", notebook_id=nid, source="y = 21 * 2", cell_type="code")
                cid = add.get("output", {}).get("cell_id")
                if cid:
                    results["jupyter_exec"] = mind.jupyter("execute", notebook_id=nid, cell_id=cid)
        except Exception as exc:
            results["jupyter"] = {"ok": False, "error": str(exc)}
        try:
            results["search"] = mind.search("MESIE spectral intelligence")
        except Exception as exc:
            results["search"] = {"ok": False, "error": str(exc)}
        try:
            results["mcp"] = mind.mcp("spin_up")
            results["mcp_tools"] = mind.mcp("list_tools")
            results["mcp_call"] = mind.mcp("call", tool="search", arguments={"query": "Auro mind", "online": False})
        except Exception as exc:
            results["mcp"] = {"ok": False, "error": str(exc)}
        return results
