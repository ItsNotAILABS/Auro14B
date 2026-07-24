"""Interior MCP portal — model-embedded tool surface for app/internet control.

Gives the mind an *interior* portal (not external SaaS) to:
  - list/call MCP tools
  - control Chrome / multi-site fleets
  - code / research / math teachers
  - polyglot engines
  - ChaosCUDA plane

UI: served at /portal alongside the main mind UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auro_native_llm.agents.multisite import MultiSiteFleet
from auro_native_llm.embedded.mcp_hub import MCPHub, MCPOrgan, MCPTool


@dataclass
class InteriorPortal:
    """Alpha interior MCP portal owned by the mind."""

    mcp: MCPOrgan
    fleet: MultiSiteFleet
    mind_ref: Any = None
    spun: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def wire_alpha_tools(self) -> Dict[str, Any]:
        """Register multi-site + control tools into the interior MCP hub."""
        hub = self.mcp.hub
        fleet = self.fleet

        hub.register(
            MCPTool(
                "portal.manifest",
                "Interior portal + fleet manifest",
                lambda a: self.manifest(),
            )
        )
        hub.register(
            MCPTool(
                "sites.spawn",
                "Open a new site agent tab",
                lambda a: fleet.spawn(a.get("url", "https://example.com"), a.get("site_id")),
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "site_id": {"type": "string"},
                    },
                },
            )
        )
        hub.register(
            MCPTool(
                "sites.open_many",
                "Open many internet sites in parallel",
                lambda a: fleet.open_many(a.get("urls") or ["https://example.com"]).to_dict(),
                {
                    "type": "object",
                    "properties": {
                        "urls": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["urls"],
                },
            )
        )
        hub.register(
            MCPTool(
                "sites.read_all",
                "Read DOM from all open site agents concurrently",
                lambda a: fleet.read_all().to_dict(),
            )
        )
        hub.register(
            MCPTool(
                "sites.act_all",
                "Run the same UI action on all open sites (click/type/eval)",
                lambda a: fleet.act_all(
                    a.get("action", "eval"),
                    x=a.get("x", 10),
                    y=a.get("y", 10),
                    text=a.get("text", ""),
                    js=a.get("js", "document.title"),
                    url=a.get("url"),
                ).to_dict(),
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "text": {"type": "string"},
                        "js": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                },
            )
        )
        hub.register(
            MCPTool(
                "sites.work",
                "Multi-site objective: open many URLs, read all DOMs, return LLM digest",
                lambda a: fleet.work_objective(
                    a.get("objective", "survey sites"),
                    a.get("urls") or ["https://example.com", "https://example.org"],
                ),
                {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "urls": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["objective", "urls"],
                },
            )
        )
        hub.register(
            MCPTool(
                "sites.list",
                "List active site agents",
                lambda a: fleet.list_sites(),
            )
        )
        hub.register(
            MCPTool(
                "sites.close",
                "Close a site agent",
                lambda a: fleet.close(a.get("site_id", "")),
                {"type": "object", "properties": {"site_id": {"type": "string"}}},
            )
        )

        # polyglot / chaos / brains if mind attached
        mind = self.mind_ref
        if mind is not None:
            hub.register(
                MCPTool(
                    "polyglot.suite",
                    "Run Python+Julia+Haskell+ChaosCUDA suite",
                    lambda a: mind.polyglot("suite").to_dict()
                    if hasattr(mind, "polyglot")
                    else {"ok": False},
                )
            )
            hub.register(
                MCPTool(
                    "brains.teach",
                    "Mini brains teach code/research/math curriculum",
                    lambda a: mind.teach_domains(
                        steps_per_lesson=int(a.get("steps_per_lesson", 1))
                    )
                    if hasattr(mind, "teach_domains")
                    else {"ok": False},
                )
            )
            hub.register(
                MCPTool(
                    "train.entangled",
                    "Entangled polyglot council train step on text",
                    lambda a: mind.train_entangled(
                        a.get("text", "MESIE teach"),
                        steps=int(a.get("steps", 1)),
                    )
                    if hasattr(mind, "train_entangled")
                    else {"ok": False},
                )
            )
            hub.register(
                MCPTool(
                    "python.run",
                    "Run sandboxed Python via interior organ",
                    lambda a: mind.python(
                        a.get("source", "print(1)"),
                        intent=a.get("intent", "portal"),
                    ).to_dict()
                    if hasattr(mind, "python")
                    else {"ok": False},
                )
            )
            # Google virtual envelope + collab
            hub.register(
                MCPTool(
                    "google.status",
                    "AI Google sandbox envelope health (Chrome/Mail/Drive/Collab)",
                    lambda a: mind.google("status") if hasattr(mind, "google") else {"ok": False},
                )
            )
            hub.register(
                MCPTool(
                    "google.act",
                    "Act on Google envelope surface: chrome|mail|drive|search|collab|calendar|sites",
                    lambda a: mind.google(
                        a.get("surface", "search"),
                        a.get("action", "list"),
                        **{k: v for k, v in a.items() if k not in ("surface", "action")},
                    )
                    if hasattr(mind, "google")
                    else {"ok": False},
                    {
                        "type": "object",
                        "properties": {
                            "surface": {"type": "string"},
                            "action": {"type": "string"},
                            "query": {"type": "string"},
                            "url": {"type": "string"},
                            "name": {"type": "string"},
                            "text": {"type": "string"},
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["surface"],
                    },
                )
            )
            hub.register(
                MCPTool(
                    "collab.post",
                    "Shared user+AI project chat (AI replies in-thread)",
                    lambda a: mind.collab(a.get("text", "")) if hasattr(mind, "collab") else {"ok": False},
                    {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                )
            )
            hub.register(
                MCPTool(
                    "collab.project",
                    "Create a shared collab project",
                    lambda a: mind.google(
                        "collab",
                        "project",
                        name=a.get("name", "Project"),
                        description=a.get("description", ""),
                    )
                    if hasattr(mind, "google")
                    else {"ok": False},
                )
            )

        self.meta["wired_alpha"] = True
        self.meta["tools"] = len(hub.tools)
        return {"ok": True, "tools": hub.list_tools(), "n": len(hub.tools)}

    def spin(self) -> Dict[str, Any]:
        if not self.mcp._wired:
            # ensure base organs wired if mind present
            pass
        self.wire_alpha_tools()
        out = self.mcp.spin_up()
        self.spun = bool(out.get("ok"))
        self.meta["spin"] = out
        return {**out, "portal": self.manifest()}

    def call(self, tool: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.meta.get("wired_alpha"):
            self.wire_alpha_tools()
        return self.mcp.call(tool, arguments)

    def list_tools(self) -> Dict[str, Any]:
        if not self.meta.get("wired_alpha"):
            self.wire_alpha_tools()
        return self.mcp.list_tools()

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema": "auro.interior.portal.v1",
            "alpha": True,
            "spun": self.spun,
            "mcp_url": self.meta.get("spin", {}).get("url"),
            "tools_n": len(self.mcp.hub.tools),
            "tools": [t["name"] for t in self.mcp.hub.list_tools()],
            "fleet": self.fleet.manifest(),
            "capabilities": [
                "multi_site_parallel",
                "internet_ui_control",
                "chrome_dom",
                "polyglot_suite",
                "mini_brain_teach",
                "entangled_train",
                "python_organ",
                "chaos_cuda",
            ],
            "lab": "Novel Chaos Labs",
        }


def build_portal(mind: Any, *, chrome_mock: bool = True) -> InteriorPortal:
    """Attach interior portal to a mind (uses mind.organs.mcp if present)."""
    from auro_native_llm.embedded.mcp_hub import MCPOrgan

    mcp = getattr(mind.organs, "mcp", None) or MCPOrgan()
    # wire base organs if empty
    if not mcp._wired:
        mcp.wire_from_mind_organs(
            monaco=getattr(mind.organs, "monaco", None),
            jupyter=getattr(mind.organs, "jupyter", None),
            search=getattr(mind.organs, "search", None),
            chrome=getattr(mind.organs, "chrome", None),
            mind_info=lambda: mind.info(),
        )
    fleet = MultiSiteFleet(mock=chrome_mock)
    portal = InteriorPortal(mcp=mcp, fleet=fleet, mind_ref=mind)
    portal.wire_alpha_tools()
    mind.organs.portal = portal  # type: ignore[attr-defined]
    mind.organs.fleet = fleet  # type: ignore[attr-defined]
    mind.organs.mcp = mcp
    return portal
