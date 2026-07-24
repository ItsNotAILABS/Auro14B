"""Execute use-case JSON suites against embedded organs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


USE_CASE_DIR = Path(__file__).resolve().parent / "use_cases"


def load_suite(domain: str) -> Dict[str, Any]:
    path = USE_CASE_DIR / f"{domain}_100.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_monaco_case(organ: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    sid = None
    last: Dict[str, Any] = {}
    for step in case["steps"]:
        op = step["op"]
        if op == "create":
            last = organ.create(
                step.get("content", ""),
                language=step.get("language", "python"),
                filename=step.get("filename", "main.py"),
            )
            sid = last.get("session", {}).get("session_id")
        elif op == "append" and sid:
            last = organ.append(sid, step.get("text", ""))
        elif op == "replace" and sid:
            last = organ.replace(sid, step.get("old", ""), step.get("new", ""))
        elif op == "insert" and sid:
            last = organ.insert(sid, int(step.get("offset", 0)), step.get("text", ""))
        elif op == "get" and sid:
            last = organ.get(sid)
        elif op == "set" and sid:
            last = organ.set_content(sid, step.get("content", ""))
        if not last.get("ok", True):
            return {"ok": False, "id": case["id"], "error": last, "step": op}
    exp = case.get("expect", {})
    version = last.get("session", {}).get("version", 0)
    ok = last.get("ok", False) and version >= int(exp.get("min_version", 1))
    return {"ok": ok, "id": case["id"], "version": version}


def run_jupyter_case(organ: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    nid = None
    last_cell = None
    last: Dict[str, Any] = {}
    for step in case["steps"]:
        op = step["op"]
        if op == "create":
            last = organ.create(step.get("title", "NB"))
            nid = last.get("notebook", {}).get("notebook_id")
        elif op == "add_cell" and nid:
            last = organ.add_cell(nid, step.get("source", ""), cell_type=step.get("cell_type", "code"))
            last_cell = last.get("cell_id")
        elif op == "execute_last" and nid and last_cell:
            last = organ.execute_cell(nid, last_cell)
        elif op == "execute_all" and nid:
            last = organ.execute_all(nid)
        elif op == "export" and nid:
            last = organ.export_ipynb(nid)
        if last.get("ok") is False:
            return {"ok": False, "id": case["id"], "error": last, "step": op}
    exp = case.get("expect", {})
    ok = last.get("ok", False)
    if exp.get("has_ipynb"):
        ok = ok and "ipynb" in last
    return {"ok": ok, "id": case["id"]}


def run_search_case(organ: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    step = case["steps"][0]
    last = organ.search(
        step.get("query", ""),
        online=bool(step.get("online", False)),
        top_k=int(step.get("top_k", 3)),
    )
    ok = last.get("ok", False) and len(last.get("hits", [])) >= int(case.get("expect", {}).get("min_hits", 0))
    return {"ok": ok, "id": case["id"], "hits": len(last.get("hits", []))}


def run_mcp_case(organ: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    for step in case["steps"]:
        op = step["op"]
        if op == "spin_up":
            last = organ.spin_up()
        elif op == "list_tools":
            last = organ.list_tools()
        elif op == "call":
            last = organ.call(step.get("tool", ""), step.get("arguments") or {})
        if last.get("ok") is False:
            return {"ok": False, "id": case["id"], "error": last, "step": op}
    tools = last.get("tools") if "tools" in last else None
    # after call, tools may be nested
    if tools is None and case["steps"][-1]["op"] == "list_tools":
        tools = last.get("tools", [])
    min_tools = int(case.get("expect", {}).get("min_tools", 0))
    # re-list for check
    listed = organ.list_tools()
    ntools = len(listed.get("tools", []))
    ok = last.get("ok", False) and ntools >= min_tools
    return {"ok": ok, "id": case["id"], "n_tools": ntools}


def run_suite(domain: str, organ: Any, limit: Optional[int] = None) -> Dict[str, Any]:
    suite = load_suite(domain)
    cases = suite["cases"][: limit or len(suite["cases"])]
    runners = {
        "monaco": run_monaco_case,
        "jupyter": run_jupyter_case,
        "search": run_search_case,
        "mcp": run_mcp_case,
    }
    run = runners[domain]
    results = [run(organ, c) for c in cases]
    passed = sum(1 for r in results if r.get("ok"))
    return {
        "domain": domain,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "ok": passed == len(results),
        "failures": [r for r in results if not r.get("ok")][:10],
    }
