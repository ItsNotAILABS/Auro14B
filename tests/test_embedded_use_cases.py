"""100 use cases × 4 domains (Monaco, Jupyter, search, MCP) + mind wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auro_native_llm.embedded.monaco import MonacoOrgan
from auro_native_llm.embedded.jupyter import JupyterOrgan
from auro_native_llm.embedded.search import SearchOrgan
from auro_native_llm.embedded.mcp_hub import MCPOrgan
from auro_native_llm.embedded.runner import load_suite, run_suite
from auro_native_llm.organism.family import build_mind

UC = Path(__file__).resolve().parents[1] / "auro_native_llm" / "embedded" / "use_cases"


def _cases(domain: str):
    data = load_suite(domain)
    assert data["count"] == 100
    return data["cases"]


# ---------------------------------------------------------------------------
# Full 100-case suites
# ---------------------------------------------------------------------------


def test_monaco_100_use_cases():
    report = run_suite("monaco", MonacoOrgan(), limit=100)
    assert report["total"] == 100
    assert report["passed"] == 100, report.get("failures")


def test_jupyter_100_use_cases():
    report = run_suite("jupyter", JupyterOrgan(), limit=100)
    assert report["total"] == 100
    assert report["passed"] == 100, report.get("failures")


def test_search_100_use_cases():
    organ = SearchOrgan(offline=True)
    report = run_suite("search", organ, limit=100)
    assert report["total"] == 100
    assert report["passed"] == 100, report.get("failures")


def test_mcp_100_use_cases():
    organ = MCPOrgan()
    organ.wire_from_mind_organs(
        monaco=MonacoOrgan(),
        jupyter=JupyterOrgan(),
        search=SearchOrgan(offline=True),
    )
    report = run_suite("mcp", organ, limit=100)
    assert report["total"] == 100
    assert report["passed"] == 100, report.get("failures")


# ---------------------------------------------------------------------------
# Parametrized samples (fast regression on ids)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _cases("monaco")[::10], ids=lambda c: c["id"])
def test_monaco_parametrized_sample(case):
    from auro_native_llm.embedded.runner import run_monaco_case

    r = run_monaco_case(MonacoOrgan(), case)
    assert r["ok"], r


@pytest.mark.parametrize("case", _cases("jupyter")[::10], ids=lambda c: c["id"])
def test_jupyter_parametrized_sample(case):
    from auro_native_llm.embedded.runner import run_jupyter_case

    r = run_jupyter_case(JupyterOrgan(), case)
    assert r["ok"], r


# ---------------------------------------------------------------------------
# Mind embedding + teach
# ---------------------------------------------------------------------------


def test_mind_has_new_organs():
    mind = build_mind("Auro-2B", lite=True)
    m = mind.organs.manifest()
    for k in ("monaco", "jupyter", "search", "mcp", "curriculum"):
        assert m[k] is True, k


def test_mind_monaco_jupyter_search_mcp_teach():
    mind = build_mind("Auro-2B", lite=True)
    r = mind.monaco("create", content="def g():\n    return 1\n")
    assert r.ok and r.train_pulse is not None
    sid = r.output["session"]["session_id"]
    assert mind.monaco("append", session_id=sid, text="\n# x\n").ok

    j = mind.jupyter("create", title="Lab")
    assert j.ok
    nid = j.output["notebook"]["notebook_id"]
    add = mind.jupyter("add_cell", notebook_id=nid, source="z = 3\nz", cell_type="code")
    assert add.ok
    assert mind.jupyter("execute", notebook_id=nid, cell_id=add.output["cell_id"]).ok

    s = mind.search("MESIE", online=False)
    assert s.ok

    mcp = mind.mcp("spin_up")
    assert mcp.ok
    tools = mind.mcp("list_tools")
    assert tools.ok and len(tools.output["tools"]) >= 3
    call = mind.mcp("call", tool="search", arguments={"query": "Auro", "online": False})
    assert call.ok

    taught = mind.teach(["monaco", "jupyter", "search", "mcp"])
    assert taught.ok
    assert taught.output["taught_lessons"] >= 10


def test_use_case_files_exist():
    for name in ("monaco_100.json", "jupyter_100.json", "search_100.json", "mcp_100.json"):
        p = UC / name
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert len(data["cases"]) == 100
