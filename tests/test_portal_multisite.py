"""Interior MCP portal + multi-site concurrent agents."""

from __future__ import annotations

from auro_native_llm.agents.multisite import MultiSiteFleet
from auro_native_llm.embedded.portal import build_portal
from auro_native_llm.organism.family import build_mind


def test_fleet_open_many_parallel():
    fleet = MultiSiteFleet(mock=True, max_workers=4)
    rep = fleet.open_many(
        [
            "https://example.com",
            "https://example.org",
            "https://www.wikipedia.org",
        ]
    )
    assert rep.ok is True
    assert rep.parallel is True
    assert len(rep.results) == 3
    assert len(fleet.sites) == 3
    reads = fleet.read_all()
    assert len(reads.results) == 3
    work = fleet.work_objective(
        "survey",
        ["https://example.com", "https://example.org"],
    )
    assert work["ok"] is True
    assert "summary_for_llm" in work
    assert work["latency_ms"] >= 0


def test_portal_spin_and_multisite_tools():
    mind = build_mind("Auro-2B", lite=True)
    portal = build_portal(mind, chrome_mock=True)
    spun = portal.spin()
    assert spun.get("ok") is True
    tools = portal.list_tools()
    names = {t["name"] for t in tools["tools"]}
    for need in (
        "sites.open_many",
        "sites.read_all",
        "sites.work",
        "sites.act_all",
        "portal.manifest",
        "chrome.navigate",
    ):
        assert need in names, names
    # call multi-site via portal MCP
    r = portal.call(
        "sites.work",
        {
            "objective": "alpha multi-site",
            "urls": ["https://example.com", "https://example.org"],
        },
    )
    assert r.get("ok") is True
    assert r["result"]["ok"] is True
    assert len(r["result"]["urls"]) == 2


def test_mind_portal_and_multi_site():
    mind = build_mind("Auro-2B", lite=True)
    spun = mind.portal_open(chrome_mock=True)
    assert spun.get("ok") is True
    out = mind.multi_site(
        "Compare example.com and example.org titles",
        ["https://example.com", "https://example.org"],
        chrome_mock=True,
    )
    assert out.get("ok") is True
    assert "summary_for_llm" in out
    assert "portal_open" in mind.info()["capabilities"]
    assert "multi_site" in mind.info()["capabilities"]
