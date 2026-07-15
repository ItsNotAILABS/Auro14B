"""Smoke tests for multi-domain analysis suites."""

from mesie.analysis import run_all_suites


def test_run_all_suites_smoke():
    data = run_all_suites()
    assert data["suite_count"] == 5
    domains = {s["domain"] for s in data["suites"]}
    assert domains == {"terrain", "robotics", "orbital", "power", "seismic"}
    for s in data["suites"]:
        assert s["plain_conclusion"]
        assert s["elapsed_ms"] >= 0