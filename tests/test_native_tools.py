"""Native tools and skills suite tests."""

from pathlib import Path

from mesie.tools.registry import TOOLS, tool_by_id


def test_registry_size():
    assert len(TOOLS) >= 29
    assert tool_by_id("ais-polyglot") is not None
    assert tool_by_id("monte-carlo") is not None
    assert tool_by_id("maesi") is not None
    assert tool_by_id("octopus") is not None
    assert tool_by_id("internal-bus") is not None
    assert tool_by_id("fast-compute") is not None
    assert tool_by_id("catalog") is not None


def test_skills_generated():
    root = Path(__file__).resolve().parents[1] / ".grok" / "skills"
    assert (root / "mesie-hub" / "SKILL.md").exists()
    assert (root / "mesie-fingerprint" / "SKILL.md").exists()
    assert (root / "mesie-enterprise" / "SKILL.md").exists()
    skill_dirs = [d for d in root.iterdir() if d.is_dir()]
    assert len(skill_dirs) >= 18
    assert (root / "mesie-internal" / "SKILL.md").exists()


def test_catalog_export(tmp_path):
    from mesie.tools.cli import cmd_catalog
    import argparse

    out = tmp_path / "catalog.json"
    args = argparse.Namespace(output=str(out))
    assert cmd_catalog(args) == 0
    data = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert data["tool_count"] == len(TOOLS)
    assert "mesie-hub" in data["skills"] or "mesie-octopus" in data["skills"]