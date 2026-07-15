"""Generate Grok-native SKILL.md files from tool registry."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from mesie.tools.registry import SKILL_CATEGORIES, TOOLS, NativeTool

ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = ROOT / ".grok" / "skills"


def _skill_body(skill_name: str, tools: List[NativeTool]) -> str:
    lines = [
        f"# {skill_name}",
        "",
        f"Native MESIE / MAESI / NeuroAIX skill — **{SKILL_CATEGORIES.get(tools[0].category, 'general')}**.",
        "",
        "## When to use",
        "",
    ]
    for t in tools:
        lines.append(f"- {t.description}")
    lines.extend(["", "## Tools in this skill", ""])
    for t in tools:
        lines.append(f"### `{t.id}` — {t.name}")
        lines.append(f"- Command: `{t.command}`")
        if t.deliverable:
            lines.append(f"- Deliverable: `{t.deliverable}`")
        lines.append("")
    lines.extend([
        "## Agent workflow",
        "",
        "1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`",
        "2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`",
        "3. Or run the command above directly.",
        "4. Read deliverable path if present; summarize results for the user.",
        "5. On failure: run `python -m mesie.tools.cli run test` to verify environment.",
        "",
        "## Repo paths",
        "",
        f"- Tools registry: `mesie/tools/registry.py`",
        f"- CLI: `python -m mesie.tools.cli list`",
    ])
    return "\n".join(lines)


def _write_skill(skill_name: str, description: str, tools: List[NativeTool]) -> Path:
    d = SKILLS_ROOT / skill_name
    d.mkdir(parents=True, exist_ok=True)
    triggers = ", ".join(sorted({tr for t in tools for tr in t.triggers}))
    frontmatter = (
        f"---\n"
        f"name: {skill_name}\n"
        f"description: >\n"
        f"  {description} Triggers: {triggers}. Use for /{skill_name} or MESIE/MAESI/NeuroAIX tasks.\n"
        f"---\n\n"
    )
    path = d / "SKILL.md"
    path.write_text(frontmatter + _skill_body(skill_name, tools), encoding="utf-8")
    return path


def _hub_skill(by_skill: Dict[str, List[NativeTool]]) -> Path:
    d = SKILLS_ROOT / "mesie-hub"
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "name: mesie-hub",
        "description: >",
        "  Master hub for MESIE, MAESI, and NeuroAIX. Routes to all native skills and tools.",
        "  Triggers: mesie, maesi, neuroaix, spectral engine, octopus, fingerprint, monte carlo.",
        "  Use when user asks to run MESIE or needs the tool catalog.",
        "---",
        "",
        "# MESIE / MAESI / NeuroAIX Hub",
        "",
        f"Unified native suite: **{len(TOOLS)} tools**, **{len(by_skill) + 1} skills** (incl. hub).",
        "",
        "```bash",
        "python -m mesie.tools.cli list",
        "python -m mesie.tools.cli run <tool-id>",
        "```",
        "",
        "## Skills map",
        "",
        "| Skill | Tools |",
        "|-------|-------|",
    ]
    for skill, tools in sorted(by_skill.items()):
        ids = ", ".join(t.id for t in tools)
        lines.append(f"| `/{skill}` | {ids} |")
    lines.extend([
        "",
        "## Quick enterprise stack",
        "",
        "1. `/mesie-embed-library` — index corpus",
        "2. `/mesie-fingerprint` — TF + LSH + ANN",
        "3. `/mesie-octopus` — eight-arm workflow",
        "4. `/mesie-maesi` — knowledge + fast compute",
        "5. `/mesie-enterprise` — Monte Carlo 10 use cases",
        "",
        "Regenerate skills: `python -m mesie.tools.cli skills`",
    ])
    path = d / "SKILL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_all() -> int:
    by_skill: Dict[str, List[NativeTool]] = {}
    for t in TOOLS:
        by_skill.setdefault(t.skill_name, []).append(t)

    count = 0
    for skill_name, tools in by_skill.items():
        desc = tools[0].description if len(tools) == 1 else f"{len(tools)} native tools for {skill_name.replace('mesie-', '')}."
        _write_skill(skill_name, desc, tools)
        count += 1
    _hub_skill(by_skill)
    count += 1
    return count


if __name__ == "__main__":
    print(f"Generated {generate_all()} skills -> {SKILLS_ROOT}")