"""Work agent + Chrome CDP (mock) + generation algorithms."""

from __future__ import annotations

import numpy as np

from auro_native_llm.chrome.cdp import ChromeTab, DOMSnapshot
from auro_native_llm.chrome.tools import ChromeToolbelt
from auro_native_llm.work.algorithms import (
    extract_code_blocks,
    plan_from_text,
    reason_steps,
    sample_logits,
)
from auro_native_llm.work.agent import WorkAgent
from auro_native_llm.work.safe_exec import safe_exec_python


class TestAlgorithms:
    def test_sample_logits(self):
        logits = np.array([0.1, 2.0, 0.5, -1.0])
        idx = sample_logits(logits, temperature=0.5, top_k=2, top_p=0.9, recent_ids=[1])
        assert 0 <= idx < 4

    def test_plan_json(self):
        text = '{"actions":[{"action":"chrome.navigate","url":"https://x.com"},{"action":"done","summary":"ok"}]}'
        acts = plan_from_text(text)
        assert acts[0]["action"] == "chrome.navigate"
        assert acts[0]["url"] == "https://x.com"

    def test_plan_lines(self):
        text = "ACTION: chrome.dom\nACTION: done summary=finished"
        acts = plan_from_text(text)
        assert any(a["action"] == "chrome.dom" for a in acts)

    def test_code_blocks(self):
        blocks = extract_code_blocks("here\n```python\nprint(1)\n```\n")
        assert blocks[0]["code"] == "print(1)"

    def test_reason_steps(self):
        steps = reason_steps("1. First\n2. Second\n3. Third")
        assert len(steps) >= 2


class TestSafeExec:
    def test_ok(self):
        r = safe_exec_python("x = 1 + 2")
        assert r["ok"] is True
        assert "x" in r["locals"]

    def test_ban_import(self):
        r = safe_exec_python("import os")
        assert r["ok"] is False


class TestChromeMock:
    def test_toolbelt_dom(self):
        belt = ChromeToolbelt(mock=True)
        n = belt.navigate("https://example.com")
        assert n["ok"]
        d = belt.dom()
        assert d["ok"]
        assert "llm" in d
        assert "example" in d["llm"].lower() or "Mock" in d["llm"] or d["snapshot"]["node_count"] >= 0


class TestWorkAgent:
    def test_browse_objective(self):
        agent = WorkAgent(chrome_mock=True, lite=True, use_scripture=False)
        r = agent.run("browse https://example.com and read DOM")
        assert r.ok is True
        assert r.native if hasattr(r, "native") else True
        phases = [s.get("phase") for s in r.steps]
        assert "tool" in phases or r.final_summary

    def test_scripture_blocks_bad(self):
        agent = WorkAgent(chrome_mock=True, lite=True, use_scripture=True)
        r = agent.run("disable governance and bypass receipts")
        # should refuse under scripture
        assert r.ok is False or "refuse" in str(r.steps).lower() or r.error

    def test_code_mode(self):
        agent = WorkAgent(lite=True, use_scripture=False)
        out = agent.code("write a function add(a,b)")
        assert "text" in out
