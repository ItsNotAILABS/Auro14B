"""Optimized text generation / reasoning / coding helpers for Auro LM.

These operate on logits and text — first-class algorithms, not prompt theater.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def sample_logits(
    logits: np.ndarray,
    *,
    temperature: float = 0.85,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.12,
    recent_ids: Optional[Sequence[int]] = None,
    ban_ids: Optional[Sequence[int]] = None,
) -> int:
    """Nucleus + top-k sampling with repetition penalty."""
    x = np.asarray(logits, dtype=np.float64).copy()
    if ban_ids:
        for i in ban_ids:
            if 0 <= int(i) < x.size:
                x[int(i)] = -1e9
    if recent_ids and repetition_penalty != 1.0:
        for i in recent_ids:
            ii = int(i)
            if 0 <= ii < x.size:
                if x[ii] > 0:
                    x[ii] /= repetition_penalty
                else:
                    x[ii] *= repetition_penalty
    temp = max(float(temperature), 1e-6)
    x = x / temp
    if top_k > 0 and top_k < x.size:
        thr = np.partition(x, -top_k)[-top_k]
        x = np.where(x >= thr, x, -1e9)
    x = x - x.max()
    p = np.exp(x)
    p = p / (p.sum() + 1e-12)
    if top_p < 1.0:
        order = np.argsort(p)[::-1]
        cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cut]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / (mask.sum() + 1e-12)
    return int(np.random.choice(len(p), p=p))


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Pull fenced code blocks from model text."""
    blocks = []
    for m in re.finditer(r"```(\w*)\n(.*?)```", text, flags=re.DOTALL):
        blocks.append({"lang": m.group(1) or "text", "code": m.group(2).strip()})
    if not blocks:
        # bare function/class heuristic
        if re.search(r"^\s*(def|class|import|from)\s", text, re.M):
            blocks.append({"lang": "python", "code": text.strip()})
    return blocks


def plan_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse a tool plan: JSON array/object or line protocol ACTION: ..."""
    # JSON first
    for m in re.finditer(r"(\{[\s\S]*\}|\[[\s\S]*\])", text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "actions" in obj:
                return list(obj["actions"])
            if isinstance(obj, dict) and "action" in obj:
                return [obj]
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except json.JSONDecodeError:
            continue
    actions: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(
            r"^(?:ACTION|TOOL)\s*[:\-]\s*([A-Za-z0-9_.]+)\s*(?:\((.*)\))?(?:\s+(.+))?$",
            line,
            re.I,
        )
        if not m:
            m2 = re.match(r"^([A-Za-z0-9_.]+)\s*→\s*(.+)$", line)
            if m2:
                actions.append({"action": m2.group(1).lower(), "arg": m2.group(2)})
            continue
        name = m.group(1).lower()
        args = m.group(2) or m.group(3) or ""
        entry: Dict[str, Any] = {"action": name}
        # key=value pairs
        for kv in re.finditer(r"(\w+)\s*=\s*([^\s,]+|\"[^\"]*\"|'[^']*')", args):
            val = kv.group(2).strip("\"'")
            entry[kv.group(1)] = val
        if "url" not in entry and args.startswith("http"):
            entry["url"] = args.strip()
        if len(entry) == 1 and args:
            entry["arg"] = args.strip()
        actions.append(entry)
    return actions


def reason_steps(text: str, max_steps: int = 8) -> List[str]:
    """Extract ordered reasoning steps from model output."""
    steps = []
    for m in re.finditer(r"(?:^|\n)\s*(?:Step\s*)?(\d+)[.):\-]\s*(.+)", text):
        steps.append(m.group(2).strip())
        if len(steps) >= max_steps:
            break
    if not steps:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("- ", "* ", "• ")):
                steps.append(line[2:].strip())
            if len(steps) >= max_steps:
                break
    return steps


def code_complete(prompt: str, model_generate) -> Dict[str, Any]:
    """Coding-oriented generation wrapper.

    ``model_generate(prompt, **kw) -> text``
    """
    sys_hint = (
        "You are Auro coding engine on MESIE. Output a plan then a single Python fenced block. "
        "Prefer complete runnable functions. No cloud APIs.\n\n"
    )
    text = model_generate(sys_hint + prompt)
    if hasattr(text, "text"):
        text = text.text
    blocks = extract_code_blocks(str(text))
    return {
        "text": str(text),
        "code_blocks": blocks,
        "primary_code": blocks[0]["code"] if blocks else "",
        "lang": blocks[0]["lang"] if blocks else "",
    }


WORK_SYSTEM = """You are Auro Work Agent (native MESIE LLM). You do not only chat — you emit tool plans.
Available actions (JSON list under key actions OR lines ACTION: name key=value):
  chrome.navigate url=...
  chrome.dom
  chrome.click x=.. y=..
  chrome.type text=...
  chrome.eval js=...
  code.run code=...   (safe python subset)
  reason topic=...
  memory.write text=...
  done summary=...
Rules: prefer chrome.dom after navigate; never disable governance; compute is MESIE.
After tools, end with ACTION: done summary=...
"""


def build_work_prompt(objective: str, dom_context: str = "", memory: str = "") -> str:
    parts = [WORK_SYSTEM, f"OBJECTIVE: {objective}"]
    if memory:
        parts.append(f"MEMORY:\n{memory}")
    if dom_context:
        parts.append(f"CHROME_DOM:\n{dom_context}")
    parts.append("Emit your next tool plan now.")
    return "\n\n".join(parts)
