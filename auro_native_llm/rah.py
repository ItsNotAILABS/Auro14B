"""Recursive Agent Harness (RAH) for Auro — same doctrine as POCKET RAH.

The recursive unit is a **full Auro sub-agent harness**:
  · role + parent capacity (MultiEmbeddedSubAgentRouter)
  · own receipt on disk (~/.auro/rah/<run_id>/)
  · optional native think when the runtime is loaded
  · parallel fan-out with a hard cap
  · verify + synthesize after leaves finish

Not a bare model call. Independent goals only.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter
from auro_native_llm.types import SubAgentRole

ROOT = Path.home() / ".auro" / "rah"
ROOT.mkdir(parents=True, exist_ok=True)
SCHEMA = "auro.rah.v1"
PROTOCOL = "AURO-RAH/1.0"
MAX_PARALLEL = 8
MAX_DEPTH = 3
MAX_LEAVES = 12


def _role_for(goal: str) -> SubAgentRole:
    low = (goal or "").lower()
    if any(w in low for w in ("code", "python", "fix", "implement", "edit", "patch")):
        return SubAgentRole.CODE_EDIT
    if any(w in low for w in ("match", "spectrum", "psd", "embed", "fingerprint")):
        return SubAgentRole.SPECTRAL_MATCH
    if any(w in low for w in ("plan", "split", "orchestrat", "fan-out", "fan out")):
        return SubAgentRole.PLAN
    if any(w in low for w in ("critique", "review", "audit")):
        return SubAgentRole.CRITIQUE
    if any(w in low for w in ("tool", "call api", "json")):
        return SubAgentRole.TOOL_PLAN
    return SubAgentRole.REASON


def plan_fanout(task: str, *, max_leaves: int = 8) -> Dict[str, Any]:
    """Split a task into independent Auro leaves (Pocket-style)."""
    text = (task or "").strip()
    leaves: List[str] = []
    for line in text.replace(";", "\n").splitlines():
        s = line.strip()
        s = re.sub(r"^\s*(?:[-*]|\d+[\.)])\s+", "", s)
        if len(s) >= 8:
            leaves.append(s[:500])
    if len(leaves) < 2:
        leaves = [
            f"Perceive and name the facts in: {text[:240]}",
            f"Reason a local Auro answer for: {text[:240]}",
            f"Plan the next host action for: {text[:240]}",
        ]
    leaves = leaves[: max(1, min(int(max_leaves or 8), MAX_LEAVES))]
    return {
        "schema": SCHEMA,
        "task": text[:800],
        "leaves": [{"id": f"L{i+1}", "goal": g, "role": _role_for(g).value} for i, g in enumerate(leaves)],
        "verify": True,
        "synthesize": True,
    }


def _think_native(goal: str, model_id: str = "Auro-2B") -> Optional[str]:
    """Best-effort native think. None if weights/runtime are not loaded."""
    try:
        from auro_native_llm.use import main as _unused  # noqa: F401
    except Exception:
        pass
    try:
        from auro_native_llm.native_runtime import bootstrap_runtime

        rt = bootstrap_runtime(model_id=model_id, lite=True)
        if rt is None:
            return None
        think = getattr(rt, "think", None) or getattr(rt, "generate", None) or getattr(rt, "run", None)
        if not callable(think):
            return None
        out = think(goal)
        if isinstance(out, dict):
            return str(out.get("text") or out.get("reply") or out)[:8000]
        return str(out)[:8000]
    except Exception:
        return None


def _run_leaf(
    leaf: Dict[str, Any],
    *,
    run_id: str,
    router: MultiEmbeddedSubAgentRouter,
    depth: int,
    max_depth: int,
) -> Dict[str, Any]:
    lid = leaf.get("id") or f"L{uuid.uuid4().hex[:6]}"
    goal = (leaf.get("goal") or leaf.get("prompt") or "").strip()
    role = _role_for(goal)
    started = time.time()
    rec: Dict[str, Any] = {
        "id": lid,
        "goal": goal[:500],
        "role": role.value,
        "depth": depth,
        "status": "running",
        "started_at": started,
        "run_id": run_id,
        "engine": "auro-rah",
    }
    native = _think_native(goal)
    try:
        d = router.dispatch(role, goal)
        rec["dispatch_ok"] = bool(getattr(d, "ok", False))
        rec["child_model"] = getattr(d, "child_model_id", "")
        rec["agent_id"] = getattr(d, "agent_id", "")
        rec["dispatch"] = getattr(d, "message", str(d))[:1500]
    except Exception as e:
        rec["dispatch_ok"] = False
        rec["error"] = str(e)[:400]
    if native:
        rec["result"] = native
        rec["native"] = True
        rec["ok"] = True
        rec["status"] = "done"
    else:
        rec["result"] = rec.get("dispatch") or rec.get("error") or ""
        rec["native"] = False
        rec["ok"] = bool(rec.get("dispatch_ok"))
        rec["status"] = "done" if rec["ok"] else "fail"
        rec.setdefault("note", "scaffold dispatch — native think not loaded")

    rec["finished_at"] = time.time()
    rec["duration_sec"] = round(rec["finished_at"] - started, 2)
    path = ROOT / run_id / f"{lid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    rec["path"] = str(path)

    if depth < max_depth and rec.get("ok") and "FANOUT:" in (rec.get("result") or ""):
        sub_goals = [ln.strip("-* ") for ln in (rec.get("result") or "").splitlines() if ln.strip().startswith(("-", "*"))]
        if len(sub_goals) >= 2:
            rec["sub_rah"] = run_rah(
                rec["goal"],
                leaves=sub_goals[:6],
                depth=depth + 1,
                max_depth=max_depth,
                max_parallel=2,
                parent_model_id=router.parent_model_id,
            )
    return rec


def _verify(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    fails = [r["id"] for r in results if not r.get("ok")]
    return {
        "ok": not fails,
        "failed": fails,
        "conflicts": [f"leaf {i} failed" for i in fails],
        "n": len(results),
    }


def _synthesize(task: str, run_id: str, results: List[Dict[str, Any]], verify: Dict[str, Any]) -> str:
    lines = [
        f"# Auro RAH · {run_id}",
        "",
        f"**Task:** {task[:400]}",
        f"**Leaves:** {len(results)} · verify={'OK' if verify.get('ok') else 'FAIL'}",
        "",
        "## Leaf results",
        "",
    ]
    for r in results:
        st = "OK" if r.get("ok") else "FAIL"
        lines.append(f"### {r.get('id')} · {st} · {r.get('role')} · {r.get('duration_sec')}s")
        lines.append((r.get("result") or r.get("error") or "")[:900] or "_(empty)_")
        lines.append("")
    lines.append(f"_Artifacts: `~/.auro/rah/{run_id}/`_")
    text = "\n".join(lines)
    try:
        (ROOT / run_id / "synthesis.md").write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text


def run_rah(
    task: str,
    *,
    leaves: Optional[List[Any]] = None,
    parent_model_id: str = "Auro-14B",
    max_parallel: int = 4,
    depth: int = 0,
    max_depth: int = MAX_DEPTH,
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fan independent Auro harness leaves; synthesize a receipt."""
    if depth > max_depth:
        return {"ok": False, "error": f"max depth {max_depth} exceeded", "protocol": PROTOCOL}
    if plan and plan.get("leaves"):
        spec = list(plan["leaves"])
        goals = []
        for item in spec:
            if isinstance(item, dict):
                goals.append(item)
            else:
                goals.append({"goal": str(item)})
    elif leaves:
        goals = [{"goal": str(g)} for g in leaves if str(g).strip()]
    else:
        goals = plan_fanout(task).get("leaves") or [{"goal": task}]

    rid = ("ar-d%d-" % depth if depth else "ar-") + uuid.uuid4().hex[:10]
    run_dir = ROOT / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    router = MultiEmbeddedSubAgentRouter(parent_model_id=parent_model_id)
    started = time.time()
    workers = max(1, min(int(max_parallel or 4), MAX_PARALLEL, len(goals)))
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_run_leaf, g if isinstance(g, dict) else {"goal": str(g)}, run_id=rid, router=router, depth=depth, max_depth=max_depth)
            for g in goals
        ]
        for f in as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda r: str(r.get("id") or ""))
    verify = _verify(results)
    synthesis = _synthesize(task, rid, results, verify)
    out = {
        "ok": bool(verify.get("ok")),
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "run_id": rid,
        "task": (task or "")[:500],
        "leaves": len(results),
        "parallel": workers,
        "depth": depth,
        "duration_sec": round(time.time() - started, 2),
        "verify": verify,
        "results": results,
        "synthesis": synthesis[:4000],
        "path": str(run_dir),
        "doctrine": "Auro RAH: full sub-agent harness leaves, filesystem receipts, parallel cap.",
    }
    (run_dir / "RUN.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
