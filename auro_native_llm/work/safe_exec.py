"""Safe-ish Python exec for code.run tool (restricted builtins)."""

from __future__ import annotations

import math
from typing import Any, Dict


_ALLOWED_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "round": round,
    "print": print,
    "True": True,
    "False": False,
    "None": None,
}


def safe_exec_python(code: str, timeout_note: str = "") -> Dict[str, Any]:
    """Execute a restricted Python snippet; return stdout-ish result dict."""
    banned = ("import ", "open(", "__import__", "subprocess", "os.", "sys.", "from ")
    # note: bare eval/exec words banned only as calls
    for b in banned:
        if b in code:
            return {"ok": False, "error": f"banned token: {b}", "code": code[:500]}
    if "exec(" in code or "eval(" in code:
        return {"ok": False, "error": "banned token: exec/eval call", "code": code[:500]}
    env: Dict[str, Any] = {"__builtins__": _ALLOWED_BUILTINS, "math": math}
    local: Dict[str, Any] = {}
    try:
        lines = [ln for ln in code.strip().splitlines() if ln.strip()]
        if not lines:
            return {"ok": True, "locals": {}, "code": ""}
        # Jupyter-style: if last line is expression, capture its value
        *body, last = lines
        if body:
            exec("\n".join(body), env, local)  # noqa: S102
        try:
            # try last as expression
            val = eval(last, env, local)  # noqa: S307 — restricted env only
            local["_"] = val
        except SyntaxError:
            exec(last, env, local)  # noqa: S102
        out = {k: repr(v)[:200] for k, v in local.items() if not k.startswith("_") or k == "_"}
        return {"ok": True, "locals": out, "code": code[:500]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "code": code[:500]}
