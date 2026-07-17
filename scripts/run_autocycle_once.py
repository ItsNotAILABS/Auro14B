"""One-shot autocycle runner with Python organ smoke."""
from __future__ import annotations

import json

from auro_native_llm.embedded.python_organ import PythonOrgan
from auro_native_llm.organism.autocycle import AutocycleConfig, run_autocycle


def smoke() -> None:
    p = PythonOrgan()
    cases = [
        (
            "phi",
            "import math\nPHI=(1+5**0.5)/2\nresult=sum(PHI**i for i in range(1,13))\nprint(result)\n",
        ),
        ("gh", "hits=github_search('MESIE',2)\nprint(len(hits))\nresult=hits\n"),
        (
            "hash",
            "import hashlib\nresult=hashlib.sha256(b'x').hexdigest()[:16]\nprint(result)\n",
        ),
    ]
    for name, src in cases:
        r = p.run(src, intent=name)
        print(name, r.ok, (r.stdout or r.error or "")[:100])
    print("smoke ok_rate", p.info()["ok_rate"])


def main() -> None:
    smoke()
    r = run_autocycle(
        AutocycleConfig(
            cycles=6,
            train_steps_per_cycle=2,
            resume_checkpoint="checkpoints/auro_minds/Auro-2B_continual",
            show=True,
        )
    )
    print(
        json.dumps(
            {
                "python_ok_rate": r.get("python_ok_rate"),
                "runs": (r.get("python_organ") or {}).get("total_runs"),
                "oks": (r.get("python_organ") or {}).get("total_ok"),
                "steps": [r.get("train_steps_before"), r.get("train_steps_after")],
                "delta": r.get("train_steps_delta"),
                "params": r.get("num_params_live"),
                "cycles_ok": [
                    ((c.get("stages") or {}).get("act") or {}).get("python_ok")
                    for c in r.get("cycles") or []
                ],
                "outs": [
                    (((c.get("stages") or {}).get("act") or {}).get("stdout_preview") or "")[:60]
                    for c in r.get("cycles") or []
                ],
                "ckpt": r.get("checkpoint"),
                "elapsed": r.get("elapsed_s"),
                "laws": r.get("laws"),
                "loop": r.get("loop"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
