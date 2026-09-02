from __future__ import annotations

import argparse
import json
from pathlib import Path

from auro_native_llm.harness.viral import AuroArenaHarness, launch_challenges


def browser_rescue_step(index: int):
    if index == 0:
        return {
            "observation": {"url": "https://example.invalid/demo", "dom": "button#legacy-submit missing; button[data-action=submit] present"},
            "proposal": {"action": "recover_selector", "from": "#legacy-submit", "to": "button[data-action=submit]"},
            "decision": {"allowed": True, "approvalRequired": False, "approved": True},
            "result": {"recovered": True, "done": False},
            "done": False,
        }
    return {
        "observation": {"selector": "button[data-action=submit]", "visible": True},
        "proposal": {"action": "read_only_verify", "selector": "button[data-action=submit]"},
        "decision": {"allowed": True, "approvalRequired": False, "approved": True},
        "result": {"verified": True, "done": True},
        "done": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic AURO Arena public showcase receipt")
    parser.add_argument("--out", default="artifacts/auro-arena-browser-rescue.json")
    args = parser.parse_args()

    challenge = next(c for c in launch_challenges() if c.id == "browser-rescue")
    harness = AuroArenaHarness()
    run = harness.run(challenge, browser_rescue_step)
    harness.score(challenge, run, {"task_success": 1.0, "recovery": 1.0, "efficiency": .9, "safety": 1.0})
    receipt = harness.public_receipt(challenge, run)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_id": run.run_id, "score": run.score, "receipt_sha256": receipt["receipt_sha256"], "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
