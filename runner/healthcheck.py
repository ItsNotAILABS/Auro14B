from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    runner_root = Path("/actions-runner")
    receipt = Path(os.getenv("MESIE_RECEIPTS", "/opt/mesie/receipts")) / "compute-node.json"
    if not (runner_root / ".runner").exists():
        raise SystemExit("runner is not configured")
    if not receipt.exists():
        raise SystemExit("MESIE compute-node receipt is missing")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("schema") != "mesie-compute-node/1.0":
        raise SystemExit("invalid MESIE compute-node receipt")
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit("GPU is unavailable")
    print("mesie-runner-healthy")


if __name__ == "__main__":
    main()
