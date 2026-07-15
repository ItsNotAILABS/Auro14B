from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def emit_receipt(kind: str, config_path: str | Path, config: dict[str, Any], status: str = "validated-scaffold") -> dict[str, Any]:
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    receipt = {
        "schema": "auro.native_llm.receipt.v1",
        "kind": kind,
        "status": status,
        "config_path": str(config_path),
        "config_sha256": sha256_text(normalized),
        "generated_at_unix": int(time.time()),
        "claim_boundary": "receipt validates scaffold/config only; it is not a trained checkpoint receipt"
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt
