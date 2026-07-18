"""Validate and hash a local Transformers.js model export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(model_dir: Path) -> dict[str, object]:
    root = model_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory not found: {root}")
    onnx_files = sorted(root.rglob("*.onnx"))
    if not onnx_files:
        raise ValueError("No ONNX model files found; refusing to claim browser readiness")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "schema": "auro.browser.model.v1",
        "model_id": root.name,
        "remote_models_allowed": False,
        "onnx_files": len(onnx_files),
        "files": files,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    (root / "auro-browser-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
