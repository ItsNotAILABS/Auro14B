#!/usr/bin/env python3
"""Inventory local AURO checkpoints without guessing from architecture names."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAMES = ("manifest.json", "checkpoint_manifest.json", "config.json", "model_card.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    manifests = []
    for name in MANIFEST_NAMES:
        candidate = path / name
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                payload = {"parse_error": str(exc)}
            manifests.append({"path": str(candidate), "sha256": sha256_file(candidate), "payload": payload})
    weight_files = []
    for pattern in ("*.pt", "*.pth", "*.safetensors", "*.npz", "*.npz.b64", "*.bin"):
        for candidate in sorted(path.glob(pattern)):
            weight_files.append({"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)})
    return {
        "path": str(path),
        "name": path.name,
        "manifests": manifests,
        "weight_files": weight_files,
        "has_weight_artifact": bool(weight_files),
        "has_manifest": bool(manifests),
        "evidence_complete": bool(weight_files and manifests),
    }


def inventory(root: Path) -> dict[str, Any]:
    candidates = []
    if root.exists():
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            if "auro" in path.name.lower() or "him" in path.name.lower():
                candidates.append(inspect_checkpoint(path))
    auro_2b = [item for item in candidates if "2b" in item["name"].lower()]
    return {
        "schema": "auro.checkpoint.inventory.v1",
        "root": str(root),
        "checkpoints": candidates,
        "auro_2b_candidates": auro_2b,
        "auro_2b_present": any(item["has_weight_artifact"] for item in auro_2b),
        "auro_2b_evidence_complete": any(item["evidence_complete"] for item in auro_2b),
        "claim_boundary": "architecture configs and README names are not substituted for local weight and manifest evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/auro_minds")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inventory(Path(args.root))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
