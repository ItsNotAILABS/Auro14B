#!/usr/bin/env python3
"""Quality gate for this deliverable packet. Zero external dependencies."""
import json
import os
import sys

REQUIRED_ARTIFACTS = [
    "README.md",
    "PACKET_POLICY.md",
    "manifest.json",
    "release_manifest.json",
    "reports/build_report.json",
    "scores/quality_score.json",
    "tools/quality_gate.py",
]

MANIFEST_KEYS = ["name", "version", "kind", "author", "created"]


def run_gate(root):
    errors = []

    # Check artifacts
    for artifact in REQUIRED_ARTIFACTS:
        if not os.path.exists(os.path.join(root, artifact)):
            errors.append(f"Missing: {artifact}")

    # Check manifest
    manifest_path = os.path.join(root, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in MANIFEST_KEYS:
                if key not in data:
                    errors.append(f"manifest.json missing key: {key}")
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"manifest.json invalid: {e}")

    # Check quality score
    score_path = os.path.join(root, "scores", "quality_score.json")
    if os.path.exists(score_path):
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                score = json.load(f)
            if not score.get("pass", False):
                errors.append("quality_score.json: pass is not true")
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"quality_score.json invalid: {e}")

    # Check README has verification
    readme_path = os.path.join(root, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if not any(kw in content for kw in ["verify", "validation", "quality_gate", "test"]):
            errors.append("README.md lacks verification instructions")

    # Report
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    else:
        print("PASSED — All checks green")
        return 0


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    sys.exit(run_gate(os.path.abspath(root)))
