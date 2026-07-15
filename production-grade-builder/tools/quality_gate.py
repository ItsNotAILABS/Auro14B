#!/usr/bin/env python3
"""
Production-Grade Quality Gate
Zero external dependencies. Validates any deliverable packet.
"""
import json
import os
import sys


REQUIRED_BASE_ARTIFACTS = [
    "README.md",
    "PACKET_POLICY.md",
    "manifest.json",
    "release_manifest.json",
    "reports/build_report.json",
    "scores/quality_score.json",
    "tools/quality_gate.py",
]

MANIFEST_REQUIRED_KEYS = ["name", "version", "kind", "author", "created"]


def check_artifacts_exist(root, artifacts):
    """Check that all required artifacts exist."""
    missing = []
    for artifact in artifacts:
        path = os.path.join(root, artifact)
        if not os.path.exists(path):
            missing.append(artifact)
    return missing


def check_manifest(root):
    """Validate manifest.json structure."""
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        return ["manifest.json does not exist"]

    errors = []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return [f"manifest.json is not valid JSON: {e}"]

    for key in MANIFEST_REQUIRED_KEYS:
        if key not in data:
            errors.append(f"manifest.json missing required key: {key}")

    return errors


def check_quality_score(root):
    """Validate quality_score.json reports pass."""
    score_path = os.path.join(root, "scores", "quality_score.json")
    if not os.path.exists(score_path):
        return ["scores/quality_score.json does not exist"]

    try:
        with open(score_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return [f"quality_score.json is not valid JSON: {e}"]

    if not data.get("pass", False):
        return ["quality_score.json reports pass=false"]

    return []


def check_readme_has_verification(root):
    """Check README contains verification instructions."""
    readme_path = os.path.join(root, "README.md")
    if not os.path.exists(readme_path):
        return ["README.md does not exist"]

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    verification_keywords = ["verify", "verification", "validate", "quality_gate", "quality gate", "test"]
    if not any(kw in content for kw in verification_keywords):
        return ["README.md does not contain verification instructions"]

    return []


def run_gate(root):
    """Run all quality gate checks."""
    print(f"Running quality gate on: {root}")
    print("=" * 60)

    all_errors = []

    # 1. Check base artifacts
    missing = check_artifacts_exist(root, REQUIRED_BASE_ARTIFACTS)
    if missing:
        all_errors.extend([f"Missing artifact: {m}" for m in missing])
    else:
        print("[PASS] All required base artifacts exist")

    # 2. Check manifest
    manifest_errors = check_manifest(root)
    if manifest_errors:
        all_errors.extend(manifest_errors)
    else:
        print("[PASS] manifest.json is valid")

    # 3. Check quality score
    score_errors = check_quality_score(root)
    if score_errors:
        all_errors.extend(score_errors)
    else:
        print("[PASS] quality_score.json reports pass")

    # 4. Check README verification
    readme_errors = check_readme_has_verification(root)
    if readme_errors:
        all_errors.extend(readme_errors)
    else:
        print("[PASS] README.md contains verification instructions")

    # Summary
    print("=" * 60)
    if all_errors:
        print(f"FAILED — {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  ✗ {err}")
        return 1
    else:
        print("PASSED — All checks green")
        return 0


def main():
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = os.getcwd()

    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    exit_code = run_gate(root)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
