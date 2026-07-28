"""Build the provenance-bound Auro + MESIE + Sovereign training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auro_foundry.corpus import CorpusBuilder
from auro_native_llm.sovereign import bind_sovereign


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unversioned-local-source"


def _discover_mesie(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("AURO_MESIE_ROOT"):
        candidates.append(Path(os.environ["AURO_MESIE_ROOT"]))
    candidates.extend([
        ROOT.parent / "Multi-Element-Spectral-Intelligence-Engine-MESIE-",
        Path.home() / "Multi-Element-Spectral-Intelligence-Engine-MESIE-",
    ])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "mesie" / "__init__.py").is_file():
            return resolved
    raise FileNotFoundError("MESIE checkout not found. Set AURO_MESIE_ROOT or pass --mesie-root.")


def _manifest_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "unified_manifest_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Auro14B unified training corpus")
    parser.add_argument("--mesie-root")
    parser.add_argument("--sovereign-root")
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--output", default="artifacts/auro14b-corpus")
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args(argv)

    mesie_root = _discover_mesie(args.mesie_root)
    sovereign = bind_sovereign(args.sovereign_root, required=True)
    assert sovereign is not None
    roots = [ROOT, mesie_root, sovereign.root]
    roots.extend(Path(value).expanduser().resolve() for value in args.source_root)

    output = Path(args.output).expanduser().resolve()
    builder = CorpusBuilder(output, max_file_bytes=args.max_file_bytes)
    manifest = builder.build(
        local_roots=roots,
        corpus_name="auro14b-mesie-sovereign-v1",
    )
    manifest.update({
        "schema": "auro.unified_training_corpus.v1",
        "required_consumers": ["Auro14B", "MESIE", "Sovereign"],
        "source_commits": {
            "Auro14B": _git_commit(ROOT),
            "MESIE": _git_commit(mesie_root),
            "Sovereign": sovereign.commit,
        },
        "sovereign_binding": sovereign.receipt(),
        "claim_boundary": (
            "Corpus inclusion proves source preparation, not gradient consumption. "
            "Training reports must separately record sampled records, steps, metrics, and checkpoint hashes."
        ),
    })
    manifest["unified_manifest_sha256"] = _manifest_hash(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "corpus": manifest["corpus_path"],
        "records": manifest["records"],
        "text_bytes": manifest["text_bytes"],
        "repositories": manifest["repositories"],
        "unified_manifest_sha256": manifest["unified_manifest_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
