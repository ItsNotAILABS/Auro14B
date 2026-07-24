#!/usr/bin/env python3
"""Print the exact Auro-4B architecture and parameter estimate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auro_native_llm.model.auro4b import build_auro4b_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("proxy", "full"), default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_auro4b_config(args.scale)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
