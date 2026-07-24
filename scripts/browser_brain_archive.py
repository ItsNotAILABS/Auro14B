#!/usr/bin/env python3
"""Build, search, inspect, or export the Browser-Brain conversation archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from auro_native_llm.browser_brain.service import BrowserBrainService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--output", default="artifacts/browser-brain/conversation-index.json")

    listing = sub.add_parser("list")
    listing.add_argument("--query", default="")
    listing.add_argument("--limit", type=int, default=100)

    show = sub.add_parser("show")
    show.add_argument("archive_id")

    timeline = sub.add_parser("timeline")
    timeline.add_argument("archive_id")

    continuation = sub.add_parser("continuation")
    continuation.add_argument("archive_id")
    continuation.add_argument("--max-characters", type=int, default=24000)

    args = parser.parse_args()
    service = BrowserBrainService(args.root)

    if args.command == "build":
        payload = service.index()
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "list":
        payload = service.list(args.query, args.limit)
    elif args.command == "show":
        payload = service.get(args.archive_id)
    elif args.command == "timeline":
        payload = service.timeline(args.archive_id)
    else:
        payload = service.continuation_context(args.archive_id, args.max_characters)

    if payload is None:
        print(json.dumps({"error": "conversation not found"}, indent=2))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
