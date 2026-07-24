"""CLI: succotash engines/models registry + multi-area corpus."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="auro-succotash",
        description="FreddyCreates/potential-succotash engines, models, agents, training corpus",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ensure", help="Clone/locate potential-succotash")
    sub.add_parser("summary", help="Registry counts")
    sub.add_parser("models", help="List models LLM can use")
    sub.add_parser("engines", help="List engines")
    sub.add_parser("agents", help="List agents")
    sub.add_parser("areas", help="List training areas")

    r = sub.add_parser("route", help="Route a task to engine/model/agent")
    r.add_argument("task")

    h = sub.add_parser("harvest", help="Harvest multi-area training corpus")
    h.add_argument("--max-files", type=int, default=800)
    h.add_argument("--max-chars", type=int, default=2_000_000)
    h.add_argument("--areas", nargs="*", default=None)

    c = sub.add_parser("corpus-texts", help="Emit training texts (count + sample)")
    c.add_argument("--max-chars", type=int, default=500_000)
    c.add_argument("--sample", type=int, default=2)

    args = p.parse_args(argv)

    if args.cmd == "ensure":
        from auro_native_llm.succotash.paths import ensure_succotash, SUCCOTASH_URL

        root = ensure_succotash()
        print(json.dumps({"ok": True, "root": str(root), "url": SUCCOTASH_URL}, indent=2))
        return 0

    if args.cmd == "summary":
        from auro_native_llm.succotash.registry import load_registry

        reg = load_registry()
        print(json.dumps(reg.summary(), indent=2))
        return 0

    if args.cmd == "models":
        from auro_native_llm.succotash.registry import load_registry

        print(json.dumps(load_registry().models_for_llm(), indent=2)[:12000])
        return 0

    if args.cmd == "engines":
        from auro_native_llm.succotash.registry import load_registry

        print(json.dumps(load_registry().engines_for_llm(), indent=2))
        return 0

    if args.cmd == "agents":
        from auro_native_llm.succotash.registry import load_registry

        print(json.dumps(load_registry().agents_for_llm(), indent=2))
        return 0

    if args.cmd == "areas":
        from auro_native_llm.succotash.corpus import area_manifest

        print(json.dumps(area_manifest(), indent=2))
        return 0

    if args.cmd == "route":
        from auro_native_llm.succotash.router import route_task

        print(json.dumps(route_task(args.task), indent=2))
        return 0

    if args.cmd == "harvest":
        from auro_native_llm.succotash.corpus import harvest_succotash_corpus

        idx = harvest_succotash_corpus(
            areas=args.areas,
            max_files=args.max_files,
            max_total_chars=args.max_chars,
        )
        by_kind: dict = {}
        for d in idx.documents:
            by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
        print(
            json.dumps(
                {
                    "ok": True,
                    "docs": len(idx.documents),
                    "chars": idx.total_chars,
                    "by_kind": by_kind,
                    "roots": idx.roots,
                    "sample_paths": [d.path for d in idx.documents[:15]],
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "corpus-texts":
        from auro_native_llm.succotash.corpus import collect_all_area_texts

        texts = collect_all_area_texts(max_chars=args.max_chars)
        print(
            json.dumps(
                {
                    "texts": len(texts),
                    "chars": sum(len(t) for t in texts),
                    "samples": [t[:300] for t in texts[: args.sample]],
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
