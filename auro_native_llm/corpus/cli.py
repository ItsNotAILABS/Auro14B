"""CLI: harvest multi-repo corpus, stats, search, feed training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="auro-corpus", description="MESIE multi-repo corpus harvest")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="Harvest local + GitHub clones")
    h.add_argument("--no-github", action="store_true", help="Local roots only")
    h.add_argument("--max-files", type=int, default=5000)
    h.add_argument("--max-chars", type=int, default=12_000_000)
    h.add_argument("--clone-max", type=int, default=30)
    h.add_argument("--save", default=str(Path.home() / ".auro_corpus" / "index.json"))

    sub.add_parser("stats", help="Show cached index stats")
    s = sub.add_parser("search", help="Search corpus")
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=8)

    t = sub.add_parser("feed-mind", help="Absorb corpus into a mind trainer")
    t.add_argument("--model", default="Auro-2B")
    t.add_argument("--max-docs", type=int, default=200)
    t.add_argument("--train-steps", type=int, default=20)

    c = sub.add_parser("clone-orgs", help="Shallow-clone public org repos into cache")
    c.add_argument("--orgs", default="ItsNotAILABS,FreddyCreates")
    c.add_argument("--max-repos", type=int, default=25)

    args = p.parse_args(argv)

    if args.cmd == "clone-orgs":
        from auro_native_llm.corpus.harvest import materialize_github_org

        out = {}
        for org in args.orgs.split(","):
            org = org.strip()
            paths = materialize_github_org(org, max_repos=args.max_repos)
            out[org] = [str(p) for p in paths]
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "harvest":
        from auro_native_llm.corpus.harvest import harvest_all
        from auro_native_llm.corpus.bridge import get_index

        idx = harvest_all(
            include_github_clones=not args.no_github,
            max_files=args.max_files,
            max_total_chars=args.max_chars,
            clone_max_repos=args.clone_max,
        )
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        idx.save(args.save)
        # refresh process cache
        import auro_native_llm.corpus.bridge as bridge

        bridge._INDEX = idx
        print(json.dumps(idx.stats(), indent=2))
        print(f"saved: {args.save}")
        return 0

    if args.cmd == "stats":
        from auro_native_llm.corpus.bridge import get_index

        idx = get_index(refresh=False, include_github=False)
        print(json.dumps(idx.stats(), indent=2))
        return 0

    if args.cmd == "search":
        from auro_native_llm.corpus.bridge import get_index

        idx = get_index(include_github=False)
        hits = idx.search(args.query, top_k=args.top_k)
        print(json.dumps({"query": args.query, "hits": hits}, indent=2))
        return 0

    if args.cmd == "feed-mind":
        from auro_native_llm.corpus.bridge import get_index
        from auro_native_llm.organism.family import build_mind
        from auro_native_llm.organism.self_train import Experience

        idx = get_index(include_github=True)
        mind = build_mind(args.model, lite=True)
        n = 0
        for d in idx.documents[: args.max_docs]:
            mind.organs.trainer.absorb(
                Experience(
                    text=d.training_block()[:3000],
                    kind=f"corpus_{d.kind}",
                    model_id=args.model,
                    reward=0.82 if d.kind == "doc" else 0.78,
                    meta={"repo": d.repo, "path": d.path},
                )
            )
            n += 1
        report = mind.organs.trainer.train_on_model(mind.language, steps=args.train_steps)
        print(json.dumps({"absorbed": n, "train": report, "corpus": idx.stats()}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
