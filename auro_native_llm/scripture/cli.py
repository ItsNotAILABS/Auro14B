"""CLI for Scriptural Systems Architecture substrate."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="auro-scripture",
        description="Scriptural Systems Architecture — canon, memory, governance, substrate",
    )
    p.add_argument("--canon", default=None, help="Path to canon JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="Substrate + canon health")
    sub.add_parser("canon", help="Print canon summary")

    g = sub.add_parser("generate", help="Governed generate")
    g.add_argument("--model", default="Auro-2B")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=64)

    t = sub.add_parser("train-step", help="Governed single train step")
    t.add_argument("--model", default="Auro-2B")
    t.add_argument("--text", required=True)

    d = sub.add_parser("dispatch", help="Governed multi-embedded dispatch")
    d.add_argument("--parent", default="Auro-14B")
    d.add_argument("--role", required=True)
    d.add_argument("--intent", required=True)

    c = sub.add_parser("claim", help="Try a doctrine claim (often refused without receipts)")
    c.add_argument("--model", default="Auro-14B")
    c.add_argument("--statement", required=True)
    c.add_argument("--claims-trained", action="store_true")

    sub.add_parser("memory", help="Memory stats")
    pers = sub.add_parser("persist", help="Persist memory + receipts")
    pers.add_argument("--memory-path", default="deliverables/auro_scripture/memory.json")
    pers.add_argument("--receipts-path", default="deliverables/auro_scripture/receipts.jsonl")

    loop = sub.add_parser("loop", help="Structured cognitive loop (retrieve→cognize→validate→act→memory)")
    loop.add_argument("--model", default="Auro-2B")
    loop.add_argument("--intent", required=True)
    loop.add_argument("--risk", type=float, default=None, help="Override action_risk 0..1")
    loop.add_argument("--max-tokens", type=int, default=48)

    dual = sub.add_parser("dual", help="Export doctrine as constitutional prompt + symbolic bundle")
    hy = sub.add_parser("hybrid", help="Run soft CAI + hard symbolic on a draft")
    hy.add_argument("--intent", required=True)
    hy.add_argument("--draft", required=True)
    hy.add_argument("--risk", type=float, default=0.2)

    args = p.parse_args(argv)

    from auro_native_llm.scripture.substrate import ScripturalSubstrate
    from auro_native_llm.scripture.canon import load_canon

    if args.cmd == "canon":
        canon = load_canon(args.canon)
        print(json.dumps(canon.to_dict(), indent=2)[:6000])
        return 0

    sub_rt = ScripturalSubstrate(canon_path=args.canon)

    if args.cmd == "health":
        print(json.dumps(sub_rt.health(), indent=2))
        return 0

    if args.cmd == "generate":
        r = sub_rt.generate(args.prompt, model_id=args.model, max_new_tokens=args.max_tokens)
        print(json.dumps(r.to_dict(), indent=2)[:8000])
        return 0 if r.ok else 1

    if args.cmd == "train-step":
        r = sub_rt.train_step_governed(args.model, args.text)
        print(json.dumps(r.to_dict(), indent=2)[:8000])
        return 0 if r.ok else 1

    if args.cmd == "dispatch":
        r = sub_rt.dispatch(args.role, args.intent, parent_model_id=args.parent)
        print(json.dumps(r.to_dict(), indent=2)[:8000])
        return 0 if r.ok else 1

    if args.cmd == "claim":
        r = sub_rt.claim(
            args.statement,
            model_id=args.model,
            claims_trained_checkpoint=args.claims_trained,
        )
        print(json.dumps(r.to_dict(), indent=2)[:8000])
        return 0 if r.ok else 1

    if args.cmd == "memory":
        print(json.dumps(sub_rt.memory.stats(), indent=2))
        return 0

    if args.cmd == "persist":
        mp = sub_rt.persist_memory(args.memory_path)
        rp = sub_rt.save_receipts(args.receipts_path)
        print(json.dumps({"memory": mp, "receipts": rp}, indent=2))
        return 0

    if args.cmd == "loop":
        from auro_native_llm.scripture.agent_loop import StructuredCognitiveLoop

        agent = StructuredCognitiveLoop(canon=sub_rt.canon, memory=sub_rt.memory, lite=True)
        result = agent.run(
            args.intent,
            model_id=args.model,
            action_risk=args.risk,
            max_new_tokens=args.max_tokens,
        )
        print(json.dumps(result.to_dict(), indent=2)[:12000])
        return 0 if result.ok else 1

    if args.cmd == "dual":
        from auro_native_llm.scripture.constitutional import ConstitutionalEngine

        eng = ConstitutionalEngine(sub_rt.canon)
        print(json.dumps(eng.dual_export(), indent=2)[:12000])
        return 0

    if args.cmd == "hybrid":
        from auro_native_llm.scripture.constitutional import hybrid_pipeline

        out = hybrid_pipeline(
            args.intent,
            args.draft,
            canon=sub_rt.canon,
            facts={"action_risk": args.risk, "no_human_approval": True, "op": "generate"},
        )
        print(json.dumps(out, indent=2)[:8000])
        return 0 if out.get("allowed") else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
