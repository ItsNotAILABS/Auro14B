"""CLI for Auro native model family — MESIE compute plane."""

from __future__ import annotations

import argparse
import json
import sys

from auro_native_llm.family import emit_family_receipt, list_model_ids, load_family, validate_family
from auro_native_llm.mesie_compute import get_compute_plane
from auro_native_llm.native_runtime import AuroNativeRuntime
from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter
from auro_native_llm.types import SubAgentRole


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="auro-family",
        description=(
            "Auro native LLM family (2B/4B/8B/14B/100B) — multi-embedded sub-agents, "
            "MESIE compute plane"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to auro_family.json (default: native_llm/configs/auro_family.json)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List model lanes in the family")
    sub.add_parser("validate", help="Validate family charter + lane configs")
    sub.add_parser("receipt", help="Emit family charter scaffold receipt")
    sub.add_parser("compute", help="Show MESIE compute plane health / capabilities")

    p_show = sub.add_parser("show", help="Show one lane")
    p_show.add_argument("model_id", help="e.g. Auro-14B")

    p_dispatch = sub.add_parser(
        "dispatch",
        help="Dispatch multi-embedded sub-agent (route only, or --native for MESIE generate)",
    )
    p_dispatch.add_argument("--parent", default="Auro-14B", help="Parent model_id")
    p_dispatch.add_argument("--role", required=True, help="SubAgentRole value")
    p_dispatch.add_argument("--intent", required=True, help="Task intent text")
    p_dispatch.add_argument("--ghost", action="store_true", help="Try MESIE ghost spawn (route only)")
    p_dispatch.add_argument(
        "--native",
        action="store_true",
        default=True,
        help="Run MESIE-native generation on child lane (default: true)",
    )
    p_dispatch.add_argument(
        "--route-only",
        action="store_true",
        help="Only route; do not run MESIE native generate",
    )

    p_council = sub.add_parser("council", help="Multi-role council under parent (MESIE native)")
    p_council.add_argument("--parent", default="Auro-14B")
    p_council.add_argument("--intent", required=True)
    p_council.add_argument("--route-only", action="store_true")

    p_gen = sub.add_parser("generate", help="MESIE-native generate on one Auro lane")
    p_gen.add_argument("--model", default="Auro-14B", help="Auro model_id")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--role", default=None)
    p_gen.add_argument("--max-tokens", type=int, default=256)

    p_embed = sub.add_parser("embed", help="MESIE-native embed")
    p_embed.add_argument("--model", default="Auro-2B")
    p_embed.add_argument("--text", required=True)

    args = parser.parse_args(argv)
    family = load_family(args.config)

    if args.command == "list":
        for lane in family.lanes:
            roles = ",".join(r.value for r in lane.subagent_roles)
            print(
                f"{lane.model_id:12}  params~{lane.parameter_target:>15,}  "
                f"tier={lane.tier.value:13}  embed={lane.can_embed_subagents}  "
                f"compute=MESIE  roles=[{roles}]"
            )
        return 0

    if args.command == "validate":
        errors = validate_family(family)
        if errors:
            print("INVALID", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        plane = get_compute_plane()
        print(
            json.dumps(
                {
                    "ok": True,
                    "models": family.model_ids(),
                    "status": family.status,
                    "compute_plane": "MESIE",
                    "native": True,
                    "capabilities": plane.capabilities,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "receipt":
        emit_family_receipt(args.config)
        return 0

    if args.command == "compute":
        plane = get_compute_plane()
        print(
            json.dumps(
                {
                    "health": plane.health(),
                    "node": plane.discover_node(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "show":
        lane = family.get_lane(args.model_id)
        if lane is None:
            print(f"unknown model_id: {args.model_id}", file=sys.stderr)
            print(f"known: {list_model_ids(args.config)}", file=sys.stderr)
            return 1
        payload = lane.to_dict()
        payload["compute_plane"] = "MESIE"
        payload["native"] = True
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "generate":
        rt = AuroNativeRuntime(parent_model_id=args.model)
        gen = rt.generate(args.prompt, model_id=args.model, role=args.role, max_tokens=args.max_tokens)
        print(json.dumps(gen.to_dict(), indent=2))
        return 0

    if args.command == "embed":
        rt = AuroNativeRuntime(parent_model_id=args.model)
        vec = rt.get_model(args.model).embed(args.text)
        print(
            json.dumps(
                {
                    "model_id": args.model,
                    "compute_plane": "MESIE",
                    "native": True,
                    "dim": len(vec),
                    "embedding": vec,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "dispatch":
        try:
            role = SubAgentRole(args.role)
        except ValueError:
            print(f"unknown role: {args.role}", file=sys.stderr)
            print("roles: " + ", ".join(r.value for r in SubAgentRole), file=sys.stderr)
            return 1

        if args.route_only:
            router = MultiEmbeddedSubAgentRouter(parent_model_id=args.parent, family=family)
            result = router.dispatch(role, args.intent, use_ghost=args.ghost)
            print(json.dumps(result.to_dict(), indent=2))
            return 0 if result.ok else 1

        rt = AuroNativeRuntime(parent_model_id=args.parent)
        result = rt.dispatch(role, args.intent)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1

    if args.command == "council":
        if args.route_only:
            router = MultiEmbeddedSubAgentRouter(parent_model_id=args.parent, family=family)
            results = router.dispatch_council(args.intent)
            print(json.dumps([r.to_dict() for r in results], indent=2))
            return 0 if all(r.ok for r in results) else 1

        rt = AuroNativeRuntime(parent_model_id=args.parent)
        results = rt.council(args.intent)
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0 if all(r.ok for r in results) else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
