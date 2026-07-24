"""CLI for Auro work agents + chrome tools."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="auro-work", description="Auro native work agent (Chrome/code/reason)")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("run", help="Run work objective with tools")
    w.add_argument("--objective", required=True)
    w.add_argument("--model", default="Auro-2B")
    w.add_argument("--real-chrome", action="store_true")
    w.add_argument("--no-scripture", action="store_true")

    c = sub.add_parser("chrome", help="Chrome CDP action")
    c.add_argument("--action", default="dom", choices=["navigate", "dom", "eval", "type", "click", "health"])
    c.add_argument("--url", default="https://example.com")
    c.add_argument("--js", default="document.title")
    c.add_argument("--text", default="")
    c.add_argument("--x", type=float, default=10)
    c.add_argument("--y", type=float, default=10)
    c.add_argument("--real-chrome", action="store_true")

    r = sub.add_parser("reason", help="Reasoning mode")
    r.add_argument("--topic", required=True)

    code = sub.add_parser("code", help="Coding mode")
    code.add_argument("--task", required=True)

    args = p.parse_args(argv)

    if args.cmd == "run":
        from auro_native_llm.work.agent import WorkAgent

        agent = WorkAgent(
            model_id=args.model,
            chrome_mock=not args.real_chrome,
            chrome_auto_start=args.real_chrome,
            use_scripture=not args.no_scripture,
        )
        result = agent.run(args.objective)
        print(json.dumps(result.to_dict(), indent=2)[:12000])
        return 0 if result.ok else 1

    if args.cmd == "chrome":
        from auro_native_llm.chrome.tools import ChromeToolbelt

        belt = ChromeToolbelt(mock=not args.real_chrome, auto_start=args.real_chrome)
        if args.action == "health":
            print(json.dumps(belt.health(), indent=2))
        elif args.action == "navigate":
            print(json.dumps(belt.navigate(args.url), indent=2))
        elif args.action == "dom":
            print(json.dumps(belt.dom(), indent=2)[:8000])
        elif args.action == "eval":
            print(json.dumps(belt.evaluate(args.js), indent=2))
        elif args.action == "type":
            print(json.dumps(belt.type_text(args.text), indent=2))
        elif args.action == "click":
            print(json.dumps(belt.click(args.x, args.y), indent=2))
        return 0

    if args.cmd == "reason":
        from auro_native_llm.work.agent import WorkAgent

        print(json.dumps(WorkAgent().reason(args.topic), indent=2)[:6000])
        return 0

    if args.cmd == "code":
        from auro_native_llm.work.agent import WorkAgent

        print(json.dumps(WorkAgent().code(args.task), indent=2)[:6000])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
