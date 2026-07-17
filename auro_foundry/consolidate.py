from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from .benchmarks import BenchmarkRunner, LmEvalBridge, STANDARD_TASK_PROFILES, built_in_probes, load_cases
from .coding_harness import CodingHarness, built_in_smoke_tasks, load_tasks
from .execution import ExecutionHarness, ExecutionPolicy
from .federation import FederationManifest
from .generation import TextGenerator
from .server import serve
from .workers import WorkerRegistry


def emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def write_manifests(workspace: str | Path) -> dict:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    federation = FederationManifest()
    workers = WorkerRegistry()
    federation_path = federation.write(root / "federation-manifest.json")
    worker_path = workers.write(root / "worker-registry.json")
    return {
        "federation": str(federation_path),
        "workers": str(worker_path),
        "repository_counts": federation.to_dict()["counts"],
        "worker_counts": workers.to_dict()["counts"],
    }


def run_native(args) -> dict:
    generator = TextGenerator(args.checkpoint, device=args.device)
    cases = load_cases(args.cases) if args.cases else built_in_probes()
    output = args.output or str(Path(args.workspace) / "benchmarks" / "native-receipt.json")
    return BenchmarkRunner(generator).run(cases, output_path=output)


def run_code(args) -> dict:
    generator = TextGenerator(args.checkpoint, device=args.device)
    tasks = load_tasks(args.tasks) if args.tasks else built_in_smoke_tasks()
    harness = CodingHarness(
        lambda prompt: generator.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=0.0),
        ExecutionHarness(ExecutionPolicy(timeout_seconds=args.timeout)),
    )
    output = args.output or str(Path(args.workspace) / "benchmarks" / "coding-receipt.json")
    return harness.run(tasks, output_path=output)


def run_official(args) -> dict:
    tasks: list[str] = []
    for value in args.task:
        tasks.extend(STANDARD_TASK_PROFILES.get(value, (value,)))
    tasks = list(dict.fromkeys(tasks))
    return LmEvalBridge(args.executable).run(
        base_url=args.base_url,
        model_id=args.model_id,
        tasks=tasks,
        output_dir=args.output,
        limit=args.limit,
        chat=not args.completions,
        batch_size=args.batch_size,
    )


def solidify(args) -> None:
    root = Path(args.workspace).resolve()
    manifests = write_manifests(root)
    native = run_native(args)
    coding = run_code(args)
    registry = WorkerRegistry()
    receipt = {
        "schema": "auro.solidification.v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifests": manifests,
        "native_benchmark": native["summary"],
        "coding_benchmark": coding["summary"],
        "worker_plans": {
            "benchmark": registry.plan("benchmark"),
            "code": registry.plan("code"),
            "release": registry.plan("release"),
        },
        "talk_url": f"http://{args.host}:{args.port}",
    }
    target = root / "solidification-receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    emit(receipt)
    if args.serve:
        if args.open_browser:
            webbrowser.open(receipt["talk_url"])
        serve(args.checkpoint, host=args.host, port=args.port, device=args.device)


def main() -> None:
    parser = argparse.ArgumentParser(prog="auro-consolidate")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest"); p.add_argument("--workspace", default="artifacts/auro-foundry"); p.set_defaults(run=lambda a: emit(write_manifests(a.workspace)))
    p = sub.add_parser("workers"); p.add_argument("--capability", default="benchmark"); p.set_defaults(run=lambda a: emit({"plan": WorkerRegistry().plan(a.capability), "registry": WorkerRegistry().to_dict()}))
    p = sub.add_parser("benchmark"); p.add_argument("--checkpoint", required=True); p.add_argument("--cases"); p.add_argument("--output"); p.add_argument("--workspace", default="artifacts/auro-foundry"); p.add_argument("--device", default="auto"); p.set_defaults(run=lambda a: emit(run_native(a)))
    p = sub.add_parser("code-eval"); p.add_argument("--checkpoint", required=True); p.add_argument("--tasks"); p.add_argument("--output"); p.add_argument("--workspace", default="artifacts/auro-foundry"); p.add_argument("--device", default="auto"); p.add_argument("--timeout", type=float, default=5.0); p.add_argument("--max-new-tokens", type=int, default=512); p.set_defaults(run=lambda a: emit(run_code(a)))
    p = sub.add_parser("official"); p.add_argument("--base-url", default="http://127.0.0.1:8090"); p.add_argument("--model-id", default="Auro"); p.add_argument("--task", action="append", default=["leaderboard"]); p.add_argument("--output", default="artifacts/auro-foundry/benchmarks/lm-eval"); p.add_argument("--limit", type=float); p.add_argument("--batch-size", default="1"); p.add_argument("--completions", action="store_true"); p.add_argument("--executable", default="lm-eval"); p.set_defaults(run=lambda a: emit(run_official(a)))
    p = sub.add_parser("solidify"); p.add_argument("--checkpoint", required=True); p.add_argument("--workspace", default="artifacts/auro-foundry"); p.add_argument("--cases"); p.add_argument("--tasks"); p.add_argument("--output"); p.add_argument("--device", default="auto"); p.add_argument("--timeout", type=float, default=5.0); p.add_argument("--max-new-tokens", type=int, default=512); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8090); p.add_argument("--serve", action="store_true"); p.add_argument("--open-browser", action="store_true"); p.set_defaults(run=solidify)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
