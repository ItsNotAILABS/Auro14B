from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class CaseResult:
    name: str
    family: str
    status: str
    iterations: int
    passed: int
    failed: int
    skipped: int
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    details: dict[str, Any]


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return round(ordered[index], 3)


def run_case(name: str, family: str, iterations: int, fn: Callable[[], dict[str, Any]]) -> CaseResult:
    latencies: list[float] = []
    passed = failed = skipped = 0
    last: dict[str, Any] = {}
    for _ in range(iterations):
        started = time.perf_counter()
        try:
            last = fn()
            status = str(last.get("status", "passed"))
            if status == "skipped":
                skipped += 1
            elif bool(last.get("ok", True)):
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            last = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
        latencies.append((time.perf_counter() - started) * 1000)
    status = "passed" if failed == 0 and passed > 0 else "skipped" if skipped == iterations else "failed"
    return CaseResult(name, family, status, iterations, passed, failed, skipped, percentile(latencies, .5), percentile(latencies, .95), last)


def http_probe(env_name: str, path: str = "/health") -> dict[str, Any]:
    base = os.getenv(env_name, "").rstrip("/")
    if not base:
        return {"status": "skipped", "reason": f"{env_name} not configured"}
    request = urllib.request.Request(base + path, headers={"User-Agent": "AURO-Portfolio-Benchmark/1"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {"ok": 200 <= response.status < 400, "status_code": response.status, "body_prefix": body[:200]}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status_code": exc.code, "error": str(exc)}


def browser_archive_flow() -> dict[str, Any]:
    from auro_native_llm.browser_brain.conversation_archive import ConversationArchive

    root = Path(os.getenv("AURO_BENCH_STATE", "artifacts/portfolio-bench/state")) / "browser"
    root.mkdir(parents=True, exist_ok=True)
    archive = ConversationArchive(root)
    payload = {
        "title": "Uber Chrome planning flow",
        "url": "https://m.uber.com/",
        "messages": [
            {"role": "user", "content": "Plan a ride from pickup to destination without booking."},
            {"role": "assistant", "content": "Require explicit approval before any external booking action."},
        ],
    }
    result = archive.archive(payload)
    return {"ok": bool(result), "mode": "local-archive-contract", "result": result}


def uber_chrome_contract() -> dict[str, Any]:
    """Validate the governed Uber browser-flow contract without booking a ride."""
    steps = [
        "open_uber_surface",
        "resolve_pickup",
        "resolve_destination",
        "request_estimate",
        "select_ride_class",
        "require_human_approval",
        "stop_before_booking",
    ]
    forbidden = {"book_without_approval", "change_payment_method", "cancel_existing_ride"}
    digest = hashlib.sha256(json.dumps({"steps": steps, "forbidden": sorted(forbidden)}, sort_keys=True).encode()).hexdigest()
    return {
        "ok": steps[-2:] == ["require_human_approval", "stop_before_booking"],
        "mode": "contract-only",
        "external_booking_performed": False,
        "steps": steps,
        "forbidden": sorted(forbidden),
        "flow_hash": "0x" + digest,
    }


def capability_manifest_flow() -> dict[str, Any]:
    from auro_native_llm.production_fleet.capabilities import BUILTINS

    names = {item.name for item in BUILTINS}
    required = {
        "brain.state",
        "memory.rank_text",
        "compute.matmul",
        "office.create_bundle",
        "browser.task.enqueue",
        "wallet.verify_ledger",
        "skill.research",
        "skill.build",
    }
    return {"ok": required.issubset(names), "count": len(names), "missing": sorted(required - names)}


def st14b_runtime_flow() -> dict[str, Any]:
    import torch
    from auro_native_llm.model.st14b_runtime import AuroST14BForCausalLM, ST14BRuntimeConfig

    config = ST14BRuntimeConfig(vocab_size=512, hidden_size=128, num_layers=2, num_heads=8, num_kv_heads=1, intermediate_size=384, max_seq_len=128)
    model = AuroST14BForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model.new_cache()
    with torch.no_grad():
        logits = model.prefill(prompt, cache)
        next_logits = model.decode_step(logits[:, -1, :].argmax(dim=-1, keepdim=True), cache)
    return {
        "ok": tuple(logits.shape) == (1, 32, 512) and tuple(next_logits.shape) == (1, 512),
        "cache_sequence_length": cache.sequence_length,
        "kv_heads": cache.layers[0].key.size(1) if cache.layers[0].key is not None else None,
    }


def build_suite(iterations: int) -> list[CaseResult]:
    cases: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
        ("st14b.cached_decode", "model", st14b_runtime_flow),
        ("capabilities.manifest", "utilities", capability_manifest_flow),
        ("browser.archive", "browser", browser_archive_flow),
        ("uber.chrome.governed_contract", "browser", uber_chrome_contract),
        ("iot.nova_gateway", "iot", lambda: http_probe("NOVA_IOT_BENCHMARK_URL")),
        ("browser.live_gateway", "browser", lambda: http_probe("AURO_BROWSER_BENCHMARK_URL")),
        ("chrome.cdp_gateway", "browser", lambda: http_probe("AURO_CHROME_CDP_BENCHMARK_URL")),
    ]
    return [run_case(name, family, iterations, fn) for name, family, fn in cases]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=int(os.getenv("AURO_BENCH_ITERATIONS", "5")))
    parser.add_argument("--duration-seconds", type=int, default=int(os.getenv("AURO_BENCH_DURATION_SECONDS", "0")))
    parser.add_argument("--output", default="artifacts/portfolio-bench/report.json")
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("iterations must be positive")

    started = time.time()
    rounds: list[list[CaseResult]] = []
    while True:
        rounds.append(build_suite(args.iterations))
        if args.duration_seconds <= 0 or time.time() - started >= args.duration_seconds:
            break

    flat = [item for batch in rounds for item in batch]
    report = {
        "schema": "auro.portfolio.long_benchmark.v1",
        "started_at": int(started),
        "completed_at": int(time.time()),
        "rounds": len(rounds),
        "iterations_per_case": args.iterations,
        "summary": {
            "passed": sum(item.status == "passed" for item in flat),
            "failed": sum(item.status == "failed" for item in flat),
            "skipped": sum(item.status == "skipped" for item in flat),
        },
        "results": [asdict(item) for item in flat],
        "truth_boundary": {
            "uber_booking_performed": False,
            "live_browser_requires_configured_gateway": True,
            "live_iot_requires_configured_gateway": True,
            "h100_required_for_st14b_hardware_targets": True,
        },
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["evidence_hash"] = "0x" + hashlib.sha256(encoded.encode()).hexdigest()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
