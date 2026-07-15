"""Demo internal API bus — register engines and route a validation + match cycle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_reference_record, list_references
from mesie.engines.registry import build_default_registry
from mesie.internal_api.bus import InternalBus


def main() -> None:
    bus = InternalBus()
    registry = build_default_registry(bus)
    names = sorted(registry.names())
    print("Registered engines:", ", ".join(names))
    print(f"Total: {len(names)}")

    ref = load_reference_record(list_references()[0])
    val = bus.request("demo", "validation", "validate", {"record": ref})
    print("\nvalidate ->", val.to_dict())

    match = bus.request(
        "demo",
        "matching",
        "match",
        {"record_a": ref, "record_b": ref},
    )
    print("match(self) composite_score:", match.data.get("composite_score"))


if __name__ == "__main__":
    main()