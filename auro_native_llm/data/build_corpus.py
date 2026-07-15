from __future__ import annotations

import argparse

from auro_native_llm.receipt import emit_receipt, load_json_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Auro corpus mixture config and emit a scaffold receipt.")
    parser.add_argument("--config", required=True, help="Path to data mixture config JSON")
    args = parser.parse_args()
    config = load_json_config(args.config)
    required = ["schema", "mixture_id", "target_tokens", "sources", "required_filters"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing data mixture config keys: {', '.join(missing)}")
    emit_receipt("data_mixture", args.config, config)


if __name__ == "__main__":
    main()
