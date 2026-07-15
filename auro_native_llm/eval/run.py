from __future__ import annotations

import argparse

from auro_native_llm.receipt import emit_receipt, load_json_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Auro evaluation gate config and emit a scaffold receipt.")
    parser.add_argument("--config", required=True, help="Path to eval gate config JSON")
    args = parser.parse_args()
    config = load_json_config(args.config)
    required = ["schema", "eval_suite", "required_gates"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing eval config keys: {', '.join(missing)}")
    emit_receipt("model_eval", args.config, config)


if __name__ == "__main__":
    main()
