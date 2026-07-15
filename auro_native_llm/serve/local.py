from __future__ import annotations

import argparse

from auro_native_llm.receipt import emit_receipt, load_json_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Auro serving contract and emit a scaffold receipt.")
    parser.add_argument("--config", required=True, help="Path to serving contract JSON")
    args = parser.parse_args()
    config = load_json_config(args.config)
    required = ["schema", "server", "routes", "required_receipts"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing serving config keys: {', '.join(missing)}")
    emit_receipt("serving_contract", args.config, config)


if __name__ == "__main__":
    main()
