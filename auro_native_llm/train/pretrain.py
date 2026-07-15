from __future__ import annotations

import argparse

from auro_native_llm.receipt import emit_receipt, load_json_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Auro pretraining config and emit a scaffold receipt.")
    parser.add_argument("--config", required=True, help="Path to model config JSON")
    args = parser.parse_args()
    config = load_json_config(args.config)
    required = ["schema", "model_id", "status", "parameter_target", "architecture"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing model config keys: {', '.join(missing)}")
    if "not-trained" not in str(config.get("status", "")):
        raise SystemExit("config status must explicitly say not-trained until checkpoint receipts exist")
    emit_receipt("pretrain_config", args.config, config)


if __name__ == "__main__":
    main()
