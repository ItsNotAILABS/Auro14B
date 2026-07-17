from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from .config import TrainConfig, preset
from .corpus import CorpusBuilder, RepositorySource
from .dataset import prepare_token_dataset
from .generation import TextGenerator
from .mesie_bridge import run_training_governed
from .server import serve
from .tokenizer import AuroBPETokenizer, iter_jsonl_text
from .training import train


def _write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _build_corpus(args) -> dict:
    builder = CorpusBuilder(args.output_dir, max_file_bytes=args.max_file_bytes)
    repositories: list[RepositorySource] = []
    if args.org:
        repositories.extend(builder.discover_org(args.org, include_private=not args.public_only))
    repositories.extend(RepositorySource(repo) for repo in args.repo)
    manifest = builder.build(repositories=repositories, local_roots=args.local_root, corpus_name=args.name)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _train_tokenizer(args) -> dict:
    tokenizer = AuroBPETokenizer(tokenizer_id=args.tokenizer_id)
    report = tokenizer.train(
        iter_jsonl_text(args.corpus),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_training_bytes=args.max_training_bytes,
    )
    tokenizer.save(args.output)
    payload = report.to_dict() | {"path": str(Path(args.output).resolve())}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _prepare_dataset(args) -> dict:
    manifest = prepare_token_dataset(
        args.corpus,
        args.tokenizer,
        args.output_dir,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _init_config(args) -> dict:
    model = preset(args.preset)
    if args.vocab_size:
        model = type(model)(**(model.to_dict() | {"vocab_size": args.vocab_size, "estimated_parameters": None}))
    config = TrainConfig(
        model=model,
        run_name=args.run_name,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        tokenizer_path=args.tokenizer,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        eval_interval=max(1, args.eval_interval),
        checkpoint_interval=max(1, args.checkpoint_interval),
        device=args.device,
        precision=args.precision,
        strategy=args.strategy,
    )
    config.save(args.output)
    payload = config.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _run_all(args) -> dict:
    root = Path(args.workspace).resolve()
    corpus_dir = root / "corpus"
    tokenizer_path = root / "tokenizer.json"
    dataset_dir = root / "dataset"
    config_path = root / "train-config.json"
    builder = CorpusBuilder(corpus_dir)
    repositories = builder.discover_org(args.org, include_private=not args.public_only) if args.org else []
    repositories.extend(RepositorySource(repo) for repo in args.repo)
    corpus = builder.build(repositories=repositories, local_roots=args.local_root, corpus_name="auro-owned-corpus")
    tokenizer = AuroBPETokenizer()
    tokenizer_report = tokenizer.train(iter_jsonl_text(corpus_dir / "corpus.jsonl"), vocab_size=args.vocab_size, max_training_bytes=args.max_training_bytes)
    tokenizer.save(tokenizer_path)
    dataset = prepare_token_dataset(corpus_dir / "corpus.jsonl", tokenizer_path, dataset_dir, validation_fraction=args.validation_fraction)
    model = preset(args.preset)
    model_dict = model.to_dict()
    model_dict.pop("estimated_parameters", None)
    model_dict["vocab_size"] = tokenizer.vocab_size
    model = type(model).from_dict(model_dict)
    config = TrainConfig(
        model=model,
        run_name=args.run_name,
        output_dir=str(root / "runs"),
        dataset_dir=str(dataset_dir),
        tokenizer_path=str(tokenizer_path),
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        eval_interval=max(1, args.eval_interval),
        checkpoint_interval=max(1, args.checkpoint_interval),
        device=args.device,
        precision=args.precision,
    )
    config.save(config_path)
    result = train(config)
    receipt = {
        "schema": "auro.foundry.end_to_end_receipt.v1",
        "corpus": corpus,
        "tokenizer": tokenizer_report.to_dict(),
        "dataset": dataset,
        "training": result.to_dict(),
        "config_path": str(config_path),
    }
    _write_json(root / "end-to-end-receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auro-foundry")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("build-corpus", help="clone/scan repos and build a redacted corpus")
    corpus.add_argument("--org")
    corpus.add_argument("--repo", action="append", default=[])
    corpus.add_argument("--local-root", action="append", default=[])
    corpus.add_argument("--public-only", action="store_true")
    corpus.add_argument("--output-dir", default="artifacts/auro-foundry/corpus")
    corpus.add_argument("--name", default="auro-owned-corpus")
    corpus.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    corpus.set_defaults(func=_build_corpus)

    tokenizer = sub.add_parser("train-tokenizer")
    tokenizer.add_argument("--corpus", required=True)
    tokenizer.add_argument("--output", default="artifacts/auro-foundry/tokenizer.json")
    tokenizer.add_argument("--tokenizer-id", default="auro-bpe-v1")
    tokenizer.add_argument("--vocab-size", type=int, default=8192)
    tokenizer.add_argument("--min-frequency", type=int, default=2)
    tokenizer.add_argument("--max-training-bytes", type=int, default=64 * 1024 * 1024)
    tokenizer.set_defaults(func=_train_tokenizer)

    dataset = sub.add_parser("prepare-dataset")
    dataset.add_argument("--corpus", required=True)
    dataset.add_argument("--tokenizer", required=True)
    dataset.add_argument("--output-dir", default="artifacts/auro-foundry/dataset")
    dataset.add_argument("--validation-fraction", type=float, default=0.01)
    dataset.set_defaults(func=_prepare_dataset)

    init = sub.add_parser("init-config")
    init.add_argument("--preset", default="micro", choices=["micro", "local", "14b", "206.7b"])
    init.add_argument("--vocab-size", type=int)
    init.add_argument("--run-name", default="auro-local-run")
    init.add_argument("--output", default="artifacts/auro-foundry/train-config.json")
    init.add_argument("--output-dir", default="artifacts/auro-foundry/runs")
    init.add_argument("--dataset-dir", default="artifacts/auro-foundry/dataset")
    init.add_argument("--tokenizer", default="artifacts/auro-foundry/tokenizer.json")
    init.add_argument("--sequence-length", type=int, default=256)
    init.add_argument("--micro-batch-size", type=int, default=2)
    init.add_argument("--gradient-accumulation-steps", type=int, default=4)
    init.add_argument("--max-steps", type=int, default=100)
    init.add_argument("--learning-rate", type=float, default=3e-4)
    init.add_argument("--eval-interval", type=int, default=25)
    init.add_argument("--checkpoint-interval", type=int, default=25)
    init.add_argument("--device", default="auto")
    init.add_argument("--precision", default="auto")
    init.add_argument("--strategy", default="auto")
    init.set_defaults(func=_init_config)

    training = sub.add_parser("train")
    training.add_argument("--config", required=True)
    training.set_defaults(func=lambda args: print(json.dumps(train(TrainConfig.load(args.config)).to_dict(), indent=2)))

    resume = sub.add_parser("resume")
    resume.add_argument("--config", required=True)
    resume.add_argument("--checkpoint", required=True)
    def _resume(args):
        config = TrainConfig.load(args.config)
        payload = config.to_dict(); payload["model"].pop("estimated_parameters", None); payload["resume_from"] = args.checkpoint
        result = train(TrainConfig.from_dict(payload))
        print(json.dumps(result.to_dict(), indent=2))
    resume.set_defaults(func=_resume)

    governed = sub.add_parser("train-governed")
    governed.add_argument("--config", required=True)
    governed.add_argument("--workdir", default=".")
    governed.add_argument("--receipt-dir", default="artifacts/auro-foundry/mesie-receipts")
    governed.add_argument("--timeout", type=float)
    governed.set_defaults(func=lambda args: print(json.dumps(run_training_governed(args.config, workdir=args.workdir, receipt_dir=args.receipt_dir, timeout_seconds=args.timeout).__dict__, indent=2)))

    generate = sub.add_parser("generate")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.set_defaults(func=lambda args: print(TextGenerator(args.checkpoint, device=args.device).generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)))

    server = sub.add_parser("serve")
    server.add_argument("--checkpoint", required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8090)
    server.add_argument("--device", default="auto")
    server.add_argument("--open-browser", action="store_true")
    def _serve(args):
        if args.open_browser:
            webbrowser.open(f"http://{args.host}:{args.port}")
        serve(args.checkpoint, host=args.host, port=args.port, device=args.device)
    server.set_defaults(func=_serve)

    all_cmd = sub.add_parser("all", help="build corpus, tokenizer, dataset, train, and emit receipts")
    all_cmd.add_argument("--workspace", default="artifacts/auro-foundry")
    all_cmd.add_argument("--org", default="ItsNotAILABS")
    all_cmd.add_argument("--repo", action="append", default=[])
    all_cmd.add_argument("--local-root", action="append", default=[])
    all_cmd.add_argument("--public-only", action="store_true")
    all_cmd.add_argument("--preset", default="micro", choices=["micro", "local", "14b", "206.7b"])
    all_cmd.add_argument("--run-name", default="auro-owned-run")
    all_cmd.add_argument("--vocab-size", type=int, default=4096)
    all_cmd.add_argument("--max-training-bytes", type=int, default=64 * 1024 * 1024)
    all_cmd.add_argument("--validation-fraction", type=float, default=0.01)
    all_cmd.add_argument("--sequence-length", type=int, default=128)
    all_cmd.add_argument("--micro-batch-size", type=int, default=2)
    all_cmd.add_argument("--gradient-accumulation-steps", type=int, default=2)
    all_cmd.add_argument("--max-steps", type=int, default=20)
    all_cmd.add_argument("--eval-interval", type=int, default=10)
    all_cmd.add_argument("--checkpoint-interval", type=int, default=10)
    all_cmd.add_argument("--device", default="auto")
    all_cmd.add_argument("--precision", default="auto")
    all_cmd.set_defaults(func=_run_all)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
