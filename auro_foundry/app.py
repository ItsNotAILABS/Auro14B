from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from .config import ModelConfig, TrainConfig, preset
from .corpus import CorpusBuilder, RepositorySource
from .dataset import prepare_token_dataset
from .generation import TextGenerator
from .mesie_bridge import run_training_governed
from .server import serve
from .tokenizer import AuroBPETokenizer, iter_jsonl_text
from .training import train


def emit(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def model_with_vocab(name: str, vocab_size: int) -> ModelConfig:
    data = preset(name).to_dict()
    data.pop("estimated_parameters", None)
    data["vocab_size"] = vocab_size
    return ModelConfig.from_dict(data)


def build_corpus(args):
    builder = CorpusBuilder(args.output)
    repos = builder.discover_org(args.org, include_private=not args.public_only) if args.org else []
    repos += [RepositorySource(value) for value in args.repo]
    result = builder.build(repositories=repos, local_roots=args.local_root)
    emit(result)


def train_tokenizer(args):
    tokenizer = AuroBPETokenizer()
    report = tokenizer.train(iter_jsonl_text(args.corpus), vocab_size=args.vocab_size, max_training_bytes=args.max_bytes)
    tokenizer.save(args.output)
    emit(report.to_dict() | {"path": str(Path(args.output).resolve())})


def prepare_dataset(args):
    emit(prepare_token_dataset(args.corpus, args.tokenizer, args.output, validation_fraction=args.validation_fraction))


def write_config(args):
    tokenizer = AuroBPETokenizer.load(args.tokenizer)
    config = TrainConfig(
        model=model_with_vocab(args.preset, tokenizer.vocab_size),
        run_name=args.run_name,
        output_dir=args.runs,
        dataset_dir=args.dataset,
        tokenizer_path=args.tokenizer,
        sequence_length=args.sequence_length,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        precision=args.precision,
    )
    config.save(args.output)
    emit(config.to_dict())


def run_all(args):
    root = Path(args.workspace).resolve()
    corpus_dir, dataset_dir = root / "corpus", root / "dataset"
    tokenizer_path, config_path = root / "tokenizer.json", root / "train-config.json"
    builder = CorpusBuilder(corpus_dir)
    repos = builder.discover_org(args.org, include_private=not args.public_only) if args.org else []
    repos += [RepositorySource(value) for value in args.repo]
    corpus = builder.build(repositories=repos, local_roots=args.local_root)
    tokenizer = AuroBPETokenizer()
    tokenizer_report = tokenizer.train(iter_jsonl_text(corpus_dir / "corpus.jsonl"), vocab_size=args.vocab_size, max_training_bytes=args.max_bytes)
    tokenizer.save(tokenizer_path)
    dataset = prepare_token_dataset(corpus_dir / "corpus.jsonl", tokenizer_path, dataset_dir, validation_fraction=args.validation_fraction)
    config = TrainConfig(
        model=model_with_vocab(args.preset, tokenizer.vocab_size),
        run_name=args.run_name,
        output_dir=str(root / "runs"),
        dataset_dir=str(dataset_dir),
        tokenizer_path=str(tokenizer_path),
        sequence_length=args.sequence_length,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        precision=args.precision,
    )
    config.save(config_path)
    result = train(config)
    receipt = {"schema": "auro.foundry.e2e.v1", "corpus": corpus, "tokenizer": tokenizer_report.to_dict(), "dataset": dataset, "training": result.to_dict()}
    (root / "end-to-end-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    emit(receipt)


def main() -> None:
    parser = argparse.ArgumentParser(prog="auro-foundry")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-corpus"); p.add_argument("--org"); p.add_argument("--repo", action="append", default=[]); p.add_argument("--local-root", action="append", default=[]); p.add_argument("--public-only", action="store_true"); p.add_argument("--output", default="artifacts/auro-foundry/corpus"); p.set_defaults(run=build_corpus)
    p = sub.add_parser("train-tokenizer"); p.add_argument("--corpus", required=True); p.add_argument("--output", default="artifacts/auro-foundry/tokenizer.json"); p.add_argument("--vocab-size", type=int, default=4096); p.add_argument("--max-bytes", type=int, default=64*1024*1024); p.set_defaults(run=train_tokenizer)
    p = sub.add_parser("prepare-dataset"); p.add_argument("--corpus", required=True); p.add_argument("--tokenizer", required=True); p.add_argument("--output", default="artifacts/auro-foundry/dataset"); p.add_argument("--validation-fraction", type=float, default=.01); p.set_defaults(run=prepare_dataset)
    p = sub.add_parser("init-config"); p.add_argument("--preset", choices=["micro","local","14b","206.7b"], default="micro"); p.add_argument("--tokenizer", required=True); p.add_argument("--dataset", required=True); p.add_argument("--output", default="artifacts/auro-foundry/train-config.json"); p.add_argument("--runs", default="artifacts/auro-foundry/runs"); p.add_argument("--run-name", default="auro-run"); p.add_argument("--sequence-length", type=int, default=128); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--accumulation", type=int, default=2); p.add_argument("--steps", type=int, default=20); p.add_argument("--eval-interval", type=int, default=10); p.add_argument("--checkpoint-interval", type=int, default=10); p.add_argument("--device", default="auto"); p.add_argument("--precision", default="auto"); p.set_defaults(run=write_config)
    p = sub.add_parser("train"); p.add_argument("--config", required=True); p.set_defaults(run=lambda a: emit(train(TrainConfig.load(a.config)).to_dict()))
    p = sub.add_parser("resume"); p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    def resume(a):
        value=TrainConfig.load(a.config).to_dict(); value["model"].pop("estimated_parameters",None); value["resume_from"]=a.checkpoint; emit(train(TrainConfig.from_dict(value)).to_dict())
    p.set_defaults(run=resume)
    p = sub.add_parser("train-governed"); p.add_argument("--config", required=True); p.add_argument("--workdir", default="."); p.add_argument("--receipts", default="artifacts/auro-foundry/mesie-receipts"); p.add_argument("--timeout", type=float); p.set_defaults(run=lambda a: emit(run_training_governed(a.config, workdir=a.workdir, receipt_dir=a.receipts, timeout_seconds=a.timeout).__dict__))
    p = sub.add_parser("generate"); p.add_argument("--checkpoint", required=True); p.add_argument("--prompt", required=True); p.add_argument("--device", default="auto"); p.add_argument("--max-new-tokens", type=int, default=128); p.add_argument("--temperature", type=float, default=.8); p.set_defaults(run=lambda a: print(TextGenerator(a.checkpoint,device=a.device).generate(a.prompt,max_new_tokens=a.max_new_tokens,temperature=a.temperature)))
    p = sub.add_parser("serve"); p.add_argument("--checkpoint", required=True); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8090); p.add_argument("--device", default="auto"); p.add_argument("--open-browser", action="store_true")
    def run_server(a):
        if a.open_browser: webbrowser.open(f"http://{a.host}:{a.port}")
        serve(a.checkpoint,host=a.host,port=a.port,device=a.device)
    p.set_defaults(run=run_server)
    p = sub.add_parser("all"); p.add_argument("--workspace", default="artifacts/auro-foundry"); p.add_argument("--org", default="ItsNotAILABS"); p.add_argument("--repo", action="append", default=[]); p.add_argument("--local-root", action="append", default=[]); p.add_argument("--public-only", action="store_true"); p.add_argument("--preset", choices=["micro","local","14b","206.7b"], default="micro"); p.add_argument("--run-name", default="auro-owned-run"); p.add_argument("--vocab-size", type=int, default=4096); p.add_argument("--max-bytes", type=int, default=64*1024*1024); p.add_argument("--validation-fraction", type=float, default=.01); p.add_argument("--sequence-length", type=int, default=128); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--accumulation", type=int, default=2); p.add_argument("--steps", type=int, default=20); p.add_argument("--eval-interval", type=int, default=10); p.add_argument("--checkpoint-interval", type=int, default=10); p.add_argument("--device", default="auto"); p.add_argument("--precision", default="auto"); p.set_defaults(run=run_all)
    args = parser.parse_args(); args.run(args)


if __name__ == "__main__": main()
