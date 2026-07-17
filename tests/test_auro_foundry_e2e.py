from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from auro_foundry.config import ModelConfig, TrainConfig
from auro_foundry.corpus import CorpusBuilder
from auro_foundry.dataset import TokenBlockDataset, prepare_token_dataset
from auro_foundry.generation import TextGenerator
from auro_foundry.server import FoundryServer
from auro_foundry.tokenizer import AuroBPETokenizer, iter_jsonl_text
from auro_foundry.training import train


def make_source(root: Path) -> Path:
    source = root / "owned-repo"
    source.mkdir()
    (source / "README.md").write_text(("Auro learns from Medina-owned repositories. " * 80) + "\n", encoding="utf-8")
    (source / "runtime.py").write_text(("def heartbeat():\n    return 'NOVA MESIE AURO'\n" * 80), encoding="utf-8")
    (source / "secret.txt").write_text("api_key=THIS_VALUE_MUST_BE_REDACTED_123456789\n" + "safe doctrine\n" * 80, encoding="utf-8")
    return source


def tiny_config(dataset: Path, tokenizer: Path, runs: Path, *, steps: int, resume: str | None = None) -> TrainConfig:
    token = AuroBPETokenizer.load(tokenizer)
    model = ModelConfig(
        model_id="Auro-E2E-Tiny",
        vocab_size=token.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
        dropout=0.0,
    )
    return TrainConfig(
        model=model,
        run_name="e2e",
        output_dir=str(runs),
        dataset_dir=str(dataset),
        tokenizer_path=str(tokenizer),
        sequence_length=32,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=steps,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=1,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        log_interval=1,
        device="cpu",
        precision="fp32",
        resume_from=resume,
    )


def test_tokenizer_lossless_byte_space() -> None:
    tokenizer = AuroBPETokenizer()
    for value in range(128):
        text = chr(value)
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size >= 266
    assert tokenizer.pad_id != tokenizer.bos_id
    assert tokenizer.bos_id != tokenizer.eos_id


def test_complete_foundry_lifecycle(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    corpus_dir = tmp_path / "corpus"
    manifest = CorpusBuilder(corpus_dir).build(local_roots=[source], corpus_name="e2e-owned")
    assert manifest["records"] >= 3
    assert manifest["text_bytes"] > 1000
    assert manifest["redactions"] >= 1
    assert len(manifest["corpus_sha256"]) == 64
    corpus_text = (corpus_dir / "corpus.jsonl").read_text(encoding="utf-8")
    assert "THIS_VALUE_MUST_BE_REDACTED" not in corpus_text
    assert "<|redacted_secret|>" in corpus_text

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = AuroBPETokenizer()
    report = tokenizer.train(iter_jsonl_text(corpus_dir / "corpus.jsonl"), vocab_size=320, max_training_bytes=2_000_000)
    tokenizer.save(tokenizer_path)
    assert report.training_documents >= 3
    assert report.training_bytes > 1000
    assert report.vocab_size >= 266
    assert len(report.sha256) == 64
    restored = AuroBPETokenizer.load(tokenizer_path)
    sample = "NOVA routes MESIE into Auro Foundry."
    assert restored.decode(restored.encode(sample)) == sample

    dataset_dir = tmp_path / "dataset"
    dataset_manifest = prepare_token_dataset(corpus_dir / "corpus.jsonl", tokenizer_path, dataset_dir, validation_fraction=0.25)
    assert dataset_manifest["train"]["tokens"] > 64
    assert dataset_manifest["validation"]["tokens"] >= 32
    assert Path(dataset_manifest["train"]["path"]).exists()
    assert Path(dataset_manifest["validation"]["path"]).exists()
    block = TokenBlockDataset(dataset_dir / "train.bin", 32)
    inputs, targets = block[0]
    assert inputs.shape == targets.shape == (32,)
    assert torch.equal(inputs[1:], targets[:-1])

    runs = tmp_path / "runs"
    first = train(tiny_config(dataset_dir, tokenizer_path, runs, steps=2))
    checkpoint = Path(first.final_checkpoint)
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".pt.sha256").exists()
    assert first.steps == 2
    assert first.tokens_seen > 0
    assert first.train_loss > 0
    assert Path(first.receipt_path).exists()

    resumed = train(tiny_config(dataset_dir, tokenizer_path, runs, steps=3, resume=str(checkpoint)))
    assert resumed.steps == 3
    assert Path(resumed.final_checkpoint).exists()
    assert resumed.tokens_seen >= first.tokens_seen

    generator = TextGenerator(resumed.final_checkpoint, device="cpu")
    assert generator.metadata["model_id"] == "Auro-E2E-Tiny"
    assert generator.metadata["parameters"] > 0
    output = generator.generate("Auro", max_new_tokens=4, temperature=0)
    assert isinstance(output, str)

    app = FoundryServer(resumed.final_checkpoint, device="cpu")
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            health = json.loads(response.read())
        assert health["status"] == "ok"
        assert health["model"]["model_id"] == "Auro-E2E-Tiny"
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "Auro"}], "max_tokens": 2, "temperature": 0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            chat = json.loads(response.read())
        assert chat["object"] == "chat.completion"
        assert isinstance(chat["choices"][0]["message"]["content"], str)
    finally:
        server.shutdown()
        server.server_close()
