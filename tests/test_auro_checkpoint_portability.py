from __future__ import annotations

from pathlib import Path

import torch

from auro_foundry.config import ModelConfig, TrainConfig
from auro_foundry.generation import load_model
from auro_foundry.model import AuroForCausalLM
from auro_foundry.tokenizer import AuroBPETokenizer


def test_relative_tokenizer_path_is_checkpoint_relative(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    elsewhere = tmp_path / "elsewhere"
    bundle.mkdir()
    elsewhere.mkdir()
    tokenizer = AuroBPETokenizer()
    tokenizer.save(bundle / "tokenizer.json")
    model_config = ModelConfig(
        model_id="Auro-Portable-Test",
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
    )
    train_config = TrainConfig(
        model=model_config,
        tokenizer_path="tokenizer.json",
        dataset_dir="dataset",
        output_dir="runs",
        max_steps=1,
        sequence_length=8,
    )
    model = AuroForCausalLM(model_config)
    checkpoint = bundle / "final.pt"
    torch.save({"config": train_config.to_dict(), "model": model.state_dict(), "step": 1, "tokens_seen": 8}, checkpoint)
    monkeypatch.chdir(elsewhere)
    loaded_model, loaded_tokenizer, metadata, device = load_model(checkpoint, "cpu")
    assert loaded_model.config.model_id == "Auro-Portable-Test"
    assert loaded_tokenizer.digest() == tokenizer.digest()
    assert metadata["tokenizer"] == str((bundle / "tokenizer.json").resolve())
    assert str(device) == "cpu"
