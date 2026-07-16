from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .tokenizer import AuroBPETokenizer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_token_dataset(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.01,
    minimum_validation_records: int = 1,
) -> dict:
    """Encode corpus JSONL into memory-mappable uint32 token streams."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    corpus = Path(corpus_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AuroBPETokenizer.load(tokenizer_path)

    train_path = output / "train.bin"
    validation_path = output / "validation.bin"
    train_records = validation_records = train_tokens = validation_tokens = 0

    with train_path.open("wb") as train_handle, validation_path.open("wb") as validation_handle:
        with corpus.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text:
                    continue
                tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
                array = np.asarray(tokens, dtype=np.uint32)
                key = str(record.get("sha256") or f"line-{line_number}")
                bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
                use_validation = bucket < validation_fraction
                if use_validation:
                    array.tofile(validation_handle)
                    validation_records += 1
                    validation_tokens += len(tokens)
                else:
                    array.tofile(train_handle)
                    train_records += 1
                    train_tokens += len(tokens)

    if validation_records < minimum_validation_records and train_tokens > 32:
        train = np.memmap(train_path, dtype=np.uint32, mode="r")
        count = min(max(32, int(len(train) * max(validation_fraction, 0.01))), len(train))
        np.asarray(train[-count:], dtype=np.uint32).tofile(validation_path)
        validation_records = 1
        validation_tokens = count

    manifest = {
        "schema": "auro.foundry.token_dataset.v1",
        "corpus_path": str(corpus.resolve()),
        "corpus_sha256": _sha256_file(corpus),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "tokenizer_sha256": tokenizer.digest(),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint32",
        "train": {
            "path": str(train_path.resolve()),
            "records": train_records,
            "tokens": train_tokens,
            "sha256": _sha256_file(train_path),
        },
        "validation": {
            "path": str(validation_path.resolve()),
            "records": validation_records,
            "tokens": validation_tokens,
            "sha256": _sha256_file(validation_path),
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


class TokenBlockDataset:
    """Map-style dataset over a packed token stream.

    Importing torch is deferred so corpus and tokenizer commands remain usable
    on machines that are not training nodes.
    """

    def __init__(self, path: str | Path, sequence_length: int, *, stride: int | None = None) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.path = Path(path)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride or sequence_length)
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        if len(self.tokens) <= self.sequence_length:
            raise ValueError(f"not enough tokens in {self.path} for sequence length {self.sequence_length}")
        self.blocks = max(1, (len(self.tokens) - self.sequence_length - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int):
        import torch

        if index < 0:
            index += self.blocks
        if not 0 <= index < self.blocks:
            raise IndexError(index)
        start = min(index * self.stride, len(self.tokens) - self.sequence_length - 1)
        chunk = np.asarray(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def iter_manifest_paths(dataset_dir: str | Path) -> Iterator[Path]:
    manifest = json.loads((Path(dataset_dir) / "dataset-manifest.json").read_text(encoding="utf-8"))
    yield Path(manifest["train"]["path"])
    yield Path(manifest["validation"]["path"])
