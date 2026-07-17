from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|receipt|>",
    "<|spectral|>",
)


@dataclass(frozen=True)
class TokenizerReport:
    tokenizer_id: str
    vocab_size: int
    merge_count: int
    training_bytes: int
    training_documents: int
    sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "vocab_size": self.vocab_size,
            "merge_count": self.merge_count,
            "training_bytes": self.training_bytes,
            "training_documents": self.training_documents,
            "sha256": self.sha256,
        }


class AuroBPETokenizer:
    """Repository-native byte BPE tokenizer.

    It starts with a complete byte alphabet, so every UTF-8 string is
    representable, then learns deterministic merges from Medina-owned corpora.
    No external tokenizer service or vocabulary is required.
    """

    def __init__(
        self,
        *,
        tokenizer_id: str = "auro-bpe-v1",
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
        merges: list[tuple[int, int, int]] | None = None,
        token_bytes: dict[int, bytes] | None = None,
    ) -> None:
        self.tokenizer_id = tokenizer_id
        self.special_tokens = tuple(special_tokens)
        self.special_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_special = {idx: token for token, idx in self.special_to_id.items()}
        offset = len(self.special_tokens)
        self.token_bytes = token_bytes or {offset + value: bytes([value]) for value in range(256)}
        self.merges = list(merges or [])
        self.merge_ranks = {(left, right): (rank, new) for rank, (left, right, new) in enumerate(self.merges)}
        self._validate()

    def _validate(self) -> None:
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")
        for left, right, new in self.merges:
            if left not in self.token_bytes or right not in self.token_bytes:
                raise ValueError("merge references an unknown token")
            expected = self.token_bytes[left] + self.token_bytes[right]
            if self.token_bytes.get(new) != expected:
                raise ValueError("merged token bytes do not match its parents")

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + len(self.token_bytes)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def train(
        self,
        texts: Iterable[str],
        *,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        max_training_bytes: int = 64 * 1024 * 1024,
    ) -> TokenizerReport:
        if vocab_size < len(self.special_tokens) + 256:
            raise ValueError("vocab_size is smaller than the lossless byte vocabulary")
        offset = len(self.special_tokens)
        sequences: list[list[int]] = []
        total_bytes = 0
        documents = 0
        for text in texts:
            raw = text.encode("utf-8", errors="replace")
            if not raw:
                continue
            remaining = max_training_bytes - total_bytes
            if remaining <= 0:
                break
            raw = raw[:remaining]
            sequences.append([offset + value for value in raw])
            total_bytes += len(raw)
            documents += 1

        target_merges = vocab_size - (len(self.special_tokens) + 256)
        next_id = max(self.token_bytes) + 1
        for _ in range(target_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            best_pair, frequency = max(
                counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1])
            )
            if frequency < min_frequency:
                break
            left, right = best_pair
            new_id = next_id
            next_id += 1
            self.token_bytes[new_id] = self.token_bytes[left] + self.token_bytes[right]
            self.merges.append((left, right, new_id))
            sequences = [self._replace_pair(sequence, left, right, new_id) for sequence in sequences]

        self.merge_ranks = {(left, right): (rank, new) for rank, (left, right, new) in enumerate(self.merges)}
        digest = self.digest()
        return TokenizerReport(
            tokenizer_id=self.tokenizer_id,
            vocab_size=self.vocab_size,
            merge_count=len(self.merges),
            training_bytes=total_bytes,
            training_documents=documents,
            sha256=digest,
        )

    @staticmethod
    def _replace_pair(sequence: list[int], left: int, right: int, new_id: int) -> list[int]:
        output: list[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
                output.append(new_id)
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return output

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        offset = len(self.special_tokens)
        sequence = [offset + value for value in text.encode("utf-8", errors="replace")]
        while len(sequence) > 1:
            selected: tuple[int, int, int] | None = None
            for left, right in zip(sequence, sequence[1:]):
                candidate = self.merge_ranks.get((left, right))
                if candidate is None:
                    continue
                rank, new_id = candidate
                if selected is None or rank < selected[0]:
                    selected = (rank, left, right)
            if selected is None:
                break
            _, left, right = selected
            new_id = self.merge_ranks[(left, right)][1]
            sequence = self._replace_pair(sequence, left, right, new_id)
        if add_bos:
            sequence.insert(0, self.bos_id)
        if add_eos:
            sequence.append(self.eos_id)
        return sequence

    def decode(self, token_ids: Iterable[int], *, keep_special: bool = False) -> str:
        chunks: list[bytes] = []
        special_chunks: list[str] = []
        for token_id in token_ids:
            if token_id in self.id_to_special:
                if keep_special:
                    special_chunks.append(self.id_to_special[token_id])
                continue
            value = self.token_bytes.get(int(token_id))
            if value is not None:
                chunks.append(value)
        text = b"".join(chunks).decode("utf-8", errors="replace")
        return "".join(special_chunks) + text if keep_special else text

    def to_dict(self) -> dict:
        return {
            "schema": "auro.foundry.tokenizer.v1",
            "tokenizer_id": self.tokenizer_id,
            "special_tokens": list(self.special_tokens),
            "token_bytes_hex": {str(key): value.hex() for key, value in sorted(self.token_bytes.items())},
            "merges": [list(item) for item in self.merges],
            "vocab_size": self.vocab_size,
        }

    def digest(self) -> str:
        normalized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["sha256"] = self.digest()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "AuroBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            tokenizer_id=payload["tokenizer_id"],
            special_tokens=tuple(payload["special_tokens"]),
            merges=[tuple(int(v) for v in merge) for merge in payload["merges"]],
            token_bytes={int(key): bytes.fromhex(value) for key, value in payload["token_bytes_hex"].items()},
        )
        expected = payload.get("sha256")
        if expected and tokenizer.digest() != expected:
            raise ValueError("tokenizer hash mismatch")
        return tokenizer


def iter_jsonl_text(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get("text")
            if isinstance(text, str) and text:
                yield text
