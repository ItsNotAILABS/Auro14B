"""Auro text tokenizer — trains on corpus, first-class for the LM.

Byte-level BPE-style merges on UTF-8 with special tokens. Pure Python/NumPy
deps only (no sentencepiece/tokenizers package required).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SPECIAL = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<spectral>": 4,
    "<meaning>": 5,
    "<moe>": 6,
}


@dataclass
class AuroTokenizer:
    vocab_size: int = 8192
    merges: List[Tuple[str, str]] = field(default_factory=list)
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self._bootstrap_byte_vocab()

    def _bootstrap_byte_vocab(self) -> None:
        self.token_to_id = dict(SPECIAL)
        # raw bytes as single-char tokens (latin-1 roundtrip)
        for i in range(256):
            ch = bytes([i]).decode("latin-1")
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return SPECIAL["<pad>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL["<bos>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL["<eos>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL["<unk>"]

    def _word_pieces(self, text: str) -> List[str]:
        # keep words and punctuation; encode as latin-1 char sequences
        parts = re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)
        tokens: List[str] = []
        for p in parts:
            tokens.extend(list(p.encode("utf-8").decode("latin-1")))
        return tokens

    def _apply_merges(self, pieces: List[str]) -> List[str]:
        if not self.merges:
            return pieces
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        tokens = pieces[:]
        while True:
            best = None
            best_rank = 10**18
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                r = merge_rank.get(pair)
                if r is not None and r < best_rank:
                    best = i
                    best_rank = r
            if best is None:
                break
            a, b = tokens[best], tokens[best + 1]
            tokens = tokens[:best] + [a + b] + tokens[best + 2 :]
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        pieces = self._apply_merges(self._word_pieces(text))
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for p in pieces:
            ids.append(self.token_to_id.get(p, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        special_ids = set(SPECIAL.values())
        chars: List[str] = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tok = self.id_to_token.get(int(i), "")
            chars.append(tok)
        raw = "".join(chars)
        try:
            return raw.encode("latin-1").decode("utf-8")
        except Exception:
            return raw

    def train(self, texts: Iterable[str], vocab_size: Optional[int] = None) -> "AuroTokenizer":
        """Learn BPE merges up to vocab_size from texts."""
        target = int(vocab_size or self.vocab_size)
        self._bootstrap_byte_vocab()
        # collect corpus as list of piece lists
        corpus: List[List[str]] = [self._word_pieces(t) for t in texts if t and t.strip()]
        if not corpus:
            return self

        merges: List[Tuple[str, str]] = []
        while len(self.token_to_id) < target:
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for pieces in corpus:
                for i in range(len(pieces) - 1):
                    pair_counts[(pieces[i], pieces[i + 1])] += 1
            if not pair_counts:
                break
            best_pair, _ = pair_counts.most_common(1)[0]
            merges.append(best_pair)
            a, b = best_pair
            merged = a + b
            if merged not in self.token_to_id:
                self.token_to_id[merged] = len(self.token_to_id)
            # apply merge to corpus
            new_corpus: List[List[str]] = []
            for pieces in corpus:
                i = 0
                out: List[str] = []
                while i < len(pieces):
                    if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(pieces[i])
                        i += 1
                new_corpus.append(out)
            corpus = new_corpus
            if len(merges) > target * 2:
                break

        self.merges = merges
        self.vocab_size = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AuroTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=int(data.get("vocab_size", 8192)))
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok
