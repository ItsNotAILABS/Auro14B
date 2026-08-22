"""Small auditable local autoregressive language-model lane.

HIM-native-v0 proves the open-weight training and loading path only. It is not a
claim of assistant quality or mature intelligence. Every load rehashes all
checkpoint artifacts and verifies the immutable training receipt before weights
enter memory.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

LEGACY_CONTROL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>",
    "<tool>", "<receipt>", "<spectral>", "<memory>", "<repository>",
    "<code>", "<test>", "<execution>", "<nova>", "<mesie>",
)
RELEASE_EXTENSION_TOKENS = ("<mathesis>", "<cain>", "<oro>")
CONTROL_TOKENS = LEGACY_CONTROL_TOKENS  # compatibility export: IDs 0..15 remain immutable
BYTE_OFFSET = len(LEGACY_CONTROL_TOKENS)
BYTE_VOCAB_END = BYTE_OFFSET + 256
VOCAB_SIZE = BYTE_VOCAB_END  # compatibility export for tokenizer v1
RELEASE_VOCAB_SIZE = BYTE_VOCAB_END + len(RELEASE_EXTENSION_TOKENS)
CHECKPOINT_FILES = ("weights.npz.b64", "config.json", "tokenizer.json", "training_report.json", "training_receipt.json")


class CheckpointIntegrityError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def environment_manifest() -> dict:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "byteorder": sys.byteorder,
    }


def corpus_digest(corpus: str | bytes | Path) -> str:
    if isinstance(corpus, Path):
        return _sha256_file(corpus)
    if isinstance(corpus, str):
        return _sha256_bytes(corpus.encode("utf-8"))
    return _sha256_bytes(bytes(corpus))


class ByteTokenizer:
    """Lossless byte tokenizer with versioned, immutable token IDs.

    v1 layout:
      controls 0..15, bytes 16..271

    v2 layout:
      controls 0..15, bytes 16..271, extension controls 272..274

    Appending extension controls after the byte range preserves every legacy
    control and byte ID. A v1 checkpoint therefore remains loadable without
    remapping embeddings or output rows.
    """

    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self, version: str = "v2"):
        normalized = str(version).lower()
        if normalized not in {"v1", "v2"}:
            raise ValueError(f"unsupported byte tokenizer version: {version}")
        self.version = normalized
        self.byte_offset = BYTE_OFFSET
        self.byte_vocab_end = BYTE_VOCAB_END
        self.control_token_ids = {token: index for index, token in enumerate(LEGACY_CONTROL_TOKENS)}
        if normalized == "v2":
            self.control_token_ids.update({
                token: BYTE_VOCAB_END + index
                for index, token in enumerate(RELEASE_EXTENSION_TOKENS)
            })
        self.control_tokens = tuple(self.control_token_ids)
        self.vocab_size = VOCAB_SIZE if normalized == "v1" else RELEASE_VOCAB_SIZE

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.byte_offset + value for value in text.encode("utf-8")]
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int]) -> str:
        data = bytes(int(i) - self.byte_offset for i in ids if self.byte_offset <= int(i) < self.byte_vocab_end)
        return data.decode("utf-8", errors="replace")

    def manifest(self) -> dict:
        return {
            "schema": "auro.byte_tokenizer.v1" if self.version == "v1" else "auro.byte_tokenizer.v2",
            "version": self.version,
            "vocab_size": self.vocab_size,
            "byte_offset": self.byte_offset,
            "byte_vocab_end_exclusive": self.byte_vocab_end,
            "control_tokens": list(self.control_tokens),
            "control_token_ids": dict(self.control_token_ids),
            "unknown_token": None,
            "byte_round_trip": True,
            "legacy_control_ids_preserved": True,
            "legacy_byte_ids_preserved": True,
        }


@dataclass(frozen=True)
class OpenHIMConfig:
    context_length: int = 16
    embedding_dim: int = 48
    hidden_dim: int = 128
    seed: int = 20260718
    tokenizer_version: str = "v2"


class OpenHIM:
    """Context MLP causal LM with explicit local weights."""

    def __init__(self, config: OpenHIMConfig = OpenHIMConfig()):
        self.config = config
        self.tokenizer = ByteTokenizer(config.tokenizer_version)
        rng = np.random.default_rng(config.seed)
        scale = 0.02
        vocab_size = self.tokenizer.vocab_size
        self.weights = {
            "embedding": rng.normal(0, scale, (vocab_size, config.embedding_dim)).astype(np.float32),
            "w1": rng.normal(0, scale, (config.context_length * config.embedding_dim, config.hidden_dim)).astype(np.float32),
            "b1": np.zeros(config.hidden_dim, np.float32),
            "w2": rng.normal(0, scale, (config.hidden_dim, vocab_size)).astype(np.float32),
            "b2": np.zeros(vocab_size, np.float32),
        }
        self.verified_checkpoint: dict | None = None

    @property
    def num_parameters(self) -> int:
        return sum(int(value.size) for value in self.weights.values())

    def logits(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = self.weights["embedding"][contexts]
        flat = embedded.reshape(len(contexts), -1)
        hidden = np.tanh(flat @ self.weights["w1"] + self.weights["b1"])
        return hidden @ self.weights["w2"] + self.weights["b2"], hidden, flat

    def generate(self, prompt: str, *, max_new_tokens: int = 120, temperature: float = 0.7, top_k: int = 24, seed: int = 7) -> str:
        rng = np.random.default_rng(seed)
        ids = self.tokenizer.encode(prompt, bos=True)
        blocked_control_ids = np.asarray(sorted(self.tokenizer.control_token_ids.values()), dtype=np.int64)
        for _ in range(max_new_tokens):
            context = ([self.tokenizer.pad_id] * self.config.context_length + ids)[-self.config.context_length:]
            logits, _, _ = self.logits(np.asarray([context], dtype=np.int64))
            scores = logits[0].astype(np.float64) / max(float(temperature), 0.05)
            scores[blocked_control_ids] = -1e9
            if 0 < top_k < len(scores):
                keep = np.argpartition(scores, -top_k)[-top_k:]
                mask = np.ones(len(scores), dtype=bool)
                mask[keep] = False
                scores[mask] = -1e9
            scores -= scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            ids.append(int(rng.choice(self.tokenizer.vocab_size, p=probabilities)))
        return self.tokenizer.decode(ids)

    def save(
        self,
        directory: str | Path,
        report: Mapping[str, object],
        *,
        corpus_sha256: str,
        source_commit: str,
        runner_identity: str,
        signing_key: str | None = None,
    ) -> dict:
        if len(corpus_sha256) != 64 or not source_commit or not runner_identity:
            raise ValueError("corpus_sha256, source_commit and runner_identity are mandatory")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        raw = io.BytesIO()
        np.savez_compressed(raw, **self.weights)
        encoded = base64.b64encode(raw.getvalue()).decode("ascii")
        (directory / "weights.npz.b64").write_text(encoded, encoding="ascii")
        (directory / "config.json").write_text(json.dumps(self.config.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tokenizer_manifest = self.tokenizer.manifest()
        (directory / "tokenizer.json").write_text(json.dumps(tokenizer_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bounded_report = dict(report)
        bounded_report.setdefault("claim_boundary", {
            "pipeline_proven": True,
            "assistant_quality_proven": False,
            "st14b_capability_proven": False,
            "mature_him_intelligence_proven": False,
        })
        (directory / "training_report.json").write_text(json.dumps(bounded_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt = {
            "schema": "auro.openhim.training_receipt.v3",
            "model": "HIM-native-v0",
            "parameter_count": self.num_parameters,
            "context_length": self.config.context_length,
            "tokenizer_version": self.tokenizer.version,
            "tokenizer_sha256": _sha256_file(directory / "tokenizer.json"),
            "corpus_sha256": corpus_sha256,
            "source_commit": source_commit,
            "runner_identity": runner_identity,
            "environment": environment_manifest(),
            "report_sha256": _sha256_file(directory / "training_report.json"),
            "claim_class": "pipeline-fixture-only",
        }
        canonical = _canonical(receipt)
        receipt["receipt_sha256"] = _sha256_bytes(canonical)
        if signing_key:
            receipt["runner_hmac_sha256"] = hmac.new(signing_key.encode(), canonical, hashlib.sha256).hexdigest()
        (directory / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {name: _sha256_file(directory / name) for name in CHECKPOINT_FILES}
        manifest_payload = {"schema": "auro.checkpoint.sha256s.v1", "files": manifest}
        manifest_payload["manifest_sha256"] = _sha256_bytes(_canonical(manifest_payload))
        (directory / "SHA256SUMS.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "weights_bytes": len(raw.getvalue()),
            "encoded_bytes": len(encoded),
            "manifest_sha256": manifest_payload["manifest_sha256"],
            "tokenizer_version": self.tokenizer.version,
            "vocab_size": self.tokenizer.vocab_size,
        }

    @classmethod
    def load(cls, directory: str | Path, *, runner_signing_key: str | None = None) -> "OpenHIM":
        directory = Path(directory)
        verification = verify_checkpoint(directory, runner_signing_key=runner_signing_key)
        config_payload = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if "tokenizer_version" not in config_payload:
            config_payload["tokenizer_version"] = "v1"
        config = OpenHIMConfig(**config_payload)
        model = cls(config)
        raw = base64.b64decode((directory / "weights.npz.b64").read_text(encoding="ascii"), validate=True)
        with np.load(io.BytesIO(raw), allow_pickle=False) as values:
            expected = set(model.weights)
            if set(values.files) != expected:
                raise CheckpointIntegrityError("weight tensor inventory mismatch")
            loaded: dict[str, np.ndarray] = {}
            for name in sorted(values.files):
                value = values[name].astype(np.float32)
                if value.shape != model.weights[name].shape:
                    raise CheckpointIntegrityError(
                        f"weight tensor shape mismatch: {name} expected={model.weights[name].shape} actual={value.shape}"
                    )
                loaded[name] = value
            model.weights = loaded
        model.verified_checkpoint = verification
        return model


def verify_checkpoint(directory: str | Path, *, runner_signing_key: str | None = None) -> dict:
    directory = Path(directory)
    manifest_path = directory / "SHA256SUMS.json"
    if not manifest_path.is_file():
        raise CheckpointIntegrityError("SHA256SUMS.json is required before loading weights")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied_manifest_hash = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if supplied_manifest_hash != _sha256_bytes(_canonical(unsigned)):
        raise CheckpointIntegrityError("checkpoint manifest seal mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(CHECKPOINT_FILES):
        raise CheckpointIntegrityError("checkpoint file inventory is incomplete")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or not hmac.compare_digest(_sha256_file(path), str(expected)):
            raise CheckpointIntegrityError(f"checkpoint artifact hash mismatch: {name}")

    tokenizer_manifest = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    if not isinstance(tokenizer_manifest, dict) or tokenizer_manifest.get("byte_round_trip") is not True or tokenizer_manifest.get("unknown_token") is not None:
        raise CheckpointIntegrityError("tokenizer manifest violates byte-lossless/no-UNK invariant")
    tokenizer_version = str(tokenizer_manifest.get("version") or ("v2" if tokenizer_manifest.get("schema") == "auro.byte_tokenizer.v2" else "v1"))
    if tokenizer_version not in {"v1", "v2"}:
        raise CheckpointIntegrityError("unsupported checkpoint tokenizer version")
    expected_tokenizer = ByteTokenizer(tokenizer_version).manifest()
    supplied_ids = tokenizer_manifest.get("control_token_ids")
    if supplied_ids is not None and supplied_ids != expected_tokenizer["control_token_ids"]:
        raise CheckpointIntegrityError("checkpoint control-token IDs do not match immutable tokenizer layout")

    receipt = json.loads((directory / "training_receipt.json").read_text(encoding="utf-8"))
    required = ("corpus_sha256", "source_commit", "runner_identity", "environment", "receipt_sha256")
    if any(not receipt.get(field) for field in required):
        raise CheckpointIntegrityError("training receipt lacks immutable provenance")
    supplied_receipt_hash = receipt["receipt_sha256"]
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256", None)
    signature = unsigned_receipt.pop("runner_hmac_sha256", None)
    canonical = _canonical(unsigned_receipt)
    if not hmac.compare_digest(supplied_receipt_hash, _sha256_bytes(canonical)):
        raise CheckpointIntegrityError("training receipt hash mismatch")
    if signature:
        if not runner_signing_key:
            raise CheckpointIntegrityError("signed training receipt requires runner verification key")
        expected_signature = hmac.new(runner_signing_key.encode(), canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            raise CheckpointIntegrityError("runner signature mismatch")
    elif os.getenv("AURO_REQUIRE_SIGNED_TRAINING_RECEIPT", "1") == "1":
        raise CheckpointIntegrityError("unsigned training receipt rejected")

    receipt_tokenizer_version = receipt.get("tokenizer_version")
    if receipt_tokenizer_version and str(receipt_tokenizer_version) != tokenizer_version:
        raise CheckpointIntegrityError("training receipt tokenizer version mismatch")
    receipt_tokenizer_sha = receipt.get("tokenizer_sha256")
    if receipt_tokenizer_sha and not hmac.compare_digest(str(receipt_tokenizer_sha), _sha256_file(directory / "tokenizer.json")):
        raise CheckpointIntegrityError("training receipt tokenizer hash mismatch")

    return {
        "verified": True,
        "manifest_sha256": supplied_manifest_hash,
        "corpus_sha256": receipt["corpus_sha256"],
        "source_commit": receipt["source_commit"],
        "runner_identity": receipt["runner_identity"],
        "signed": bool(signature),
        "claim_class": receipt.get("claim_class"),
        "tokenizer_version": tokenizer_version,
        "tokenizer_sha256": _sha256_file(directory / "tokenizer.json"),
    }
