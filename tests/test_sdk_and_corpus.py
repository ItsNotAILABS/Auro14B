"""Tests for SpectralCorpus, SpectralIntelligenceSDK, and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mesie.io.corpus import SpectralCorpus
from mesie.sdk import SpectralIntelligenceSDK
from mesie.cli import main as cli_main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_json_record(path: Path, record_id: str, n_points: int = 32) -> None:
    """Write a simple JSON spectral record to disk."""
    data = {
        "record_id": record_id,
        "frequency": np.linspace(0.1, 50.0, n_points).tolist(),
        "amplitude": np.random.default_rng(42).random(n_points).tolist(),
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Create a temp directory with several JSON spectral records."""
    for i in range(5):
        _make_json_record(tmp_path / f"record_{i}.json", f"rec_{i}")
    # Subdirectory
    sub = tmp_path / "subdir"
    sub.mkdir()
    _make_json_record(sub / "nested.json", "nested_rec")
    return tmp_path


@pytest.fixture
def single_record_path(tmp_path: Path) -> Path:
    """Create a single JSON record file."""
    p = tmp_path / "single.json"
    _make_json_record(p, "single_rec")
    return p


# ---------------------------------------------------------------------------
# SpectralCorpus tests
# ---------------------------------------------------------------------------


class TestSpectralCorpus:
    def test_from_directory_recursive(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        assert len(corpus) == 6  # 5 top-level + 1 nested

    def test_from_directory_non_recursive(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir, recursive=False)
        assert len(corpus) == 5

    def test_from_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            SpectralCorpus.from_directory("/nonexistent/path")

    def test_from_directory_no_files(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No supported files"):
            SpectralCorpus.from_directory(tmp_path)

    def test_get_by_id(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        record = corpus.get("rec_0")
        assert record is not None
        assert record.record_id == "rec_0"

    def test_get_missing_returns_none(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        assert corpus.get("nonexistent") is None

    def test_record_ids(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        ids = corpus.record_ids
        assert "rec_0" in ids
        assert "nested_rec" in ids  # record_id from JSON content

    def test_filter(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        filtered = corpus.filter(lambda r: r.record_id.startswith("rec_"))
        assert len(filtered) == 5

    def test_iteration(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        records = list(corpus)
        assert len(records) == 6

    def test_indexing(self, corpus_dir: Path):
        corpus = SpectralCorpus.from_directory(corpus_dir)
        r = corpus[0]
        assert r.record_id in corpus.record_ids

    def test_from_files(self, corpus_dir: Path):
        files = list(corpus_dir.glob("*.json"))
        corpus = SpectralCorpus.from_files(files)
        assert len(corpus) == 5

    def test_skip_errors(self, tmp_path: Path):
        # Create one valid and one invalid file
        _make_json_record(tmp_path / "good.json", "good")
        (tmp_path / "bad.json").write_text("not valid json{{{", encoding="utf-8")
        corpus = SpectralCorpus.from_directory(tmp_path, skip_errors=True)
        assert len(corpus) == 1


# ---------------------------------------------------------------------------
# SpectralIntelligenceSDK tests
# ---------------------------------------------------------------------------


class TestSDK:
    def test_version(self):
        engine = SpectralIntelligenceSDK()
        assert engine.version == "0.3.0"

    def test_repr(self):
        engine = SpectralIntelligenceSDK()
        assert "SpectralIntelligenceSDK" in repr(engine)

    def test_load_record(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        assert record.record_id == "single_rec"  # record_id from JSON content

    def test_load_corpus(self, corpus_dir: Path):
        engine = SpectralIntelligenceSDK()
        corpus = engine.load_corpus(corpus_dir)
        assert len(corpus) == 6
        assert engine.corpus is corpus

    def test_match(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        result = engine.match(record, record)
        assert result.score > 0.9  # Self-match should be high

    def test_rank_requires_corpus(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        with pytest.raises(RuntimeError, match="No corpus loaded"):
            engine.rank(record)

    def test_rank_with_corpus(self, corpus_dir: Path):
        engine = SpectralIntelligenceSDK()
        engine.load_corpus(corpus_dir)
        candidate = engine.load(corpus_dir / "record_0.json")
        results = engine.rank(candidate, top_k=3)
        assert len(results) <= 3
        assert results[0].score >= results[-1].score

    def test_embed(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        embedding = engine.embed(record)
        assert embedding.ndim == 2
        assert embedding.shape[0] == 1

    def test_embed_multiple(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        embedding = engine.embed([record, record])
        assert embedding.shape[0] == 2

    def test_validate(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        report = engine.validate(record)
        assert report is not None

    def test_normalize(self, single_record_path: Path):
        engine = SpectralIntelligenceSDK()
        record = engine.load(single_record_path)
        normalized = engine.normalize(record)
        assert normalized.record_id == record.record_id

    def test_generate_psd(self):
        engine = SpectralIntelligenceSDK()
        record = engine.generate_psd()
        assert len(record.components) > 0

    def test_generate_fas(self):
        engine = SpectralIntelligenceSDK()
        record = engine.generate_fas()
        assert len(record.components) > 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_load_corpus_command(self, corpus_dir: Path, capsys):
        cli_main(["load-corpus", str(corpus_dir)])
        captured = capsys.readouterr()
        assert "Loaded 6 records" in captured.out

    def test_load_corpus_list(self, corpus_dir: Path, capsys):
        cli_main(["load-corpus", str(corpus_dir), "--list"])
        captured = capsys.readouterr()
        assert "rec_0" in captured.out

    def test_info_command(self, single_record_path: Path, capsys):
        cli_main(["info", str(single_record_path)])
        captured = capsys.readouterr()
        assert "Record ID:" in captured.out
        assert "Components:" in captured.out

    def test_no_command_shows_help(self, capsys):
        with pytest.raises(SystemExit):
            cli_main([])

    def test_mesie_entrypoint_installed(self):
        result = subprocess.run(
            [sys.executable, "-m", "mesie.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "MESIE" in result.stdout
