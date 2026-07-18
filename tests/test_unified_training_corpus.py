from __future__ import annotations

import json
from pathlib import Path

from auro_foundry.corpus import CorpusBuilder


def test_builder_does_not_ingest_its_own_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "knowledge.md").write_text("useful training material " * 10, encoding="utf-8")
    output = source / "artifacts" / "corpus"
    manifest = CorpusBuilder(output).build(local_roots=[source], corpus_name="self-safe")
    assert manifest["records"] == 1
    records = [json.loads(line) for line in (output / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["path"] for record in records] == ["knowledge.md"]
