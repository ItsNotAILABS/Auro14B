"""Load and embed user spectral libraries for the EMBED arm / internal API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from mesie import load_record, validate_record
from mesie.embeddings.retrieval import SpectralRetriever
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.io.loaders import RecordInput

PathLike = Union[str, Path]


@dataclass
class UserCorpusEntry:
    id: str
    file: str
    embedding: List[float]
    valid: bool
    validation_level: int


@dataclass
class UserSpectralCorpus:
    """Embedded user library — feeds Octopus EMBED arm and embedding engine."""

    entries: List[UserCorpusEntry] = field(default_factory=list)
    index_path: Optional[Path] = None
    source_label: str = "user"

    @property
    def count(self) -> int:
        return len(self.entries)

    def to_index_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_label,
            "count": self.count,
            "entries": [
                {
                    "id": e.id,
                    "file": e.file,
                    "embedding": e.embedding,
                    "valid": e.valid,
                    "validation_level": e.validation_level,
                }
                for e in self.entries
            ],
        }

    def save(self, path: PathLike) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_index_dict(), indent=2), encoding="utf-8")
        self.index_path = out
        return out

    def record_paths(self) -> List[Path]:
        return [Path(e.file) for e in self.entries if Path(e.file).exists()]


def _load_spectral_file(fp: Path) -> RecordInput:
    """Load MESIE or bundled reference JSON."""
    try:
        return load_record(fp)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        from data import _normalize_reference_payload
    except ImportError:
        from data.__init__ import _normalize_reference_payload  # type: ignore

    payload = json.loads(fp.read_text(encoding="utf-8"))
    return load_record(_normalize_reference_payload(payload))


def _collect_json_paths(paths: Sequence[PathLike]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.glob("**/*.json")))
        elif p.is_file() and p.suffix.lower() == ".json":
            found.append(p)
    return found


def embed_paths(
    paths: Sequence[PathLike],
    *,
    vectorizer: Optional[SpectralVectorizer] = None,
    save_to: Optional[PathLike] = None,
) -> UserSpectralCorpus:
    """Embed JSON spectra from files or folders."""
    vec = vectorizer or SpectralVectorizer(n_bands=8)
    files = _collect_json_paths(paths)
    entries: List[UserCorpusEntry] = []
    for fp in files:
        rec = _load_spectral_file(fp)
        report = validate_record(rec)
        entries.append(
            UserCorpusEntry(
                id=rec.record_id,
                file=str(fp.resolve()),
                embedding=vec.transform(rec).tolist(),
                valid=report.is_valid,
                validation_level=report.level,
            )
        )
    corpus = UserSpectralCorpus(entries=entries, source_label="user")
    if save_to:
        corpus.save(save_to)
    return corpus


def load_user_index(path: PathLike) -> UserSpectralCorpus:
    """Load `my_spectral_index.json` or compatible index."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    entries = [
        UserCorpusEntry(
            id=e["id"],
            file=e.get("file", ""),
            embedding=e["embedding"],
            valid=bool(e.get("valid", True)),
            validation_level=int(e.get("validation_level", 0)),
        )
        for e in data.get("entries", [])
    ]
    return UserSpectralCorpus(entries=entries, index_path=p, source_label=data.get("source", "user"))


def attach_corpus_to_retriever(
    corpus: UserSpectralCorpus,
    retriever: SpectralRetriever,
) -> int:
    """Index user records into an existing retriever (reloads from files)."""
    records = [_load_spectral_file(p) for p in corpus.record_paths()]
    if not records:
        return 0
    retriever.index(records)
    return len(records)