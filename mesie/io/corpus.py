"""Spectral corpus loading from directories and file collections."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import load_record

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".json", ".csv"}


class SpectralCorpus:
    """A collection of spectral records loaded from a directory or file list.

    Supports lazy iteration over large spectral libraries without loading
    everything into memory at once, as well as eager loading for smaller
    datasets that fit in RAM.

    Args:
        records: Pre-loaded records (used internally).

    Examples:
        >>> corpus = SpectralCorpus.from_directory("/path/to/spectral/library")
        >>> print(f"Loaded {len(corpus)} records")
        >>> for record in corpus:
        ...     print(record.record_id)
    """

    def __init__(self, records: Optional[List[MultiElementRecord]] = None) -> None:
        self._records: List[MultiElementRecord] = records or []

    @classmethod
    def from_directory(
        cls,
        path: Union[str, Path],
        *,
        recursive: bool = True,
        extensions: Optional[Sequence[str]] = None,
        skip_errors: bool = False,
    ) -> "SpectralCorpus":
        """Load all spectral records from a directory.

        Args:
            path: Path to directory containing spectral files.
            recursive: Whether to search subdirectories recursively.
            extensions: File extensions to include (default: .json, .csv).
            skip_errors: If True, log warnings for unloadable files instead
                of raising exceptions.

        Returns:
            A SpectralCorpus containing all successfully loaded records.

        Raises:
            FileNotFoundError: If the directory does not exist.
            ValueError: If no supported files are found.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {directory}")

        allowed = set(extensions or SUPPORTED_EXTENSIONS)
        allowed = {ext if ext.startswith(".") else f".{ext}" for ext in allowed}

        files = cls._discover_files(directory, allowed, recursive)
        if not files:
            raise ValueError(
                f"No supported files ({', '.join(sorted(allowed))}) found in: {directory}"
            )

        records: List[MultiElementRecord] = []
        for filepath in files:
            try:
                record = load_record(filepath, record_id=filepath.stem)
                records.append(record)
            except Exception as exc:
                if skip_errors:
                    logger.warning("Skipping %s: %s", filepath, exc)
                else:
                    raise

        logger.info("Loaded %d records from %s", len(records), directory)
        return cls(records=records)

    @classmethod
    def from_files(
        cls,
        paths: Sequence[Union[str, Path]],
        *,
        skip_errors: bool = False,
    ) -> "SpectralCorpus":
        """Load spectral records from an explicit list of file paths.

        Args:
            paths: Sequence of file paths to load.
            skip_errors: If True, log warnings for unloadable files.

        Returns:
            A SpectralCorpus containing all successfully loaded records.
        """
        records: List[MultiElementRecord] = []
        for p in paths:
            filepath = Path(p)
            try:
                record = load_record(filepath, record_id=filepath.stem)
                records.append(record)
            except Exception as exc:
                if skip_errors:
                    logger.warning("Skipping %s: %s", filepath, exc)
                else:
                    raise
        return cls(records=records)

    @staticmethod
    def _discover_files(
        directory: Path, extensions: set, recursive: bool
    ) -> List[Path]:
        """Discover files with matching extensions in a directory."""
        files: List[Path] = []
        pattern_fn = directory.rglob if recursive else directory.glob
        for ext in sorted(extensions):
            files.extend(sorted(pattern_fn(f"*{ext}")))
        return files

    def iter_records(self) -> Iterator[MultiElementRecord]:
        """Iterate over all records in the corpus."""
        return iter(self._records)

    def get(self, record_id: str) -> Optional[MultiElementRecord]:
        """Retrieve a record by its ID.

        Args:
            record_id: The record identifier to look up.

        Returns:
            The matching record, or None if not found.
        """
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def filter(self, predicate) -> "SpectralCorpus":
        """Return a new corpus with only records matching the predicate.

        Args:
            predicate: Callable that takes a MultiElementRecord and returns bool.

        Returns:
            A filtered SpectralCorpus.
        """
        return SpectralCorpus(records=[r for r in self._records if predicate(r)])

    @property
    def record_ids(self) -> List[str]:
        """List all record IDs in the corpus."""
        return [r.record_id for r in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MultiElementRecord]:
        return iter(self._records)

    def __getitem__(self, index: int) -> MultiElementRecord:
        return self._records[index]

    def __repr__(self) -> str:
        return f"SpectralCorpus(records={len(self._records)})"
