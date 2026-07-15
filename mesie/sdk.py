"""Unified SDK entrypoint for the MESIE Spectral Intelligence Engine.

Provides a single high-level interface for all MESIE capabilities:
corpus loading, matching, generation, embeddings, and validation.

Example:
    >>> from mesie.sdk import SpectralIntelligenceSDK
    >>> engine = SpectralIntelligenceSDK()
    >>> corpus = engine.load_corpus("/path/to/spectral/library")
    >>> result = engine.match(reference, candidate)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from mesie.core.config import GenerationConfig
from mesie.core.records import MultiElementRecord
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.generation.fas import generate_fas
from mesie.generation.psd import generate_psd
from mesie.generation.rotdnn import generate_rotdnn
from mesie.io.corpus import SpectralCorpus
from mesie.io.loaders import RecordInput, load_record
from mesie.matching.matcher import MatchResult, SpectralMatcher, match_records
from mesie.processing.normalize import normalize_record
from mesie.validation.validators import ValidationReport, validate_record


class SpectralIntelligenceSDK:
    """Unified entrypoint for MESIE — the Multi-Element Spectral Intelligence Engine.

    Wraps all engine capabilities behind a single, discoverable interface.
    Designed for interactive use, scripting, and integration into larger
    systems.

    Args:
        phase_aware: Enable phase-aware matching by default.
        n_bands: Number of frequency bands for embedding vectorization.

    Example:
        >>> engine = SpectralIntelligenceSDK()
        >>> corpus = engine.load_corpus("./spectral_data")
        >>> record = engine.load("signal.json")
        >>> results = engine.rank(record, top_k=5)
    """

    def __init__(
        self,
        *,
        phase_aware: bool = False,
        n_bands: int = 8,
    ) -> None:
        self._matcher = SpectralMatcher(phase_aware=phase_aware)
        self._vectorizer = SpectralVectorizer(n_bands=n_bands)
        self._corpus: Optional[SpectralCorpus] = None

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def load_corpus(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = True,
        skip_errors: bool = False,
    ) -> SpectralCorpus:
        """Load a spectral library from a directory and fit the matcher.

        Args:
            path: Path to directory containing spectral files.
            recursive: Search subdirectories recursively.
            skip_errors: Skip unloadable files instead of raising.

        Returns:
            The loaded SpectralCorpus.
        """
        self._corpus = SpectralCorpus.from_directory(
            path, recursive=recursive, skip_errors=skip_errors
        )
        self._matcher.fit(list(self._corpus))
        return self._corpus

    @property
    def corpus(self) -> Optional[SpectralCorpus]:
        """The currently loaded corpus, if any."""
        return self._corpus

    # ------------------------------------------------------------------
    # Record loading
    # ------------------------------------------------------------------

    def load(self, source: RecordInput, record_id: Optional[str] = None) -> MultiElementRecord:
        """Load a single spectral record from any supported format.

        Args:
            source: File path, dict, array, or existing record.
            record_id: Optional record identifier override.

        Returns:
            A MultiElementRecord instance.
        """
        return load_record(source, record_id=record_id)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, reference: RecordInput, candidate: RecordInput) -> MatchResult:
        """Match two spectral records and return a composite score.

        Args:
            reference: Reference record.
            candidate: Candidate record.

        Returns:
            MatchResult with score and metric breakdown.
        """
        return self._matcher.score(reference, candidate)

    def rank(self, candidate: RecordInput, top_k: int = 10) -> List[MatchResult]:
        """Rank corpus records against a candidate.

        Requires a corpus to be loaded first via load_corpus().

        Args:
            candidate: The candidate record to rank against.
            top_k: Number of top results to return.

        Returns:
            Sorted list of MatchResults.

        Raises:
            RuntimeError: If no corpus has been loaded.
        """
        if not self._corpus or len(self._corpus) == 0:
            raise RuntimeError(
                "No corpus loaded. Call load_corpus() first."
            )
        return self._matcher.rank_matches(candidate, top_k=top_k)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_psd(self, config: Optional[GenerationConfig] = None, **kwargs) -> MultiElementRecord:
        """Generate a Power Spectral Density record.

        Args:
            config: Optional generation configuration (uses default if None).
            **kwargs: Additional parameters passed to generate_psd.

        Returns:
            Generated MultiElementRecord.
        """
        cfg = config or GenerationConfig()
        return generate_psd(config=cfg, **kwargs)

    def generate_fas(self, config: Optional[GenerationConfig] = None, **kwargs) -> MultiElementRecord:
        """Generate a Fourier Amplitude Spectrum record.

        Args:
            config: Optional generation configuration (uses default if None).
            **kwargs: Additional parameters passed to generate_fas.

        Returns:
            Generated MultiElementRecord.
        """
        cfg = config or GenerationConfig()
        return generate_fas(config=cfg, **kwargs)

    def generate_rotdnn(self, config: Optional[GenerationConfig] = None, **kwargs) -> MultiElementRecord:
        """Generate a RotDnn spectrum record.

        Args:
            config: Optional generation configuration (uses default if None).
            **kwargs: Additional parameters passed to generate_rotdnn.

        Returns:
            Generated MultiElementRecord.
        """
        cfg = config or GenerationConfig()
        return generate_rotdnn(config=cfg, **kwargs)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, records: Union[RecordInput, Sequence[RecordInput]]) -> np.ndarray:
        """Compute spectral embeddings for one or more records.

        Args:
            records: A single record or sequence of records.

        Returns:
            Embedding array of shape (n_records, embedding_dim).
        """
        if isinstance(records, (list, tuple)):
            loaded = [load_record(r) for r in records]
        else:
            loaded = [load_record(records)]
        vectors = [self._vectorizer.transform(r) for r in loaded]
        return np.vstack(vectors)

    # ------------------------------------------------------------------
    # Validation & Normalization
    # ------------------------------------------------------------------

    def validate(self, record: RecordInput) -> ValidationReport:
        """Validate a spectral record.

        Args:
            record: Record to validate.

        Returns:
            ValidationReport with any issues found.
        """
        return validate_record(load_record(record))

    def normalize(self, record: RecordInput) -> MultiElementRecord:
        """Normalize a spectral record.

        Args:
            record: Record to normalize.

        Returns:
            Normalized MultiElementRecord.
        """
        return normalize_record(load_record(record))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Return the MESIE version string."""
        from mesie import __version__
        return __version__

    def __repr__(self) -> str:
        corpus_info = f", corpus={len(self._corpus)} records" if self._corpus else ""
        return f"SpectralIntelligenceSDK(v{self.version}{corpus_info})"
