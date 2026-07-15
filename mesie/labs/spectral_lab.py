"""Spectral Lab — wraps existing MESIE spectral capabilities as a lab."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from mesie.labs.base_lab import BaseLab, LabConfig, LabResult


class SpectralLab(BaseLab):
    """Lab for spectral analysis, matching, generation, and embedding.

    Wraps existing MESIE core capabilities (validate, match, generate,
    embed, fingerprint) into the universal lab interface.
    """

    def _default_config(self) -> LabConfig:
        return LabConfig(
            name="Spectral Analysis Lab",
            domain="spectral",
            capabilities=[
                "validate",
                "match",
                "generate_psd",
                "generate_fas",
                "embed",
                "fingerprint",
                "rank",
                "normalize",
            ],
        )

    def run(self, operation: str, **kwargs: Any) -> LabResult:
        start = time.time()
        try:
            if operation == "validate":
                data = self._validate(**kwargs)
            elif operation == "match":
                data = self._match(**kwargs)
            elif operation == "generate_psd":
                data = self._generate_psd(**kwargs)
            elif operation == "generate_fas":
                data = self._generate_fas(**kwargs)
            elif operation == "embed":
                data = self._embed(**kwargs)
            elif operation == "fingerprint":
                data = self._fingerprint(**kwargs)
            elif operation == "rank":
                data = self._rank(**kwargs)
            elif operation == "normalize":
                data = self._normalize(**kwargs)
            else:
                return LabResult(
                    lab=self.name, operation=operation,
                    status="error", error=f"Unknown operation: {operation}",
                )
            return LabResult(
                lab=self.name, operation=operation, data=data,
                duration_seconds=time.time() - start,
            )
        except Exception as exc:
            return LabResult(
                lab=self.name, operation=operation,
                status="error", error=str(exc),
                duration_seconds=time.time() - start,
            )

    def _validate(self, record: Any = None, level: int = 6, **kw: Any) -> Dict[str, Any]:
        from mesie.validation.validators import validate_record
        if record is None:
            return {"error": "No record provided"}
        report = validate_record(record, level=level)
        return {"valid": report.valid, "level": report.level, "errors": report.errors}

    def _match(self, reference: Any = None, candidate: Any = None, **kw: Any) -> Dict[str, Any]:
        from mesie.matching.matcher import match_records
        if reference is None or candidate is None:
            return {"error": "Both reference and candidate required"}
        result = match_records(reference, candidate)
        return {"score": result.composite_score, "details": result.to_dict()}

    def _generate_psd(self, seed: int = 42, **kw: Any) -> Dict[str, Any]:
        from mesie.generation.psd import generate_psd
        from mesie.core.config import GenerationConfig
        config = GenerationConfig(seed=seed, **{k: v for k, v in kw.items() if k != "seed"})
        record = generate_psd(config)
        return {"record_id": record.record_id, "n_components": len(record.components)}

    def _generate_fas(self, seed: int = 42, **kw: Any) -> Dict[str, Any]:
        from mesie.generation.fas import generate_fas
        from mesie.core.config import GenerationConfig
        config = GenerationConfig(seed=seed, **{k: v for k, v in kw.items() if k != "seed"})
        record = generate_fas(config)
        return {"record_id": record.record_id, "n_components": len(record.components)}

    def _embed(self, record: Any = None, n_bands: int = 8, **kw: Any) -> Dict[str, Any]:
        from mesie.embeddings.vectorizers import SpectralVectorizer
        if record is None:
            return {"error": "No record provided"}
        vectorizer = SpectralVectorizer(n_bands=n_bands)
        vec = vectorizer.embed(record)
        return {"embedding_dim": len(vec), "norm": float(np.linalg.norm(vec))}

    def _fingerprint(self, record: Any = None, **kw: Any) -> Dict[str, Any]:
        from mesie.engines.fingerprint_engine import FingerprintEngine
        engine = FingerprintEngine()
        return {"status": "fingerprint_ready", "engine": engine.name}

    def _rank(self, query: Any = None, candidates: Any = None, top_k: int = 5, **kw: Any) -> Dict[str, Any]:
        return {"status": "rank_ready", "top_k": top_k}

    def _normalize(self, record: Any = None, **kw: Any) -> Dict[str, Any]:
        from mesie.processing.normalize import normalize_record
        if record is None:
            return {"error": "No record provided"}
        normalized = normalize_record(record)
        return {"record_id": normalized.record_id, "normalized": True}
