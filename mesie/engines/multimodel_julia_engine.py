"""Multi-Model Julia Engines — Python↔Julia hybrid spectral engines.

These engines combine Python's ecosystem with Julia's numerical performance
to deliver multi-model spectral intelligence. Each engine routes computations
to the optimal runtime (Python or Julia) based on the operation type.

Engines:
- MultiModelValidationEngine: Cross-runtime validation with consensus
- MultiModelMatchingEngine: Hybrid matching using both Python and Julia metrics
- MultiModelEmbeddingEngine: Dual-runtime embedding with fusion
- MultiModelFingerprintEngine: Julia-accelerated fingerprinting with Python ANN
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record, RecordInput
from mesie.polyglot.julia_bridge import JuliaBridge, JuliaBridgeError


class MultiModelValidationEngine(Engine):
    """Multi-model validation — consensus across Python and Julia runtimes.

    Runs validation in both runtimes and produces a consensus result
    with cross-runtime confidence scoring. Detects discrepancies between
    runtimes for higher assurance.
    """

    name = "multimodel_validation"
    capabilities = [
        "validate",
        "validate_consensus",
        "validate_julia",
        "validate_python",
        "cross_validate",
        "health",
    ]

    def __init__(self, julia_backend: str = "auto") -> None:
        self._julia = JuliaBridge(backend=julia_backend)

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "health":
                julia_health = self._safe_julia_health()
                return EngineResponse(True, self.name, action, {
                    "status": "ok",
                    "models": ["python", "julia"],
                    "julia_status": julia_health.get("status", "unavailable"),
                    "engine": self.name,
                })

            if action == "validate_julia":
                rec_data = self._record_to_dict(message.payload["record"])
                result = self._julia.validate(rec_data)
                return EngineResponse(True, self.name, action, {
                    "is_valid": result.is_valid,
                    "level": result.level,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "n_points": result.n_points,
                    "runtime": "julia",
                })

            if action == "validate_python":
                from mesie.validation.validators import validate_record
                rec = load_record(message.payload["record"])
                report = validate_record(rec)
                return EngineResponse(True, self.name, action, {
                    "is_valid": report.is_valid,
                    "level": report.quality_level,
                    "errors": [str(e) for e in report.errors],
                    "warnings": [str(w) for w in report.warnings],
                    "runtime": "python",
                })

            if action in ("validate", "validate_consensus", "cross_validate"):
                return self._consensus_validate(message.payload)

        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))
        except JuliaBridgeError as exc:
            return EngineResponse(False, self.name, action, error=f"Julia error: {exc}")

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _consensus_validate(self, payload: Dict[str, Any]) -> EngineResponse:
        """Run validation in both runtimes and merge results."""
        from mesie.validation.validators import validate_record

        rec = load_record(payload["record"])
        rec_data = self._record_to_dict(payload["record"])

        # Python validation
        t0 = time.perf_counter()
        py_report = validate_record(rec)
        py_ms = (time.perf_counter() - t0) * 1000

        # Julia validation
        t0 = time.perf_counter()
        try:
            jl_result = self._julia.validate(rec_data)
            jl_available = True
        except JuliaBridgeError:
            jl_result = None
            jl_available = False
        jl_ms = (time.perf_counter() - t0) * 1000

        # Consensus
        if jl_available and jl_result is not None:
            agreement = py_report.is_valid == jl_result.is_valid
            confidence = 1.0 if agreement else 0.5
            combined_errors = list(set(
                [str(e) for e in py_report.errors] + jl_result.errors
            ))
            combined_warnings = list(set(
                [str(w) for w in py_report.warnings] + jl_result.warnings
            ))
            is_valid = py_report.is_valid and jl_result.is_valid
        else:
            agreement = None
            confidence = 0.7  # single-runtime confidence
            combined_errors = [str(e) for e in py_report.errors]
            combined_warnings = [str(w) for w in py_report.warnings]
            is_valid = py_report.is_valid

        return EngineResponse(True, self.name, "validate_consensus", {
            "is_valid": is_valid,
            "consensus": agreement,
            "confidence": confidence,
            "errors": combined_errors,
            "warnings": combined_warnings,
            "models_used": ["python", "julia"] if jl_available else ["python"],
            "python_result": {
                "is_valid": py_report.is_valid,
                "level": py_report.level,
                "latency_ms": round(py_ms, 2),
            },
            "julia_result": {
                "is_valid": jl_result.is_valid,
                "level": jl_result.level,
                "latency_ms": round(jl_ms, 2),
            } if jl_available and jl_result else None,
            "engine": self.name,
        })

    def _record_to_dict(self, record_input: Any) -> Dict[str, Any]:
        """Convert record input to a dict suitable for Julia bridge."""
        from mesie.polyglot.contract import record_to_dict
        return record_to_dict(record_input)

    def _safe_julia_health(self) -> Dict[str, Any]:
        try:
            return self._julia.health()
        except JuliaBridgeError:
            return {"status": "unavailable"}


class MultiModelMatchingEngine(Engine):
    """Multi-model matching — hybrid Python+Julia spectral comparison.

    Combines Python's flexible metric computation with Julia's optimized
    numerical routines. Produces fused scores weighted by runtime strengths.
    """

    name = "multimodel_matching"
    capabilities = [
        "match",
        "match_fused",
        "match_julia",
        "match_python",
        "compare_runtimes",
        "health",
    ]

    def __init__(
        self,
        julia_backend: str = "auto",
        fusion_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._julia = JuliaBridge(backend=julia_backend)
        self._weights = fusion_weights or {"python": 0.5, "julia": 0.5}

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "health":
                return EngineResponse(True, self.name, action, {
                    "status": "ok",
                    "models": ["python", "julia"],
                    "fusion_weights": self._weights,
                    "engine": self.name,
                })

            if action == "match_julia":
                a_data = self._record_to_dict(message.payload["record_a"])
                b_data = self._record_to_dict(message.payload["record_b"])
                result = self._julia.match(a_data, b_data)
                return EngineResponse(True, self.name, action, {
                    "composite_score": result.composite_score,
                    "cosine": result.cosine,
                    "rmse": result.rmse,
                    "pearson": result.pearson,
                    "n_compared": result.n_compared,
                    "runtime": "julia",
                })

            if action == "match_python":
                from mesie.matching.matcher import match_records
                a = load_record(message.payload["record_a"])
                b = load_record(message.payload["record_b"])
                result = match_records(a, b)
                return EngineResponse(True, self.name, action, {
                    "composite_score": result.composite_score,
                    "metrics": result.metrics,
                    "runtime": "python",
                })

            if action in ("match", "match_fused", "compare_runtimes"):
                return self._fused_match(message.payload, compare=(action == "compare_runtimes"))

        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))
        except JuliaBridgeError as exc:
            return EngineResponse(False, self.name, action, error=f"Julia error: {exc}")

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _fused_match(self, payload: Dict[str, Any], compare: bool = False) -> EngineResponse:
        """Run matching in both runtimes and fuse scores."""
        from mesie.matching.matcher import match_records

        a = load_record(payload["record_a"])
        b = load_record(payload["record_b"])
        a_data = self._record_to_dict(payload["record_a"])
        b_data = self._record_to_dict(payload["record_b"])

        # Python matching
        t0 = time.perf_counter()
        py_result = match_records(a, b)
        py_ms = (time.perf_counter() - t0) * 1000

        # Julia matching
        t0 = time.perf_counter()
        try:
            jl_result = self._julia.match(a_data, b_data)
            jl_available = True
        except JuliaBridgeError:
            jl_result = None
            jl_available = False
        jl_ms = (time.perf_counter() - t0) * 1000

        # Fused score
        if jl_available and jl_result is not None:
            w_py = self._weights["python"]
            w_jl = self._weights["julia"]
            fused_score = (w_py * py_result.composite_score + w_jl * jl_result.composite_score)
            models_used = ["python", "julia"]
        else:
            fused_score = py_result.composite_score
            models_used = ["python"]

        result_data: Dict[str, Any] = {
            "fused_score": round(fused_score, 6),
            "models_used": models_used,
            "reference_id": a.record_id,
            "candidate_id": b.record_id,
            "python_result": {
                "composite_score": py_result.composite_score,
                "metrics": py_result.metrics,
                "latency_ms": round(py_ms, 2),
            },
            "engine": self.name,
        }

        if jl_available and jl_result is not None:
            result_data["julia_result"] = {
                "composite_score": jl_result.composite_score,
                "cosine": jl_result.cosine,
                "rmse": jl_result.rmse,
                "pearson": jl_result.pearson,
                "n_compared": jl_result.n_compared,
                "latency_ms": round(jl_ms, 2),
            }

        if compare:
            result_data["runtime_agreement"] = (
                abs(py_result.composite_score - jl_result.composite_score) < 0.1
                if jl_available and jl_result else None
            )

        action_name = "compare_runtimes" if compare else "match_fused"
        return EngineResponse(True, self.name, action_name, result_data)

    def _record_to_dict(self, record_input: Any) -> Dict[str, Any]:
        from mesie.polyglot.contract import record_to_dict
        return record_to_dict(record_input)


class MultiModelEmbeddingEngine(Engine):
    """Multi-model embedding — dual-runtime spectral vectorization with fusion.

    Computes embeddings in both Python and Julia, then fuses them via
    concatenation or weighted averaging for richer representations.
    """

    name = "multimodel_embedding"
    capabilities = [
        "embed",
        "embed_fused",
        "embed_julia",
        "embed_python",
        "batch_embed",
        "health",
    ]

    def __init__(
        self,
        julia_backend: str = "auto",
        n_bands: int = 8,
        fusion_mode: str = "concatenate",
    ) -> None:
        self._julia = JuliaBridge(backend=julia_backend)
        self._n_bands = n_bands
        self._fusion_mode = fusion_mode  # "concatenate" or "average"

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "health":
                return EngineResponse(True, self.name, action, {
                    "status": "ok",
                    "models": ["python", "julia"],
                    "fusion_mode": self._fusion_mode,
                    "n_bands": self._n_bands,
                    "engine": self.name,
                })

            if action == "embed_julia":
                rec_data = self._record_to_dict(message.payload["record"])
                n_bands = int(message.payload.get("n_bands", self._n_bands))
                embedding = self._julia.embed(rec_data, n_bands=n_bands)
                return EngineResponse(True, self.name, action, {
                    "embedding": embedding.tolist(),
                    "dim": len(embedding),
                    "runtime": "julia",
                })

            if action == "embed_python":
                from mesie.embeddings.vectorizers import SpectralVectorizer
                rec = load_record(message.payload["record"])
                vectorizer = SpectralVectorizer(n_bands=self._n_bands)
                embedding = vectorizer.transform(rec)
                return EngineResponse(True, self.name, action, {
                    "embedding": embedding.tolist(),
                    "dim": len(embedding),
                    "runtime": "python",
                })

            if action in ("embed", "embed_fused"):
                return self._fused_embed(message.payload)

            if action == "batch_embed":
                records = message.payload.get("records", [])
                results = []
                for r in records:
                    rec_data = self._record_to_dict(r)
                    embedding = self._julia.embed(rec_data, n_bands=self._n_bands)
                    results.append(embedding.tolist())
                return EngineResponse(True, self.name, action, {
                    "embeddings": results,
                    "count": len(results),
                    "runtime": "julia",
                })

        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))
        except JuliaBridgeError as exc:
            return EngineResponse(False, self.name, action, error=f"Julia error: {exc}")

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _fused_embed(self, payload: Dict[str, Any]) -> EngineResponse:
        """Compute embeddings in both runtimes and fuse them."""
        from mesie.embeddings.vectorizers import SpectralVectorizer

        rec = load_record(payload["record"])
        rec_data = self._record_to_dict(payload["record"])
        n_bands = int(payload.get("n_bands", self._n_bands))

        # Python embedding
        vectorizer = SpectralVectorizer(n_bands=n_bands)
        py_emb = vectorizer.transform(rec)

        # Julia embedding
        try:
            jl_emb = self._julia.embed(rec_data, n_bands=n_bands)
            jl_available = True
        except JuliaBridgeError:
            jl_emb = None
            jl_available = False

        # Fuse
        if jl_available and jl_emb is not None:
            if self._fusion_mode == "concatenate":
                fused = np.concatenate([py_emb, jl_emb])
            else:  # average
                # Pad to same length
                max_dim = max(len(py_emb), len(jl_emb))
                py_padded = np.zeros(max_dim)
                py_padded[:len(py_emb)] = py_emb
                jl_padded = np.zeros(max_dim)
                jl_padded[:len(jl_emb)] = jl_emb
                fused = (py_padded + jl_padded) / 2.0
            # L2 normalize
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            models_used = ["python", "julia"]
        else:
            fused = py_emb
            models_used = ["python"]

        return EngineResponse(True, self.name, "embed_fused", {
            "embedding": fused.tolist(),
            "dim": len(fused),
            "fusion_mode": self._fusion_mode,
            "models_used": models_used,
            "record_id": rec.record_id,
            "engine": self.name,
        })

    def _record_to_dict(self, record_input: Any) -> Dict[str, Any]:
        from mesie.polyglot.contract import record_to_dict
        return record_to_dict(record_input)


class MultiModelFingerprintEngine(Engine):
    """Multi-model fingerprinting — Julia-accelerated binary fingerprints with Python ANN.

    Uses Julia for fast binary fingerprint computation and Python for
    approximate nearest neighbor indexing and retrieval.
    """

    name = "multimodel_fingerprint"
    capabilities = [
        "fingerprint",
        "fingerprint_julia",
        "fingerprint_python",
        "batch_fingerprint",
        "hamming_distance",
        "health",
    ]

    def __init__(self, julia_backend: str = "auto", resolution: int = 16) -> None:
        self._julia = JuliaBridge(backend=julia_backend)
        self._resolution = resolution
        self._index: Dict[str, np.ndarray] = {}

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "health":
                return EngineResponse(True, self.name, action, {
                    "status": "ok",
                    "models": ["python", "julia"],
                    "resolution": self._resolution,
                    "indexed": len(self._index),
                    "engine": self.name,
                })

            if action in ("fingerprint", "fingerprint_julia"):
                rec_data = self._record_to_dict(message.payload["record"])
                resolution = int(message.payload.get("resolution", self._resolution))
                fp = self._julia.fingerprint(rec_data, resolution=resolution)
                rec = load_record(message.payload["record"])
                self._index[rec.record_id] = fp
                return EngineResponse(True, self.name, action, {
                    "fingerprint": fp.tolist(),
                    "resolution": resolution,
                    "record_id": rec.record_id,
                    "runtime": "julia",
                })

            if action == "fingerprint_python":
                from mesie.embeddings.fingerprint import SpectralFingerprintPipeline
                rec = load_record(message.payload["record"])
                pipeline = SpectralFingerprintPipeline()
                fp_result = pipeline.process(rec)
                return EngineResponse(True, self.name, action, {
                    "fingerprint": fp_result.to_dict(),
                    "record_id": rec.record_id,
                    "runtime": "python",
                })

            if action == "batch_fingerprint":
                records = message.payload.get("records", [])
                results = []
                for r in records:
                    rec_data = self._record_to_dict(r)
                    fp = self._julia.fingerprint(rec_data, resolution=self._resolution)
                    rec = load_record(r)
                    self._index[rec.record_id] = fp
                    results.append({"record_id": rec.record_id, "fingerprint": fp.tolist()})
                return EngineResponse(True, self.name, action, {
                    "fingerprints": results,
                    "count": len(results),
                    "runtime": "julia",
                })

            if action == "hamming_distance":
                id_a = message.payload["record_id_a"]
                id_b = message.payload["record_id_b"]
                if id_a not in self._index or id_b not in self._index:
                    return EngineResponse(False, self.name, action,
                                          error="Record(s) not fingerprinted yet")
                fp_a = self._index[id_a]
                fp_b = self._index[id_b]
                distance = int(np.sum(fp_a != fp_b))
                similarity = 1.0 - (distance / max(len(fp_a), 1))
                return EngineResponse(True, self.name, action, {
                    "hamming_distance": distance,
                    "similarity": round(similarity, 4),
                    "resolution": len(fp_a),
                    "record_id_a": id_a,
                    "record_id_b": id_b,
                })

        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))
        except JuliaBridgeError as exc:
            return EngineResponse(False, self.name, action, error=f"Julia error: {exc}")

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _record_to_dict(self, record_input: Any) -> Dict[str, Any]:
        from mesie.polyglot.contract import record_to_dict
        return record_to_dict(record_input)
