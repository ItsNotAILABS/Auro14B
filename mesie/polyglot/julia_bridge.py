"""Julia bridge — call MESIE Julia spectral functions from Python.

Supports two backends:
1. **juliacall** (preferred): In-process Julia via the juliacall package.
2. **subprocess**: Falls back to JSON IPC over stdin/stdout with the Julia CLI.

Usage::

    from mesie.polyglot.julia_bridge import JuliaBridge

    bridge = JuliaBridge()  # auto-selects best backend
    result = bridge.validate(record_dict)
    score = bridge.match(record_a, record_b)
    embedding = bridge.embed(record_dict)
    fingerprint = bridge.fingerprint(record_dict)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_JULIA_BINDINGS = Path(__file__).resolve().parents[3] / "bindings" / "julia" / "MESIEPolyglot"
# Fallback: search relative to the repo root if running from an installed package
if not _JULIA_BINDINGS.exists():
    _alt = Path(__file__).resolve().parents[2] / "bindings" / "julia" / "MESIEPolyglot"
    if _alt.exists():
        _JULIA_BINDINGS = _alt
_CLI_PATH = _JULIA_BINDINGS / "cli.jl"
_MODULE_PATH = _JULIA_BINDINGS / "src" / "MESIEPolyglot.jl"


class JuliaBackend:
    """Abstract interface for Julia execution backends."""

    def call(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class SubprocessBackend(JuliaBackend):
    """Execute Julia via subprocess JSON IPC."""

    def __init__(self, julia_exec: str = "julia", cli_path: Optional[Path] = None):
        self.julia_exec = julia_exec
        self.cli_path = cli_path or _CLI_PATH

    def call(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = {"action": action, **payload}
        input_json = json.dumps(request)

        try:
            result = subprocess.run(
                [self.julia_exec, str(self.cli_path)],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise JuliaBridgeError(
                f"Julia executable not found: {self.julia_exec}. "
                "Install Julia from https://julialang.org/downloads/"
            )
        except subprocess.TimeoutExpired:
            raise JuliaBridgeError("Julia subprocess timed out after 30s")

        if result.returncode != 0:
            raise JuliaBridgeError(f"Julia process failed: {result.stderr.strip()}")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise JuliaBridgeError(f"Invalid JSON from Julia: {e}")


class JuliaCallBackend(JuliaBackend):
    """Execute Julia in-process via juliacall (zero-copy when possible)."""

    def __init__(self) -> None:
        self._jl: Any = None
        self._dispatch_fn: Any = None

    def _ensure_initialized(self) -> None:
        if self._jl is not None:
            return

        try:
            from juliacall import Main as jl  # type: ignore[import]
        except ImportError:
            raise JuliaBridgeError(
                "juliacall not installed. Install with: pip install juliacall"
            )

        # Load the MESIEPolyglot module and cache the dispatch function
        module_path = str(_MODULE_PATH).replace("\\", "/")
        jl.seval(f'include("{module_path}")')
        jl.seval("using JSON")
        self._jl = jl
        # Cache a reference to the dispatch function to avoid repeated seval
        self._dispatch_fn = jl.seval(
            "req_json -> JSON.json(MESIEPolyglot.dispatch_action(JSON.parse(req_json)))"
        )

    def call(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized()
        # Serialize request to JSON and pass as a single string argument
        # to avoid any code injection via string interpolation
        request = {"action": action, **payload}
        request_json = json.dumps(request)
        result_json = self._dispatch_fn(request_json)
        return json.loads(str(result_json))


class JuliaBridgeError(Exception):
    """Raised when Julia bridge encounters an error."""


@dataclass
class JuliaValidationResult:
    """Result from Julia spectral validation."""

    is_valid: bool
    level: int
    errors: List[str]
    warnings: List[str]
    n_points: int
    runtime: str = "julia"


@dataclass
class JuliaMatchResult:
    """Result from Julia spectral matching."""

    composite_score: float
    cosine: float
    rmse: float
    pearson: float
    n_compared: int
    runtime: str = "julia"


class JuliaBridge:
    """High-level Python↔Julia bridge for MESIE spectral operations.

    Automatically selects the best available backend:
    - juliacall (in-process, fast) if installed
    - subprocess (JSON IPC) as fallback

    Parameters
    ----------
    backend : str, optional
        Force a specific backend: "juliacall", "subprocess", or "auto" (default).
    julia_exec : str, optional
        Path to julia executable for subprocess backend.
    """

    def __init__(
        self,
        backend: str = "auto",
        julia_exec: str = "julia",
    ) -> None:
        self._backend = self._init_backend(backend, julia_exec)
        self._backend_name = backend if backend != "auto" else self._detect_backend_name()

    def _init_backend(self, backend: str, julia_exec: str) -> JuliaBackend:
        if backend == "juliacall":
            return JuliaCallBackend()
        elif backend == "subprocess":
            return SubprocessBackend(julia_exec=julia_exec)
        elif backend == "auto":
            try:
                import juliacall  # noqa: F401
                logger.info("Using juliacall backend for Julia bridge")
                return JuliaCallBackend()
            except ImportError:
                logger.info("juliacall not available, using subprocess backend")
                return SubprocessBackend(julia_exec=julia_exec)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'juliacall', 'subprocess', or 'auto'.")

    def _detect_backend_name(self) -> str:
        if isinstance(self._backend, JuliaCallBackend):
            return "juliacall"
        return "subprocess"

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend_name

    def health(self) -> Dict[str, Any]:
        """Check Julia runtime health."""
        response = self._backend.call("health", {})
        return response.get("data", response)

    def validate(self, record: Dict[str, Any]) -> JuliaValidationResult:
        """Validate a spectral record using Julia.

        Parameters
        ----------
        record : dict
            Spectral record with frequency/amplitude arrays or components list.

        Returns
        -------
        JuliaValidationResult
        """
        response = self._backend.call("validate", {"record": record})
        data = response.get("data", response)
        return JuliaValidationResult(
            is_valid=data.get("is_valid", False),
            level=data.get("level", 0),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            n_points=data.get("n_points", 0),
            runtime=data.get("runtime", "julia"),
        )

    def match(self, record_a: Dict[str, Any], record_b: Dict[str, Any]) -> JuliaMatchResult:
        """Match two spectral records using Julia's optimized routines.

        Parameters
        ----------
        record_a : dict
            First spectral record.
        record_b : dict
            Second spectral record.

        Returns
        -------
        JuliaMatchResult
        """
        response = self._backend.call("match", {"record_a": record_a, "record_b": record_b})
        data = response.get("data", response)
        metrics = data.get("metrics", {})
        return JuliaMatchResult(
            composite_score=data.get("composite_score", 0.0),
            cosine=metrics.get("cosine", 0.0),
            rmse=metrics.get("rmse", 1.0),
            pearson=metrics.get("pearson", 0.0),
            n_compared=data.get("n_compared", 0),
            runtime=data.get("runtime", "julia"),
        )

    def embed(self, record: Dict[str, Any], n_bands: int = 8) -> np.ndarray:
        """Compute spectral embedding using Julia.

        Parameters
        ----------
        record : dict
            Spectral record.
        n_bands : int
            Number of frequency bands for embedding (default 8).

        Returns
        -------
        numpy.ndarray
            L2-normalized embedding vector of dimension n_bands * 4.
        """
        response = self._backend.call("embed", {"record": record, "n_bands": n_bands})
        data = response.get("data", response)
        embedding = data.get("embedding", [])
        return np.array(embedding, dtype=np.float64)

    def fingerprint(self, record: Dict[str, Any], resolution: int = 16) -> np.ndarray:
        """Compute binary spectral fingerprint using Julia.

        Parameters
        ----------
        record : dict
            Spectral record.
        resolution : int
            Fingerprint resolution in bits (default 16).

        Returns
        -------
        numpy.ndarray
            Binary fingerprint array of uint8 values (0 or 1).
        """
        response = self._backend.call("fingerprint", {"record": record, "resolution": resolution})
        data = response.get("data", response)
        fp = data.get("fingerprint", [])
        return np.array(fp, dtype=np.uint8)

    def batch_validate(self, records: List[Dict[str, Any]]) -> List[JuliaValidationResult]:
        """Validate multiple records. Returns list of validation results."""
        return [self.validate(r) for r in records]

    def batch_embed(self, records: List[Dict[str, Any]], n_bands: int = 8) -> np.ndarray:
        """Embed multiple records. Returns matrix of shape (n_records, n_bands*4)."""
        embeddings = [self.embed(r, n_bands=n_bands) for r in records]
        return np.vstack(embeddings) if embeddings else np.empty((0, n_bands * 4))
