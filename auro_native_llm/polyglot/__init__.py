"""Polyglot compute — Python + Julia + Haskell + CUDA plane for Auro."""

from auro_native_llm.polyglot.runtimes import PolyglotRuntime, RuntimeReport
from auro_native_llm.polyglot.organ import PolyglotOrgan
from auro_native_llm.polyglot.cuda_plane import CudaPlane, get_cuda_plane
from auro_native_llm.polyglot.entangled import (
    PolyglotOrchestrator,
    get_orchestrator,
    DEFAULT_ROLES,
    RoleKind,
)

__all__ = [
    "CudaPlane",
    "DEFAULT_ROLES",
    "PolyglotOrgan",
    "PolyglotOrchestrator",
    "PolyglotRuntime",
    "RoleKind",
    "RuntimeReport",
    "get_cuda_plane",
    "get_orchestrator",
]
