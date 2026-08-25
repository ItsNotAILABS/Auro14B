"""AURO browser-WebGPU training substrate."""
from .coordinator import (
    Cluster,
    CoordinatorHandler,
    decode_f32,
    encode_f32,
    flatten_matrix,
    serve,
)

__all__ = [
    "Cluster",
    "CoordinatorHandler",
    "decode_f32",
    "encode_f32",
    "flatten_matrix",
    "serve",
]
