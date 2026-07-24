"""ChaosCUDA — sovereign local accelerator for machines without NVIDIA CUDA.

Novel Chaos Labs plane: multi-thread blocked GEMM + Julia spectral engines +
polyglot teachers. Works on Windows ARM64 today.
"""

from auro_native_llm.chaos_cuda.plane import ChaosCudaPlane, get_chaos_cuda

__all__ = ["ChaosCudaPlane", "get_chaos_cuda"]
