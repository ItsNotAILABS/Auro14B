"""MESIE Virtual Processor — prompts as measurable work calls.

Hybrid doctrine (ItsNotAILabs / GHOST):
  MESIE/Ghost = deterministic numerical/signal path (default)
  LLM         = escalate only for planning/NL when justified
"""

from auro_native_llm.vproc.processor import (
    MesieVirtualProcessor,
    WorkCall,
    WorkMetrics,
    run_work_call,
)
from auro_native_llm.vproc.hybrid import HybridRuntime, hybrid_execute

__all__ = [
    "MesieVirtualProcessor",
    "WorkCall",
    "WorkMetrics",
    "run_work_call",
    "HybridRuntime",
    "hybrid_execute",
]
