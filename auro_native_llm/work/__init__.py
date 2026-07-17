"""Auro work agents — native LLM that acts (Chrome, code, reason), not chat-only."""

from auro_native_llm.work.agent import WorkAgent, WorkResult
from auro_native_llm.work.algorithms import (
    code_complete,
    extract_code_blocks,
    plan_from_text,
    reason_steps,
    sample_logits,
)

__all__ = [
    "WorkAgent",
    "WorkResult",
    "code_complete",
    "extract_code_blocks",
    "plan_from_text",
    "reason_steps",
    "sample_logits",
]
