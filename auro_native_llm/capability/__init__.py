"""Capability-state discovery — ban 'I can't' compression.

Before any inability claim, run discovery. Classify every constraint
with the eight-status taxonomy (never collapse to a single no).
"""

from auro_native_llm.capability.taxonomy import CapabilityStatus, classify_message
from auro_native_llm.capability.discover import (
    CapabilityReport,
    discover_capabilities,
    run_discovery,
)

__all__ = [
    "CapabilityStatus",
    "classify_message",
    "CapabilityReport",
    "discover_capabilities",
    "run_discovery",
]
