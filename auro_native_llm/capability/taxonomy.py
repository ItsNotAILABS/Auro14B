"""Eight-status taxonomy for capability claims.

Most assistants compress all of these into \"I can't.\"
That is a magnificent little compression algorithm for destroying useful information.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class CapabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    AVAILABLE_BUT_UNDISCOVERED = "AVAILABLE_BUT_UNDISCOVERED"
    NOT_CURRENTLY_CONFIGURED = "NOT_CURRENTLY_CONFIGURED"
    UNSUPPORTED_BY_DEFAULT_TEMPLATE = "UNSUPPORTED_BY_DEFAULT_TEMPLATE"
    PROHIBITED_BY_POLICY = "PROHIBITED_BY_POLICY"
    BLOCKED_BY_PERMISSION = "BLOCKED_BY_PERMISSION"
    TECHNICALLY_UNAVAILABLE = "TECHNICALLY_UNAVAILABLE"
    UNTESTED = "UNTESTED"
    UNKNOWN = "UNKNOWN"


# What each status *means* for the operator
STATUS_MEANING = {
    CapabilityStatus.AVAILABLE: "Proven present and usable right now.",
    CapabilityStatus.AVAILABLE_BUT_UNDISCOVERED: (
        "Exists in the environment but was not found until discovery ran. "
        "Do not claim missing."
    ),
    CapabilityStatus.NOT_CURRENTLY_CONFIGURED: (
        "Software/runtime exists or can exist, but this project has not wired it yet."
    ),
    CapabilityStatus.UNSUPPORTED_BY_DEFAULT_TEMPLATE: (
        "Default scaffold/template does not include it; custom path may still work."
    ),
    CapabilityStatus.PROHIBITED_BY_POLICY: (
        "Policy/governance forbids the action even if technically possible."
    ),
    CapabilityStatus.BLOCKED_BY_PERMISSION: (
        "OS/user/token permissions prevent access; elevating or re-auth may fix."
    ),
    CapabilityStatus.TECHNICALLY_UNAVAILABLE: (
        "Not present on this machine/platform (e.g. no wheel, no binary)."
    ),
    CapabilityStatus.UNTESTED: (
        "Possibly present; discovery did not exercise a proving probe yet."
    ),
    CapabilityStatus.UNKNOWN: (
        "Insufficient evidence. Run discovery again; do not invent a stereotype."
    ),
}


def classify_message(status: CapabilityStatus | str, detail: str = "") -> str:
    """Human sentence that does NOT collapse to 'I can't'."""
    st = CapabilityStatus(status) if isinstance(status, str) else status
    base = STATUS_MEANING[st]
    if detail:
        return f"[{st.value}] {detail} — {base}"
    return f"[{st.value}] {base}"


def cannot_collapse(status: CapabilityStatus, feature: str, evidence: str) -> str:
    """Anti-pattern guard: expand a would-be 'I can't' into classified form."""
    return (
        f"Feature `{feature}` is not AVAILABLE. "
        f"Status={status.value}. Evidence: {evidence}. "
        f"Meaning: {STATUS_MEANING[status]} "
        f"Next: re-run discover_capabilities() before asserting permanent inability."
    )
