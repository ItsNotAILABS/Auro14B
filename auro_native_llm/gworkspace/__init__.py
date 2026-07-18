"""Google Virtual Workspace — collab + AI sandbox envelope.

Two surfaces, one system:

1. **Collab** — place where you and the AI work projects together
   (shared docs, tasks, notes, browser sessions, messages).

2. **Sandbox Envelope** — the AI's own virtual Google stack:
   Chrome tabs, Search, Gmail, Drive/Docs, Calendar, Sites —
   isolated profile, receipted actions, no operator credentials leaked.

Real Chrome CDP is used when available for public web navigation;
mail/drive/calendar are sandboxed virtual services the AI can fully control.
"""

from auro_native_llm.gworkspace.envelope import GoogleVirtualEnvelope, get_envelope
from auro_native_llm.gworkspace.collab import CollabWorkspace, get_collab
from auro_native_llm.gworkspace.suite import GoogleSuite

__all__ = [
    "GoogleVirtualEnvelope",
    "get_envelope",
    "CollabWorkspace",
    "get_collab",
    "GoogleSuite",
]
