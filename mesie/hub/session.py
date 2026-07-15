"""Session management for the research hub."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HubSession:
    """A single user/client session on the research hub.

    Tracks state, history, and context for a connected client.

    Attributes:
        session_id: Unique session identifier.
        user: Optional user identifier.
        created_at: Session creation timestamp.
        last_active: Last activity timestamp.
        context: Persistent session context/state.
        history: List of actions performed in this session.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    _max_history: int = 200

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_active = time.time()

    def record_action(self, action: str, params: Dict[str, Any], result: Any = None) -> None:
        """Record an action in session history."""
        self.touch()
        entry = {
            "action": action,
            "params": params,
            "result_summary": str(result)[:200] if result else None,
            "timestamp": time.time(),
        }
        self.history.append(entry)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

    def set_context(self, key: str, value: Any) -> None:
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    @property
    def is_expired(self) -> bool:
        """Session expires after 1 hour of inactivity."""
        return (time.time() - self.last_active) > 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user": self.user,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "history_length": len(self.history),
            "context_keys": list(self.context.keys()),
        }


class SessionManager:
    """Manages multiple concurrent hub sessions.

    Provides session creation, lookup, and cleanup.
    """

    def __init__(self, max_sessions: int = 100) -> None:
        self._sessions: Dict[str, HubSession] = {}
        self._max_sessions = max_sessions

    def create(self, user: Optional[str] = None) -> HubSession:
        """Create a new session."""
        self._cleanup_expired()
        session = HubSession(user=user)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[HubSession]:
        """Get an active session by ID."""
        session = self._sessions.get(session_id)
        if session and session.is_expired:
            del self._sessions[session_id]
            return None
        return session

    def close(self, session_id: str) -> bool:
        """Close and remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_active(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        self._cleanup_expired()
        return [s.to_dict() for s in self._sessions.values()]

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def _cleanup_expired(self) -> None:
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired
        ]
        for sid in expired:
            del self._sessions[sid]
