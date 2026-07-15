"""Library bridge — unifies all MESIE library modules into a single access layer.

Provides a centralized state-aware bridge that manages cross-library
communication, shared state, and event routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.embeddings.encoders import SpectralFeatureEncoder
from mesie.features.electro_spectral import ElectroSpectralLayer
from mesie.matching.matcher import SpectralMatcher, MatchResult


class BridgeState(Enum):
    """State of the library bridge."""

    IDLE = "idle"
    CONNECTED = "connected"
    PROCESSING = "processing"
    ERROR = "error"


class LibraryBridge:
    """Centralized bridge unifying MESIE library access.

    Manages shared state across embeddings, matching, features,
    and encoding subsystems. Supports event hooks for cross-library
    communication.

    Args:
        vectorizer: Shared SpectralVectorizer.
    """

    def __init__(self, vectorizer: Optional[SpectralVectorizer] = None) -> None:
        self._vectorizer = vectorizer or SpectralVectorizer()
        self._encoder = SpectralFeatureEncoder()
        self._electro = ElectroSpectralLayer()
        self._matcher = SpectralMatcher()
        self._state = BridgeState.IDLE
        self._event_hooks: Dict[str, List[Callable[..., Any]]] = {}
        self._shared_state: Dict[str, Any] = {}
        self._records_cache: List[MultiElementRecord] = []

    @property
    def state(self) -> BridgeState:
        """Current bridge state."""
        return self._state

    def activate(self) -> None:
        """Activate the bridge for processing."""
        self._state = BridgeState.CONNECTED

    def register_hook(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for cross-library events.

        Args:
            event: Event name.
            callback: Function to call when event fires.
        """
        if event not in self._event_hooks:
            self._event_hooks[event] = []
        self._event_hooks[event].append(callback)

    def _fire_event(self, event: str, **kwargs: Any) -> None:
        """Fire an event to all registered hooks."""
        for cb in self._event_hooks.get(event, []):
            cb(**kwargs)

    def set_shared(self, key: str, value: Any) -> None:
        """Set a shared state value accessible across libraries.

        Args:
            key: State key.
            value: State value.
        """
        self._shared_state[key] = value
        self._fire_event("state_updated", key=key, value=value)

    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get a shared state value.

        Args:
            key: State key.
            default: Default value if key not found.

        Returns:
            The state value or default.
        """
        return self._shared_state.get(key, default)

    def process_record(self, record: RecordInput) -> Dict[str, Any]:
        """Process a record through all bridged libraries.

        Args:
            record: Input spectral record.

        Returns:
            Combined result dictionary from all libraries.
        """
        self._state = BridgeState.PROCESSING
        try:
            rec = load_record(record)
            self._records_cache.append(rec)

            embedding = self._vectorizer.transform(rec)
            features = self._encoder.encode(rec)
            signature = self._electro.compute_signature(rec)

            result = {
                "record_id": rec.record_id,
                "embedding": embedding,
                "features": features,
                "signature": {
                    "centroid": signature.spectral_centroid,
                    "spread": signature.spectral_spread,
                    "resonance": signature.frequency_resonance,
                    "coherence": signature.coherence_signature,
                },
            }

            self._fire_event("record_processed", record_id=rec.record_id, result=result)
            self._state = BridgeState.CONNECTED
            return result

        except Exception:
            self._state = BridgeState.ERROR
            raise

    def match_records(
        self,
        reference: RecordInput,
        candidates: Sequence[RecordInput],
    ) -> List[MatchResult]:
        """Match a reference record against candidates using bridged systems.

        Args:
            reference: Reference spectral record.
            candidates: Candidate records to compare.

        Returns:
            List of match results.
        """
        ref = load_record(reference)
        self._matcher.fit([ref])
        return [self._matcher.match(c) for c in candidates]

    @property
    def cache_size(self) -> int:
        """Number of records in cache."""
        return len(self._records_cache)

    @property
    def vectorizer(self) -> SpectralVectorizer:
        """Shared vectorizer instance."""
        return self._vectorizer
