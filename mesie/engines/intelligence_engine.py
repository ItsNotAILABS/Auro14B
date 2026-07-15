"""Intelligence engine — reasoning and memory objects."""

from __future__ import annotations

from typing import Optional

from mesie.ai.intelligence_protocols import IntelligenceConfig, IntelligenceProtocol
from mesie.cognitive.memory_adapter import SpectralMemoryAdapter
from mesie.engines.base import Engine
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record


class IntelligenceEngine(Engine):
    name = "intelligence"
    capabilities = ["reason", "memory", "observe"]

    def __init__(self) -> None:
        self._protocol = IntelligenceProtocol(IntelligenceConfig())
        self._memory = SpectralMemoryAdapter()
        self._vectorizer = SpectralVectorizer()

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            rec = load_record(message.payload["record"])
            amp = rec.components[0].amplitude

            if action == "observe":
                self._protocol.observe(amp)
                return EngineResponse(True, self.name, action, {"observed": True, "record_id": rec.record_id})

            if action == "reason":
                emb = self._vectorizer.transform(rec)
                result = self._protocol.reason(emb)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "conclusion": result.conclusion,
                        "confidence": result.confidence,
                        "evidence": getattr(result, "evidence", {}),
                    },
                )

            if action == "memory":
                obj = self._memory.to_memory_object(rec)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {"keys": list(obj.keys()), "semantic_id": obj.get("semantic_id")},
                )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")