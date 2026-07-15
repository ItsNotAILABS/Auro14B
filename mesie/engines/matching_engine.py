"""Matching engine — compare and rank spectral records."""

from __future__ import annotations

from typing import Optional

from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record
from mesie.matching.matcher import match_records
from mesie.matching.ranking import rank_candidates


class MatchingEngine(Engine):
    name = "matching"
    capabilities = ["match", "rank"]

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "match":
                a = load_record(message.payload["record_a"])
                b = load_record(message.payload["record_b"])
                result = match_records(a, b)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "composite_score": result.composite_score,
                        "metrics": result.metrics,
                    },
                )

            if action == "rank":
                query = load_record(message.payload["query"])
                candidates = [load_record(c) for c in message.payload["candidates"]]
                top_k = int(message.payload.get("top_k", 5))
                ranked = rank_candidates(query, candidates, top_k=top_k)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "ranked": [
                            {"record_id": r.candidate_id, "score": r.score}
                            for r in ranked
                        ],
                    },
                )
        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")