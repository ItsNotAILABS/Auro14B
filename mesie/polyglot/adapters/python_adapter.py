"""Python native adapter — full MESIE stack."""

from __future__ import annotations

from typing import List, Optional, Tuple

from mesie import match_records, validate_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.io.loaders import load_record
from mesie.matching.ranking import rank_candidates
from mesie.polyglot.adapters.base import PolyglotAdapter
from mesie.polyglot.contract import AISVectorMessage, PolyglotAction, RuntimeId


class PythonAdapter(PolyglotAdapter):
    runtime = RuntimeId.PYTHON

    def __init__(self, n_bands: int = 8) -> None:
        self._vectorizer = SpectralVectorizer(n_bands=n_bands)

    def available(self) -> bool:
        return True

    def _handle(self, message: AISVectorMessage) -> Tuple[dict, Optional[List[float]]]:
        action = message.action
        if action == PolyglotAction.HEALTH:
            return {"status": "ok", "engine": "mesie-python"}, None

        if action == PolyglotAction.VALIDATE:
            rec = load_record(message.record or {})
            rep = validate_record(rec)
            return {
                "is_valid": rep.is_valid,
                "level": rep.level,
                "errors": rep.errors[:10],
                "warnings": rep.warnings[:10],
            }, None

        if action == PolyglotAction.MATCH:
            a = load_record(message.record_a or message.record or {})
            b = load_record(message.record_b or {})
            result = match_records(a, b)
            return {
                "composite_score": result.composite_score,
                "metrics": result.metrics,
                "reference_id": result.reference_id,
                "candidate_id": result.candidate_id,
            }, None

        if action == PolyglotAction.EMBED:
            rec = load_record(message.record or {})
            vec = self._vectorizer.transform(rec)
            return {"dim": len(vec), "record_id": rec.record_id}, vec.tolist()

        if action == PolyglotAction.RANK:
            query = load_record(message.record or {})
            pool = [load_record(c) for c in message.candidates]
            ranked = rank_candidates(query, pool, top_k=message.top_k)
            return {
                "ranked": [
                    {"candidate_id": r.candidate_id, "score": r.composite_score}
                    for r in ranked
                ],
            }, None

        raise ValueError(f"Unsupported action for python: {action}")