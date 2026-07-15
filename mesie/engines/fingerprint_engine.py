"""Fingerprint engine — TF, salient, LSH, ANN on internal API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mesie.embeddings.fingerprint import SpectralFingerprintPipeline
from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record, RecordInput


class FingerprintEngine(Engine):
    name = "fingerprint"
    capabilities = ["process", "index", "query", "ann_status", "explain"]

    def __init__(self) -> None:
        self._pipeline = SpectralFingerprintPipeline()

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "process":
                rec = load_record(message.payload["record"])
                fp = self._pipeline.process(rec)
                return EngineResponse(True, self.name, action, fp.to_dict())

            if action == "index":
                records = [load_record(r) for r in message.payload["records"]]
                n = self._pipeline.index_records(records)
                return EngineResponse(True, self.name, action, {"indexed": n})

            if action == "query":
                rec = load_record(message.payload["record"])
                top_k = int(message.payload.get("top_k", 5))
                hits = self._pipeline.query(rec, top_k=top_k)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "hits": [
                            {
                                "item_id": h.item_id,
                                "distance": round(h.distance, 4),
                                "similarity": round(h.similarity, 4),
                                "lsh_bucket": h.lsh_bucket,
                            }
                            for h in hits
                        ],
                    },
                )

            if action == "ann_status":
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "index_size": self._pipeline.store.index.size,
                        "fingerprints": len(self._pipeline.store.fingerprints),
                        "use_lsh": self._pipeline.store.index.use_lsh,
                    },
                )

            if action == "explain":
                qid = message.payload["query_id"]
                hid = message.payload["hit_id"]
                return EngineResponse(True, self.name, action, self._pipeline.explain_match(qid, hid))

        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")