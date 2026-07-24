"""NEXUS Relay evidence adapter for AURO continuous learning.

Relay remains a product lane. This module accepts only normalized, receipt-
bearing evidence and converts it into AURO memory candidates. It never imports
Relay billing, dashboard, or deployment code into the model core.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class RelayEvidence:
    source_url: str
    final_url: str
    text: str
    source_receipt_sha256: str
    content_sha256: str
    citations: tuple[str, ...]
    entities: tuple[str, ...]
    confidence: float
    relay_request_id: str = ""
    schema: str = "nexus.relay.auro-evidence.v1"

    def validate(self) -> None:
        if not self.source_url.startswith(("http://", "https://")):
            raise ValueError("Relay evidence requires a public HTTP(S) source URL")
        if not self.final_url.startswith(("http://", "https://")):
            raise ValueError("Relay evidence requires a public HTTP(S) final URL")
        if not self.text.strip():
            raise ValueError("Relay evidence text is empty")
        if len(self.source_receipt_sha256) != 64 or len(self.content_sha256) != 64:
            raise ValueError("Relay evidence requires 64-character SHA-256 receipts")
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.content_sha256:
            raise ValueError("Relay content hash does not match normalized text")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Relay evidence confidence must be between 0 and 1")

    def memory_candidate(self) -> dict[str, Any]:
        self.validate()
        candidate = asdict(self)
        candidate["citation_count"] = len(self.citations)
        candidate["candidate_sha256"] = _sha(candidate)
        candidate["status"] = "candidate"
        return candidate


def from_relay_response(response: Mapping[str, Any]) -> RelayEvidence:
    """Convert the stable Relay read contract into model-side evidence."""
    receipt = dict(response.get("receipt") or {})
    intelligence = dict(response.get("intelligence") or {})
    text = str(response.get("text") or response.get("content") or "")
    return RelayEvidence(
        source_url=str(receipt.get("source_url") or response.get("url") or ""),
        final_url=str(receipt.get("final_url") or receipt.get("source_url") or response.get("url") or ""),
        text=text,
        source_receipt_sha256=str(receipt.get("receipt_sha256") or ""),
        content_sha256=str(receipt.get("content_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()),
        citations=tuple(str(x) for x in intelligence.get("citations", []) if x),
        entities=tuple(str(x) for x in intelligence.get("entities", []) if x),
        confidence=float(intelligence.get("confidence", 0.0)),
        relay_request_id=str(receipt.get("request_id") or ""),
    )


def batch_memory_candidates(responses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for response in responses:
        candidates.append(from_relay_response(response).memory_candidate())
    return candidates
