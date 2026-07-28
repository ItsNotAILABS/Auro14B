"""Cryptographically verified NEXUS Relay evidence adapter for AURO learning."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _public_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def verify_relay_receipt(receipt: Mapping[str, Any], signing_key: str | bytes) -> bool:
    """Verify Relay authorship over the normalized receipt object.

    The Relay issuer signs every receipt field except ``signature_hmac_sha256``.
    A bare 64-character digest is not proof of Relay authorship.
    """
    signature = str(receipt.get("signature_hmac_sha256") or "")
    issuer = str(receipt.get("issuer") or "")
    if issuer != "nexus-relay" or len(signature) != 64:
        return False
    payload = dict(receipt)
    payload.pop("signature_hmac_sha256", None)
    key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    expected = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


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
    relay_issuer: str = ""
    authorship_verified: bool = False
    content_hash_verified: bool = False
    citation_urls_verified: bool = False
    schema: str = "nexus.relay.auro-evidence.v2"

    def validate(self) -> None:
        if not _public_http_url(self.source_url):
            raise ValueError("Relay evidence requires a public HTTP(S) source URL")
        if not _public_http_url(self.final_url):
            raise ValueError("Relay evidence requires a public HTTP(S) final URL")
        if not self.text.strip():
            raise ValueError("Relay evidence text is empty")
        if len(self.source_receipt_sha256) != 64 or len(self.content_sha256) != 64:
            raise ValueError("Relay evidence requires SHA-256 custody fields")
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.content_sha256 or not self.content_hash_verified:
            raise ValueError("Relay content hash is not independently verified")
        if not self.authorship_verified or self.relay_issuer != "nexus-relay":
            raise ValueError("Relay receipt authorship is not cryptographically verified")
        if not self.citations or not self.citation_urls_verified:
            raise ValueError("Relay evidence requires independently validated citation URLs")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Relay evidence confidence must be between 0 and 1")

    def memory_candidate(self) -> dict[str, Any]:
        self.validate()
        candidate = asdict(self)
        candidate["citation_count"] = len(self.citations)
        candidate["candidate_sha256"] = _sha(candidate)
        candidate["status"] = "verified_candidate"
        return candidate


def from_relay_response(response: Mapping[str, Any], signing_key: str | bytes | None = None) -> RelayEvidence:
    """Convert a signed Relay response into a verified model-side candidate."""
    receipt = dict(response.get("receipt") or {})
    intelligence = dict(response.get("intelligence") or {})
    text = str(response.get("text") or response.get("content") or "")
    key = signing_key if signing_key is not None else os.environ.get("NEXUS_RELAY_RECEIPT_KEY", "")
    if not key:
        raise ValueError("NEXUS_RELAY_RECEIPT_KEY is required for Relay evidence admission")
    authorship_verified = verify_relay_receipt(receipt, key)
    content_sha = str(receipt.get("content_sha256") or "")
    content_hash_verified = bool(content_sha) and hmac.compare_digest(
        content_sha,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    citations = tuple(str(x) for x in intelligence.get("citations", []) if x)
    citation_urls_verified = bool(citations) and all(_public_http_url(value) for value in citations)
    evidence = RelayEvidence(
        source_url=str(receipt.get("source_url") or response.get("url") or ""),
        final_url=str(receipt.get("final_url") or receipt.get("source_url") or response.get("url") or ""),
        text=text,
        source_receipt_sha256=str(receipt.get("receipt_sha256") or ""),
        content_sha256=content_sha,
        citations=citations,
        entities=tuple(str(x) for x in intelligence.get("entities", []) if x),
        confidence=float(intelligence.get("confidence", 0.0)),
        relay_request_id=str(receipt.get("request_id") or ""),
        relay_issuer=str(receipt.get("issuer") or ""),
        authorship_verified=authorship_verified,
        content_hash_verified=content_hash_verified,
        citation_urls_verified=citation_urls_verified,
    )
    evidence.validate()
    return evidence


def batch_memory_candidates(
    responses: Sequence[Mapping[str, Any]], signing_key: str | bytes | None = None
) -> list[dict[str, Any]]:
    return [from_relay_response(response, signing_key=signing_key).memory_candidate() for response in responses]
