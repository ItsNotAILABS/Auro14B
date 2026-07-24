import hashlib
import hmac
import json
from dataclasses import asdict

import pytest

from auro_native_llm.continuous.receipts import AgentReceipt, ReceiptStore, canonical_json
from auro_native_llm.continuous.relay_bridge import from_relay_response


def make_receipt(run_id: str, agent: str) -> AgentReceipt:
    return AgentReceipt(
        run_id=run_id,
        agent=agent,
        status="passed",
        started_at_unix=1,
        completed_at_unix=2,
        outputs={"ok": True},
    )


def test_agent_receipts_are_signed_and_hash_chained(tmp_path):
    store = ReceiptStore(tmp_path, signing_key="trusted-key", signer_id="ci-evidence-worker")
    first_path = store.write(make_receipt("run-1", "corpus"))
    second_path = store.write(make_receipt("run-1", "memory"))
    first = json.loads(first_path.read_text())
    second = json.loads(second_path.read_text())

    assert first["previous_receipt_sha256"] == ""
    assert second["previous_receipt_sha256"] == first["receipt_sha256"]
    assert first["signer_id"] == "ci-evidence-worker"
    assert len(first["signature_hmac_sha256"]) == 64
    head = json.loads((tmp_path / "chain-head.json").read_text())
    assert head["receipt_sha256"] == second["receipt_sha256"]
    assert head["signed"] is True


def test_unsigned_relay_digest_is_not_authorship(tmp_path):
    text = "source material"
    forged = {
        "text": text,
        "url": "https://example.com/source",
        "intelligence": {"citations": ["https://example.com/source"], "confidence": 0.99},
        "receipt": {
            "issuer": "nexus-relay",
            "source_url": "https://example.com/source",
            "final_url": "https://example.com/source",
            "request_id": "request-1",
            "receipt_sha256": "a" * 64,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }
    with pytest.raises(ValueError, match="authorship"):
        from_relay_response(forged, signing_key="relay-key")


def test_signed_relay_receipt_is_admitted_only_with_matching_content_and_urls():
    text = "verified source material"
    receipt = {
        "issuer": "nexus-relay",
        "source_url": "https://example.com/source",
        "final_url": "https://example.com/source",
        "request_id": "request-1",
        "receipt_sha256": "b" * 64,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    receipt["signature_hmac_sha256"] = hmac.new(
        b"relay-key", canonical_json(receipt), hashlib.sha256
    ).hexdigest()
    response = {
        "text": text,
        "intelligence": {
            "citations": ["https://example.com/source"],
            "entities": ["AURO"],
            "confidence": 0.8,
        },
        "receipt": receipt,
    }
    evidence = from_relay_response(response, signing_key="relay-key")
    candidate = evidence.memory_candidate()
    assert candidate["authorship_verified"] is True
    assert candidate["content_hash_verified"] is True
    assert candidate["citation_urls_verified"] is True
    assert candidate["status"] == "verified_candidate"
