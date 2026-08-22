from __future__ import annotations

import json

import pytest

from auro_native_llm.production_fleet.receipts import ReceiptLedger


def test_receipt_ledger_round_trip_and_chain(tmp_path):
    path = tmp_path / "receipts.jsonl"
    ledger = ReceiptLedger(path)
    first = ledger.record("capability", "alpha", True, {"value": 1})
    second = ledger.record("model_response", "beta", True, {"value": 2})

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.previous_hash == first.receipt_hash
    assert ledger.verify() == {"valid": True, "count": 2, "head": second.receipt_hash}
    assert path.read_bytes().endswith(b"\n")

    restored = ReceiptLedger(path)
    assert restored.verify()["valid"] is True
    assert restored.tail(2)[-1]["receipt_hash"] == second.receipt_hash


def test_receipt_ledger_rejects_incomplete_trailing_record(tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_bytes(b'{"sequence":1')
    with pytest.raises(ValueError, match="incomplete trailing record"):
        ReceiptLedger(path)


def test_receipt_ledger_rejects_tampered_chain(tmp_path):
    path = tmp_path / "receipts.jsonl"
    ledger = ReceiptLedger(path)
    ledger.record("capability", "alpha", True, {"value": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["subject"] = "tampered"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid receipt ledger"):
        ReceiptLedger(path)
