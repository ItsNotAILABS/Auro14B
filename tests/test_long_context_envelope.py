import numpy as np

from auro_native_llm.context import ContextEnvelope


def test_accepts_294912_tokens_with_bounded_dense_view():
    tokens = np.arange(294_912, dtype=np.int64) % 32_000
    envelope = ContextEnvelope(accepted_limit=294_912, dense_window=32_768, chunk_size=4_096, retrieval_budget=8_192)
    dense, receipt, chunks = envelope.ingest(tokens)
    assert receipt.accepted_tokens == 294_912
    assert receipt.dense_tokens == 32_768
    assert receipt.retrieved_tokens <= 8_192
    assert receipt.truncated_input_tokens == 0
    assert len(dense) == 32_768
    assert len(chunks) > 1
    assert len(receipt.envelope_sha256) == 64


def test_context_selection_is_deterministic():
    tokens = np.tile(np.arange(4096, dtype=np.int64), 72)
    envelope = ContextEnvelope()
    dense_a, receipt_a, _ = envelope.ingest(tokens)
    dense_b, receipt_b, _ = envelope.ingest(tokens)
    assert np.array_equal(dense_a, dense_b)
    assert receipt_a == receipt_b


def test_input_above_limit_reports_truncation():
    tokens = np.arange(300_000, dtype=np.int64)
    dense, receipt, _ = ContextEnvelope().ingest(tokens)
    assert receipt.accepted_tokens == 294_912
    assert receipt.truncated_input_tokens == 5_088
    assert dense[-1] == tokens[-1]
