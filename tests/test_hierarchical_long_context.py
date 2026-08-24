import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.long_context import AuroLongContextModel


def test_hierarchical_long_context_retrieves_and_primes_working_memory():
    model = AuroLanguageModel.build("Auro-2B", mode="dev")
    long_model = AuroLongContextModel(model, dense_window=256)

    vocab = max(32, model.config.vocab_size)
    # Repeated thematic islands plus a recent query-like tail create enough
    # structure for embedding-space hierarchy to rank historical regions.
    history = np.concatenate(
        [
            np.arange(0, 1200, dtype=np.int64) % vocab,
            np.full(300, 17, dtype=np.int64),
            np.arange(300, 1300, dtype=np.int64) % vocab,
            np.full(300, 17, dtype=np.int64),
        ]
    )
    recent = np.concatenate(
        [np.arange(200, 400, dtype=np.int64) % vocab, np.full(80, 17, dtype=np.int64)]
    )
    tokens = np.concatenate([history, recent])

    dense, receipt, chunks = long_model.prepare_context(tokens)
    assert dense.size <= long_model.envelope.dense_window
    assert receipt.accepted_tokens == tokens.size
    assert receipt.retrieved_tokens > 0
    assert receipt.macro_candidates > 0
    assert receipt.meso_candidates > 0
    assert receipt.micro_candidates > 0
    assert receipt.selected_micro_ids
    assert any(chunk.level == "macro" for chunk in chunks)
    assert any(chunk.level == "meso" for chunk in chunks)
    assert any(chunk.level == "micro" for chunk in chunks)

    memory = model.delta_attention.working_memory
    assert memory.tokens_seen == 0
    assert long_model.hierarchy.prime_working_memory() is True
    assert memory.tokens_seen == 1
    assert memory.compute_pressure > 0.0
    assert long_model.hierarchy.last_receipt.working_memory_primed is True


def test_short_context_does_not_require_hierarchical_retrieval():
    model = AuroLanguageModel.build("Auro-2B", mode="dev")
    long_model = AuroLongContextModel(model, dense_window=256)
    tokens = np.arange(64, dtype=np.int64)
    dense, receipt, chunks = long_model.prepare_context(tokens)
    np.testing.assert_array_equal(dense, tokens)
    assert receipt.retrieved_tokens == 0
    assert receipt.selected_micro_ids == []
    assert chunks == []
