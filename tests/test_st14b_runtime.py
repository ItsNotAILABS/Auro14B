from __future__ import annotations

import unittest

import torch

from auro_native_llm.model.st14b_runtime import (
    AuroST14BForCausalLM,
    ST14BRuntimeConfig,
    kv_cache_bytes,
    mha_equivalent_kv_cache_bytes,
    parameter_count,
)


class ST14BRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ST14BRuntimeConfig(
            vocab_size=128,
            hidden_size=64,
            num_layers=2,
            num_heads=8,
            num_kv_heads=1,
            intermediate_size=176,
            max_seq_len=32,
        )
        self.model = AuroST14BForCausalLM(self.config).eval()

    def test_geometry_is_eight_to_one_gqa(self) -> None:
        self.assertEqual(self.config.kv_group_size, 8)
        self.assertEqual(self.config.head_dim, 8)

    def test_prefill_populates_native_kv_heads(self) -> None:
        cache = self.model.new_cache()
        tokens = torch.tensor([[1, 2, 3, 4]])
        logits = self.model.prefill(tokens, cache)
        self.assertEqual(tuple(logits.shape), (1, 4, 128))
        self.assertEqual(cache.sequence_length, 4)
        self.assertEqual(cache.layers[0].key.shape[1], 1)

    def test_decode_step_extends_cache_by_one(self) -> None:
        cache = self.model.new_cache()
        self.model.prefill(torch.tensor([[1, 2, 3]]), cache)
        logits = self.model.decode_step(torch.tensor([[4]]), cache)
        self.assertEqual(tuple(logits.shape), (1, 128))
        self.assertEqual(cache.sequence_length, 4)

    def test_cached_generation_runs(self) -> None:
        output = self.model.generate_cached(torch.tensor([[1, 2, 3]]), max_new_tokens=4)
        self.assertEqual(tuple(output.shape), (1, 7))

    def test_cache_math_matches_eighty_seven_point_five_percent_reduction(self) -> None:
        gqa = kv_cache_bytes(self.config, batch_size=1, sequence_length=32, bytes_per_element=2)
        mha = mha_equivalent_kv_cache_bytes(self.config, batch_size=1, sequence_length=32, bytes_per_element=2)
        self.assertEqual(gqa * 8, mha)
        self.assertAlmostEqual(1.0 - (gqa / mha), 0.875)

    def test_tied_embeddings_avoid_duplicate_output_matrix(self) -> None:
        self.assertIs(self.model.lm_head.weight, self.model.embedding.weight)
        self.assertGreater(parameter_count(self.model), 0)

    def test_decode_step_rejects_multiple_tokens(self) -> None:
        cache = self.model.new_cache()
        with self.assertRaises(ValueError):
            self.model.decode_step(torch.tensor([[1, 2]]), cache)


if __name__ == "__main__":
    unittest.main()
