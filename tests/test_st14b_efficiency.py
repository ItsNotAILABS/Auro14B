from __future__ import annotations

import unittest

from auro_native_llm.model.st14b import (
    ST14BArchitecture,
    build_st14b_config,
    st14b_quantization_matrix,
)


class ST14BEfficiencyTests(unittest.TestCase):
    def test_geometry_matches_research_contract(self) -> None:
        spec = ST14BArchitecture()
        spec.validate()
        self.assertEqual(spec.hidden_dim, 5120)
        self.assertEqual(spec.num_layers, 40)
        self.assertEqual(spec.num_heads, 40)
        self.assertEqual(spec.num_kv_heads, 5)
        self.assertEqual(spec.gqa_ratio, 8)
        self.assertEqual(spec.head_dim, 128)
        self.assertEqual(spec.vocab_size, 128000)
        self.assertEqual(spec.max_seq_len, 8192)

    def test_parameter_estimate_is_14b_class(self) -> None:
        estimate = ST14BArchitecture().dense_parameter_estimate()
        self.assertGreater(estimate, 14_000_000_000)
        self.assertLess(estimate, 14_500_000_000)
        self.assertEqual(estimate, 14_339_691_520)

    def test_gqa_reduces_theoretical_kv_cache_by_87_5_percent(self) -> None:
        spec = ST14BArchitecture()
        gqa = spec.kv_cache_bytes()
        mha = spec.mha_equivalent_kv_cache_bytes()
        self.assertEqual(gqa, 838_860_800)
        self.assertEqual(mha, 6_710_886_400)
        self.assertAlmostEqual(1.0 - gqa / mha, 0.875)

    def test_auro_profile_is_dense_serving_lane(self) -> None:
        config = build_st14b_config()
        self.assertEqual(config.model_id, "AURO-ST-14B")
        self.assertFalse(config.use_moe)
        self.assertFalse(config.use_cross_modal)
        self.assertFalse(config.use_spectral_encoder)
        self.assertEqual(config.num_kv_heads, 5)
        self.assertEqual(config.extra["status"], "PROTOTYPE")
        self.assertTrue(config.extra["benchmark_required"])
        self.assertTrue(config.extra["research_claims_are_targets_not_results"])

    def test_hardware_numbers_are_not_marked_as_results(self) -> None:
        boundary = ST14BArchitecture().report()["claim_boundary"]
        self.assertTrue(boundary["architecture_contract"])
        for name, value in boundary.items():
            if name != "architecture_contract":
                self.assertFalse(value, name)

    def test_quantization_requires_promotion_evidence(self) -> None:
        rows = st14b_quantization_matrix()
        by_precision = {row["precision"]: row for row in rows}
        self.assertEqual(set(by_precision), {"bf16", "fp8", "int8", "int4"})
        self.assertEqual(by_precision["bf16"]["status"], "REFERENCE")
        self.assertEqual(by_precision["fp8"]["status"], "PLANNED")
        self.assertIn("promotion_gate", by_precision["int4"])


if __name__ == "__main__":
    unittest.main()
