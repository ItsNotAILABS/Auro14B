from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from benchmarks.portfolio_flows import (
    build_suite,
    capability_manifest_flow,
    st14b_runtime_flow,
    uber_chrome_contract,
)


class PortfolioFlowTests(unittest.TestCase):
    def test_st14b_runtime_flow(self) -> None:
        result = st14b_runtime_flow()
        self.assertTrue(result["ok"])
        self.assertEqual(result["kv_heads"], 1)
        self.assertEqual(result["cache_sequence_length"], 33)

    def test_uber_contract_never_books(self) -> None:
        result = uber_chrome_contract()
        self.assertTrue(result["ok"])
        self.assertFalse(result["external_booking_performed"])
        self.assertEqual(result["steps"][-1], "stop_before_booking")
        self.assertIn("book_without_approval", result["forbidden"])

    def test_capability_manifest_covers_core_utilities(self) -> None:
        result = capability_manifest_flow()
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["count"], 20)
        self.assertEqual(result["missing"], [])

    def test_suite_skips_unconfigured_live_gateways(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("AURO_BENCH_STATE")
            os.environ["AURO_BENCH_STATE"] = tmp
            try:
                results = build_suite(1)
            finally:
                if old is None:
                    os.environ.pop("AURO_BENCH_STATE", None)
                else:
                    os.environ["AURO_BENCH_STATE"] = old
        indexed = {item.name: item for item in results}
        self.assertEqual(indexed["iot.nova_gateway"].status, "skipped")
        self.assertEqual(indexed["browser.live_gateway"].status, "skipped")
        self.assertEqual(indexed["chrome.cdp_gateway"].status, "skipped")
        self.assertEqual(indexed["uber.chrome.governed_contract"].status, "passed")


if __name__ == "__main__":
    unittest.main()
