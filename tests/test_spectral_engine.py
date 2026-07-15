import unittest
import numpy as np

from spectral_engine import (
    GenerationConfig,
    SpectralMatcher,
    generate_psd,
    load_record,
    validate_record,
)


class TestSpectralEngine(unittest.TestCase):
    def test_load_record_dict(self):
        payload = {
            "record_id": "r1",
            "components": [
                {
                    "name": "a",
                    "frequency": [1.0, 2.0, 3.0],
                    "amplitude": [0.2, 0.3, 0.4],
                }
            ],
        }
        rec = load_record(payload)
        self.assertEqual(rec.record_id, "r1")
        self.assertEqual(len(rec.components), 1)

    def test_validate_non_monotonic(self):
        payload = {
            "record_id": "bad",
            "components": [
                {
                    "name": "a",
                    "frequency": [1.0, 3.0, 2.0],
                    "amplitude": [0.2, 0.3, 0.4],
                }
            ],
        }
        report = validate_record(payload)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("non-monotonically increasing frequency values" in e for e in report.errors))

    def test_match_identical_records_high_score(self):
        freq = np.array([1.0, 2.0, 3.0, 4.0])
        amp = np.array([0.4, 0.8, 0.6, 0.2])
        payload = {
            "record_id": "r1",
            "components": [
                {"name": "a", "frequency": freq.tolist(), "amplitude": amp.tolist()}
            ],
        }
        matcher = SpectralMatcher()
        result = matcher.score(payload, payload)
        self.assertGreater(result.score, 0.99)

    def test_generate_psd_positive(self):
        rec = generate_psd(GenerationConfig(seed=3))
        self.assertEqual(rec.representation, "psd")
        self.assertEqual(rec.components[0].units, "psd")
        self.assertTrue(np.all(rec.components[0].amplitude >= 0.0))


if __name__ == "__main__":
    unittest.main()
