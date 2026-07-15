"""Spectral feature encoders for embedding generation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record


class SpectralFeatureEncoder:
    """Encode spectral records into feature dictionaries.

    Provides a structured feature extraction interface that outputs
    named feature dictionaries suitable for downstream ML pipelines.
    """

    def encode(self, record: RecordInput) -> Dict[str, float]:
        """Encode a record into a feature dictionary.

        Args:
            record: Input spectral record.

        Returns:
            Dictionary of named feature values.
        """
        rec = load_record(record)
        features: Dict[str, float] = {}

        if not rec.components:
            return features

        comp = rec.components[0]
        amp = np.abs(comp.amplitude)
        freq = comp.frequency
        total = max(float(np.sum(amp)), 1e-12)

        features["mean_amplitude"] = float(np.mean(amp))
        features["std_amplitude"] = float(np.std(amp))
        features["max_amplitude"] = float(np.max(amp))
        features["spectral_centroid"] = float(np.sum(freq * amp) / total)
        features["spectral_spread"] = float(
            np.sqrt(np.sum(((freq - features["spectral_centroid"]) ** 2) * amp) / total)
        )
        features["peak_frequency"] = float(freq[np.argmax(amp)])
        features["n_components"] = float(len(rec.components))
        features["total_energy"] = float(np.sum(amp ** 2))

        # Spectral entropy
        p = amp / total
        p = p[p > 0]
        features["spectral_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))

        return features

    def encode_batch(self, records: List[RecordInput]) -> List[Dict[str, float]]:
        """Encode a batch of records.

        Args:
            records: List of input records.

        Returns:
            List of feature dictionaries.
        """
        return [self.encode(r) for r in records]
