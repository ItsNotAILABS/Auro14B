"""Tests for spectral embeddings."""

import numpy as np
import pytest

from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.embeddings.encoders import SpectralFeatureEncoder
from mesie.embeddings.retrieval import SpectralRetriever


class TestSpectralVectorizer:
    def _make_payload(self, record_id="r1"):
        return {
            "record_id": record_id,
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0, 5.0], "amplitude": [0.2, 0.5, 0.8, 0.4, 0.1]}
            ],
        }

    def test_transform_shape(self):
        vectorizer = SpectralVectorizer()
        payload = self._make_payload()
        emb = vectorizer.transform(payload)
        assert emb.ndim == 1
        assert len(emb) == vectorizer.embedding_dim

    def test_transform_deterministic(self):
        vectorizer = SpectralVectorizer()
        payload = self._make_payload()
        emb1 = vectorizer.transform(payload)
        emb2 = vectorizer.transform(payload)
        np.testing.assert_array_equal(emb1, emb2)

    def test_fit_transform(self):
        vectorizer = SpectralVectorizer()
        payload = self._make_payload()
        emb = vectorizer.fit_transform(payload)
        assert len(emb) > 0

    def test_batch_transform(self):
        vectorizer = SpectralVectorizer()
        payloads = [self._make_payload(f"r{i}") for i in range(3)]
        matrix = vectorizer.batch_transform(payloads)
        assert matrix.shape == (3, vectorizer.embedding_dim)

    def test_different_records_different_embeddings(self):
        vectorizer = SpectralVectorizer()
        p1 = {"record_id": "r1", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.1, 0.2, 0.3]}]}
        p2 = {"record_id": "r2", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.9, 0.8, 0.7]}]}
        emb1 = vectorizer.transform(p1)
        emb2 = vectorizer.transform(p2)
        assert not np.allclose(emb1, emb2)


class TestFeatureEncoder:
    def test_encode_features(self):
        encoder = SpectralFeatureEncoder()
        payload = {
            "record_id": "r1",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0], "amplitude": [0.2, 0.5, 0.8, 0.3]}
            ],
        }
        features = encoder.encode(payload)
        assert "spectral_centroid" in features
        assert "peak_frequency" in features
        assert "total_energy" in features
        assert features["peak_frequency"] == 3.0

    def test_encode_batch(self):
        encoder = SpectralFeatureEncoder()
        payloads = [
            {"record_id": f"r{i}", "components": [{"name": "a", "frequency": [1.0, 2.0], "amplitude": [0.1 * i + 0.1, 0.2]}]}
            for i in range(3)
        ]
        results = encoder.encode_batch(payloads)
        assert len(results) == 3


class TestRetriever:
    def test_index_and_query(self):
        retriever = SpectralRetriever()
        records = [
            {"record_id": "r1", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.1, 0.2, 0.3]}]},
            {"record_id": "r2", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.9, 0.8, 0.7]}]},
        ]
        retriever.index(records)
        assert retriever.size == 2

        query = {"record_id": "q", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.1, 0.2, 0.3]}]}
        results = retriever.query(query, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "r1"  # Closest match
