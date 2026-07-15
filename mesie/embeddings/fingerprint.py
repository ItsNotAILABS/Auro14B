"""End-to-end fingerprint pipeline: TF → salient → embed/hash → ANN."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.embeddings.ann import ANNHit, ANNIndex
from mesie.embeddings.lsh import LSHSignature
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.io.loaders import RecordInput, load_record
from mesie.signal.salient import SalientFeatureExtractor, SalientFeatureSet
from mesie.signal.time_frequency import TimeFrequencyMap, TimeFrequencyTransform


@dataclass
class FingerprintResult:
    record_id: str
    tf_method: str
    tf_shape: List[int]
    n_salient_points: int
    salient_vector: List[float]
    dense_embedding: List[float]
    combined_vector: List[float]
    lsh_hex: str
    lsh_bucket: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FingerprintStore:
    """In-memory vector DB with ANN + optional disk persistence."""

    index: ANNIndex = field(default_factory=lambda: ANNIndex(use_lsh=True))
    fingerprints: Dict[str, FingerprintResult] = field(default_factory=dict)

    def save_json(self, path: Path) -> None:
        payload = {
            "count": len(self.fingerprints),
            "metric": self.index.metric,
            "entries": [fp.to_dict() for fp in self.fingerprints.values()],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SpectralFingerprintPipeline:
    """TF transform → salient features → embedding + LSH → ANN lookup."""

    def __init__(
        self,
        *,
        use_synthetic_stft: bool = False,
        max_salient: int = 32,
        lsh_planes: int = 16,
    ) -> None:
        self.tf = TimeFrequencyTransform()
        self.salient = SalientFeatureExtractor(max_points=max_salient)
        self.vectorizer = SpectralVectorizer(n_bands=8)
        self.use_synthetic_stft = use_synthetic_stft
        self.store = FingerprintStore(
            index=ANNIndex(use_lsh=True, lsh_planes=lsh_planes, metric="cosine"),
        )

    def _build_vector(self, record: RecordInput) -> tuple:
        rec = load_record(record)
        if self.use_synthetic_stft:
            tf_map = self.tf.synthetic_signal_from_record(rec)
        else:
            tf_map = self.tf.from_record(rec)
        salient_set = self.salient.extract(tf_map)
        dense = self.vectorizer.transform(rec)
        combined = np.concatenate([salient_set.feature_vector, dense])
        combined = combined / max(np.linalg.norm(combined), 1e-12)
        return rec, tf_map, salient_set, dense, combined

    def process(self, record: RecordInput) -> FingerprintResult:
        rec, tf_map, salient_set, dense, combined = self._build_vector(record)
        sig = self.store.index.add(rec.record_id, combined)
        fp = FingerprintResult(
            record_id=rec.record_id,
            tf_method=tf_map.method,
            tf_shape=list(tf_map.shape),
            n_salient_points=salient_set.n_points,
            salient_vector=salient_set.feature_vector.tolist(),
            dense_embedding=dense.tolist(),
            combined_vector=combined.tolist(),
            lsh_hex=sig.to_hex() if sig else "",
            lsh_bucket=sig.bucket_key if sig else "",
        )
        self.store.fingerprints[rec.record_id] = fp
        return fp

    def index_records(self, records: Sequence[RecordInput]) -> int:
        for r in records:
            self.process(r)
        return self.store.index.size

    def query(
        self,
        record: RecordInput,
        top_k: int = 5,
        *,
        index_query: bool = False,
    ) -> List[ANNHit]:
        rec, _, _, _, combined = self._build_vector(record)
        if index_query:
            self.store.index.add(rec.record_id, combined)
        return self.store.index.query(combined, top_k=top_k)

    def query_by_id(self, record_id: str, top_k: int = 5) -> List[ANNHit]:
        fp = self.store.fingerprints.get(record_id)
        if fp is None:
            return []
        q = np.array(fp.combined_vector, dtype=np.float64)
        return self.store.index.query(q, top_k=top_k, probe_exact=False)

    def explain_match(self, query_id: str, hit_id: str) -> Dict[str, Any]:
        """Compare salient landmarks between two indexed fingerprints."""
        a = self.store.fingerprints.get(query_id)
        b = self.store.fingerprints.get(hit_id)
        if not a or not b:
            return {"error": "missing fingerprint"}
        va = np.array(a.salient_vector)
        vb = np.array(b.salient_vector)
        sim = float(np.dot(va, vb) / (max(np.linalg.norm(va), 1e-12) * max(np.linalg.norm(vb), 1e-12)))
        return {
            "query": query_id,
            "hit": hit_id,
            "salient_cosine": round(sim, 4),
            "lsh_same_bucket": a.lsh_bucket == b.lsh_bucket,
            "query_salient_points": a.n_salient_points,
            "hit_salient_points": b.n_salient_points,
        }