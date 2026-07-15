"""TF → salient → LSH hash → ANN matching demo."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie.embeddings import SpectralFingerprintPipeline


def main() -> None:
    pipe = SpectralFingerprintPipeline()
    refs = [load_reference_record(n) for n in list_references()]
    pipe.index_records(refs)

    query = refs[0]
    fp = pipe.store.fingerprints[query.record_id]
    hits = pipe.query(query, top_k=3)

    print("=== MESIE Fingerprint / ANN Pipeline ===\n")
    print(f"Query: {query.record_id}")
    print(f"TF: {fp.tf_method} shape={fp.tf_shape}")
    print(f"Salient points: {fp.n_salient_points}")
    print(f"LSH bucket: {fp.lsh_bucket}")
    print(f"Combined vector dim: {len(fp.combined_vector)}\n")
    print("ANN hits (LSH pre-filter + cosine rerank):")
    for h in hits:
        ex = pipe.explain_match(query.record_id, h.item_id)
        print(
            f"  {h.item_id}: sim={h.similarity:.4f} dist={h.distance:.4f} "
            f"same_bucket={ex.get('lsh_same_bucket')}"
        )


if __name__ == "__main__":
    main()