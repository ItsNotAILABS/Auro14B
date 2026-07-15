"""Rank spectral candidates — enterprise retrieval smoke demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie.matching.ranking import rank_candidates


def main() -> None:
    names = list_references()
    query = load_reference_record(names[0])
    pool = [load_reference_record(n) for n in names[1:6]]
    ranked = rank_candidates(query, pool, top_k=3)
    print(f"Query: {query.record_id}")
    for i, hit in enumerate(ranked, 1):
        rid = hit.candidate_id
        print(f"  {i}. {rid}  score={hit.composite_score:.4f}")


if __name__ == "__main__":
    main()