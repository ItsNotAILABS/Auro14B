"""Embed your spectral JSON into library/my_spectral_index.json and optional Octopus load.

Usage:
  python scripts/embed_my_library.py path/to/your/spectra_folder
  python scripts/embed_my_library.py sample.json --octopus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesie.library.user_corpus import embed_paths
from mesie.octopus import OctopusConfig, OctopusController


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed user spectral JSON library")
    parser.add_argument("paths", nargs="+", help="JSON files or folders")
    parser.add_argument(
        "--out",
        default=str(ROOT / "library" / "my_spectral_index.json"),
        help="Output index path",
    )
    parser.add_argument(
        "--octopus",
        action="store_true",
        help="Run Octopus standard cycle using the new index",
    )
    args = parser.parse_args()

    corpus = embed_paths(args.paths, save_to=args.out)
    print(f"Embedded {corpus.count} files -> {args.out}")

    if args.octopus and corpus.count > 0:
        first = corpus.record_paths()[0]
        from mesie import load_record

        rec = load_record(first)
        octopus = OctopusController(
            config=OctopusConfig(user_index_path=args.out, movement_steps=2),
        )
        report = octopus.run_standard_cycle(rec)
        print(report.plain_summary)
        if report.user_library.get("nearest_in_user_library"):
            print("Nearest in your library:", report.user_library["nearest_in_user_library"])


if __name__ == "__main__":
    main()