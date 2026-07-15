#!/usr/bin/env python3
"""MESIE Spectral DRACO diagnostic benchmark runner.

This script runs the complete DRACO-equivalent diagnostic benchmark on MESIE architecture.
Results are saved to deliverables/ directory for analysis and reporting.

Usage:
    python scripts/mesie_spectral_draco_benchmark.py [--architecture NAME] [--output PATH]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def main():
    """Run MESIE Spectral DRACO diagnostic benchmark."""
    parser = argparse.ArgumentParser(
        description="Run MESIE Spectral DRACO diagnostic benchmark"
    )
    parser.add_argument(
        "--architecture",
        default="MESIE-v0.4.0",
        help="Architecture name to benchmark (default: MESIE-v0.4.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for results JSON (default: deliverables/mesie_spectral_draco_[timestamp].json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    try:
        from mesie.foundation.evaluation.draco_diagnostic import MESIESpectralDRACO

        # Create output directory if needed
        if args.output is None:
            output_dir = Path("deliverables")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = str(output_dir / f"mesie_spectral_draco_{timestamp}.json")

        verbose = args.verbose and not args.quiet

        # Run benchmark
        if verbose:
            print(f"\n{'='*70}")
            print("MESIE Spectral DRACO Diagnostic Benchmark")
            print(f"{'='*70}")
            print(f"Architecture: {args.architecture}")
            print(f"Output: {args.output}")
            print(f"{'='*70}\n")

        benchmark = MESIESpectralDRACO(architecture_name=args.architecture)
        report = benchmark.run_full_diagnostic(verbose=verbose)

        # Export results
        benchmark.export_results_json(report, args.output)

        if verbose:
            print(f"\n{'='*70}")
            print("✓ Benchmark completed successfully")
            print(f"Results saved to: {args.output}")
            print(f"{'='*70}\n")

        return 0

    except ImportError as e:
        print(f"Error: Could not import MESIE module: {e}", file=sys.stderr)
        print(
            "Please ensure MESIE is installed: pip install -e \".[dev,full]\"",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
