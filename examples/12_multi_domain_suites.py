"""Run terrain, robotics, orbital, power, and seismic MESIE analysis suites."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesie.analysis import run_all_suites

if __name__ == "__main__":
    data = run_all_suites()
    for line in data["executive_summary"]:
        print(line)
        print()