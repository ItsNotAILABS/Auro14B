"""Example 10: 50-day orbital-edge analysis (earthquake coupling + forward forecast)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
script = ROOT / "scripts" / "orbital_edge_50d_analysis.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, str(script)], check=True)