"""Octopus engineering demo — eight arms, internal API, cross-engine workflow."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.octopus import OctopusController, OctopusConfig


def main() -> None:
    freq = np.linspace(0.2, 25.0, 64)
    amp = 0.5 + np.exp(-((freq - 5.0) ** 2) / 8.0)
    record = MultiElementRecord(
        record_id="demo-octopus-001",
        components=[SpectralComponent(name="ch", frequency=freq, amplitude=amp)],
    )
    candidate = MultiElementRecord(
        record_id="demo-octopus-002",
        components=[SpectralComponent(name="ch", frequency=freq, amplitude=amp * 0.9)],
    )

    octopus = OctopusController(config=OctopusConfig(movement_steps=4))
    print("Engines on bus:", ", ".join(octopus.list_engines()))
    report = octopus.run_standard_cycle(record, candidate=candidate)
    print("\n--- Octopus run ---")
    print(report.plain_summary)
    print("\nArms used:", report.arms_used)
    print("Workflow embedding present:", bool(report.workflow.get("run", {}).get("data", {}).get("workflow_embedding")))


if __name__ == "__main__":
    main()