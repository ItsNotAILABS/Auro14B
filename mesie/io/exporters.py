"""Export spectral records to various formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mesie.core.records import MultiElementRecord


def _record_to_dict(record: MultiElementRecord) -> Dict[str, Any]:
    """Convert a record to a serializable dictionary."""
    components = []
    for c in record.components:
        comp_dict: Dict[str, Any] = {
            "name": c.name,
            "frequency": c.frequency.tolist(),
            "amplitude": c.amplitude.tolist(),
            "domain": c.domain,
            "units": c.units,
            "element_weight": c.element_weight,
        }
        if c.phase is not None:
            comp_dict["phase"] = c.phase.tolist()
        if c.node_id:
            comp_dict["node_id"] = c.node_id
        if c.metadata:
            comp_dict["metadata"] = c.metadata
        components.append(comp_dict)

    return {
        "record_id": record.record_id,
        "components": components,
        "representation": record.representation,
        "lineage": record.lineage,
    }


def export_record(record: MultiElementRecord, path: str, format: str = "json") -> None:
    """Export a spectral record to a file.

    Args:
        record: The record to export.
        path: Output file path.
        format: Output format ('json' or 'csv').

    Raises:
        ValueError: If format is not supported.
    """
    output_path = Path(path)

    if format == "json":
        data = _record_to_dict(record)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif format == "csv":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV export.")
        if not record.components:
            raise ValueError("Cannot export empty record to CSV.")
        data_dict = {"frequency": record.components[0].frequency}
        for c in record.components:
            data_dict[c.name] = c.amplitude
        df = pd.DataFrame(data_dict)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: '{format}'. Use 'json' or 'csv'.")
