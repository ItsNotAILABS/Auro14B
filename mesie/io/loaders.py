"""Spectral record loading from various sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent, SpectralMetadata

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

ArrayLike = Union[np.ndarray, Sequence[float]]
RecordInput = Union[MultiElementRecord, np.ndarray, Mapping[str, Any], str, Path]


def _as_float_array(values: ArrayLike, field_name: str) -> np.ndarray:
    """Convert input to a 1D float numpy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional; got shape {arr.shape}.")
    return arr


def _load_from_dict(payload: Mapping[str, Any], record_id: Optional[str] = None) -> MultiElementRecord:
    """Load a record from a dictionary payload."""
    rid = str(payload.get("record_id") or record_id or "record")
    representation = str(payload.get("representation", "single"))

    components: List[SpectralComponent] = []
    if "components" in payload:
        for idx, comp in enumerate(payload["components"]):
            components.append(
                SpectralComponent(
                    name=str(comp.get("name", f"component_{idx}")),
                    frequency=_as_float_array(comp["frequency"], "frequency"),
                    amplitude=_as_float_array(comp["amplitude"], "amplitude"),
                    phase=_as_float_array(comp["phase"], "phase") if comp.get("phase") is not None else None,
                    domain=str(comp.get("domain", "frequency")),
                    units=str(comp.get("units", "linear")),
                    element_weight=float(comp.get("element_weight", 1.0)),
                    node_id=comp.get("node_id"),
                    metadata=dict(comp.get("metadata", {})),
                )
            )
    elif "frequency" in payload and "amplitude" in payload:
        components.append(
            SpectralComponent(
                name=str(payload.get("name", "component_0")),
                frequency=_as_float_array(payload["frequency"], "frequency"),
                amplitude=_as_float_array(payload["amplitude"], "amplitude"),
                phase=_as_float_array(payload["phase"], "phase") if payload.get("phase") is not None else None,
                units=str(payload.get("units", "linear")),
                node_id=payload.get("node_id"),
            )
        )
    else:
        raise ValueError("Dictionary input must provide either 'components' or 'frequency'/'amplitude'.")

    metadata = SpectralMetadata(
        source=str(payload.get("source", "")),
        units=str(payload.get("units", "linear")),
        domain=str(payload.get("domain", "frequency")),
    )

    return MultiElementRecord(
        record_id=rid,
        components=components,
        metadata=metadata,
        lineage=list(payload.get("lineage", [])),
        representation=representation,
        electro_metadata=dict(payload.get("electro_metadata", {})),
        node_tags=dict(payload.get("node_tags", {})),
    )


def load_record(source: RecordInput, record_id: Optional[str] = None) -> MultiElementRecord:
    """Load a spectral record from various input formats.

    Supports:
        - MultiElementRecord (passthrough)
        - Dictionary/Mapping with 'components' or 'frequency'/'amplitude'
        - NumPy array (1D amplitude or 2D frequency+amplitude)
        - File path (CSV or JSON)
        - pandas DataFrame (if pandas available)

    Args:
        source: Input data in any supported format.
        record_id: Optional record identifier override.

    Returns:
        A validated MultiElementRecord instance.

    Raises:
        TypeError: If input type is not supported.
        ValueError: If input data is malformed.
        FileNotFoundError: If file path does not exist.
    """
    if isinstance(source, MultiElementRecord):
        return source

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            if not HAS_PANDAS:
                raise ImportError("pandas is required to load CSV files. Install with: pip install pandas")
            source = pd.read_csv(path)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                return _load_from_dict(json.load(f), record_id=record_id)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or JSON.")

    if HAS_PANDAS and isinstance(source, pd.DataFrame):
        lower_cols = {c.lower(): c for c in source.columns}
        if "frequency" not in lower_cols:
            raise ValueError("DataFrame input must contain a 'frequency' column.")
        fcol = lower_cols["frequency"]
        acols = [c for c in source.columns if c != fcol]
        if not acols:
            raise ValueError("DataFrame must contain at least one amplitude column.")
        components = [
            SpectralComponent(
                name=str(c),
                frequency=_as_float_array(source[fcol].to_numpy(), "frequency"),
                amplitude=_as_float_array(source[c].to_numpy(), f"amplitude[{c}]"),
            )
            for c in acols
        ]
        representation = "single" if len(components) == 1 else "multi"
        return MultiElementRecord(
            record_id=record_id or "record",
            components=components,
            representation=representation,
        )

    if isinstance(source, Mapping):
        return _load_from_dict(source, record_id=record_id)

    if isinstance(source, np.ndarray):
        arr = np.asarray(source, dtype=float)
        if arr.ndim == 1:
            freq = np.arange(arr.shape[0], dtype=float)
            comp = SpectralComponent(name="component_0", frequency=freq, amplitude=arr)
            return MultiElementRecord(record_id=record_id or "record", components=[comp], representation="single")
        if arr.ndim == 2 and arr.shape[1] >= 2:
            comp = SpectralComponent(name="component_0", frequency=arr[:, 0], amplitude=arr[:, 1])
            return MultiElementRecord(record_id=record_id or "record", components=[comp], representation="single")
        raise ValueError("NumPy input must be 1D amplitude or 2D array with frequency and amplitude columns.")

    raise TypeError(f"Unsupported input type: {type(source)!r}")
