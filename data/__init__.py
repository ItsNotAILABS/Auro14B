"""MESIE reference data loader.

Provides convenience functions for loading bundled reference datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent


def get_reference_path(name: str) -> Path:
    """Get the full path to a reference data file."""
    if not name.endswith(".json"):
        name = f"{name}.json"

    ref_path = DATA_DIR / "reference" / name
    if ref_path.exists():
        return ref_path

    bench_path = DATA_DIR / "benchmarks" / name
    if bench_path.exists():
        return bench_path

    raise FileNotFoundError(f"Data file not found: {name}")


def get_library_path(name: str) -> Path:
    """Get the full path to a spectral library data file."""
    if not name.endswith(".json"):
        name = f"{name}.json"

    lib_path = DATA_DIR / "spectral_library" / name
    if lib_path.exists():
        return lib_path

    raise FileNotFoundError(f"Spectral library file not found: {name}")


def _normalize_component(comp: dict[str, Any]) -> dict[str, Any]:
    """Map bundled reference JSON keys to mesie.io loader keys."""
    out = dict(comp)
    if "frequency" not in out and "frequencies" in out:
        out["frequency"] = out["frequencies"]
    if "amplitude" not in out and "amplitudes" in out:
        out["amplitude"] = out["amplitudes"]
    if "name" not in out:
        out["name"] = out.get("component_id", out.get("direction", "component"))
    units = out.get("units")
    if not units and "metadata" in out:
        units = out["metadata"].get("units", {}).get("amplitude")
    if isinstance(units, str):
        out["units"] = units
    elif isinstance(comp.get("metadata"), dict) and "units" in comp["metadata"]:
        out["units"] = comp["metadata"]["units"]
    return out


def _normalize_reference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize bundled reference/benchmark JSON to load_record schema."""
    out = dict(payload)
    record_type = str(out.get("record_type", "")).lower()
    if record_type == "psd":
        out["representation"] = out.get("representation", "psd")
    elif record_type == "fas":
        out["representation"] = out.get("representation", "fas")
    elif record_type.startswith("rotd"):
        out["representation"] = out.get("representation", "rotdnn")

    meta = out.get("metadata") or {}
    units = meta.get("units", {})
    if isinstance(units, dict):
        out.setdefault("units", units.get("amplitude", "linear"))

    if "components" in out:
        comps = [_normalize_component(c) for c in out["components"]]
        default_unit = out.get("units", "linear")
        for c in comps:
            if "units" not in c:
                c["units"] = default_unit
        out["components"] = comps
    return out


def load_reference(name: str) -> dict[str, Any]:
    """Load a reference dataset by name (raw JSON)."""
    path = get_reference_path(name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_reference_record(name: str):
    """Load a reference dataset as a MultiElementRecord."""
    from mesie.io.loaders import load_record

    return load_record(_normalize_reference_payload(load_reference(name)))


def load_benchmark(name: str) -> dict[str, Any]:
    """Load a benchmark dataset by name."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    path = DATA_DIR / "benchmarks" / name
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {name}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_references() -> list[str]:
    """List all available reference datasets."""
    ref_dir = DATA_DIR / "reference"
    if not ref_dir.exists():
        return []
    return [f.stem for f in ref_dir.glob("*.json")]


def load_reference_record(name: str) -> "MultiElementRecord":
    """Load a reference dataset as a MultiElementRecord.

    Args:
        name: Dataset name (e.g., 'earthquake_psd_reference').

    Returns:
        MultiElementRecord instance.
    """
    from mesie.core.records import MultiElementRecord, SpectralComponent

    data = load_reference(name)
    components = []
    for comp_data in data.get("components", []):
        import numpy as np
        components.append(SpectralComponent(
            name=comp_data.get("component_id", comp_data.get("direction", "default")),
            frequency=np.array(comp_data.get("frequencies", []), dtype=float),
            amplitude=np.array(comp_data.get("amplitudes", []), dtype=float),
        ))

    return MultiElementRecord(
        record_id=data.get("record_id", name),
        components=components,
    )


def list_benchmarks() -> list[str]:
    """List all available benchmark datasets."""
    bench_dir = DATA_DIR / "benchmarks"
    if not bench_dir.exists():
        return []
    return [f.stem for f in bench_dir.glob("*.json")]


def load_library(name: str) -> dict[str, Any]:
    """Load a spectral library dataset by name."""
    path = get_library_path(name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_library() -> list[str]:
    """List all available spectral library datasets."""
    lib_dir = DATA_DIR / "spectral_library"
    if not lib_dir.exists():
        return []
    return [f.stem for f in lib_dir.glob("*.json")]