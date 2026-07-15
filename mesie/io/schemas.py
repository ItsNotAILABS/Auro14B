"""Schema definitions for spectral record validation."""

from __future__ import annotations

from typing import Any, Dict, List

# JSON schema for spectral record input validation
RECORD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["record_id", "components"],
    "properties": {
        "record_id": {"type": "string"},
        "representation": {"type": "string", "enum": ["single", "multi", "psd", "fas", "rotdnn"]},
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "frequency", "amplitude"],
                "properties": {
                    "name": {"type": "string"},
                    "frequency": {"type": "array", "items": {"type": "number"}},
                    "amplitude": {"type": "array", "items": {"type": "number"}},
                    "phase": {"type": "array", "items": {"type": "number"}},
                    "domain": {"type": "string"},
                    "units": {"type": "string"},
                    "element_weight": {"type": "number"},
                    "node_id": {"type": "string"},
                },
            },
        },
        "lineage": {"type": "array", "items": {"type": "string"}},
    },
}


def get_supported_formats() -> List[str]:
    """Return list of supported file formats."""
    return ["json", "csv"]
