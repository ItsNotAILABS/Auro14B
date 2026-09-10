"""Cold storage for MoE expert weights: stale-by-default, hot-on-demand.

Expert weights live as one ``.npz`` file per (layer, expert) under a root
directory. Nothing is resident until the pager asks for it; loads are
``mmap``'d so the OS pages tensors in on first touch, and every byte moved is
counted. A ``manifest.json`` carries per-expert shapes, dtypes and sha256
digests so a run can prove *which* weights it executed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

WEIGHT_KEYS = ("gate_proj", "up_proj", "down_proj")


def _sha256_arrays(weights: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in WEIGHT_KEYS:
        arr = np.ascontiguousarray(weights[key])
        h.update(key.encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


class ExpertColdStore:
    """On-disk expert weights with integrity manifest and byte accounting."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bytes_read = 0
        self.loads = 0

    def _path(self, layer: int, expert: int) -> Path:
        return self.root / f"layer_{layer:02d}" / f"expert_{expert:02d}.npz"

    def save_expert(
        self,
        layer: int,
        expert_id: int,
        weights: Mapping[str, np.ndarray],
        meta: Dict[str, Any] | None = None,
    ) -> Path:
        """Persist one expert's weights. Returns the file path."""
        missing = [k for k in WEIGHT_KEYS if k not in weights]
        if missing:
            raise ValueError(f"expert weights missing keys: {missing}")
        path = self._path(layer, expert_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.ascontiguousarray(weights[k]) for k in WEIGHT_KEYS}
        np.savez(path, **arrays)
        manifest = self._read_manifest()
        manifest[f"{layer}:{expert_id}"] = {
            "path": str(path.relative_to(self.root)),
            "shapes": {k: list(arrays[k].shape) for k in WEIGHT_KEYS},
            "dtype": str(arrays[WEIGHT_KEYS[0]].dtype),
            "nbytes": int(sum(a.nbytes for a in arrays.values())),
            "sha256": _sha256_arrays(arrays),
            "meta": meta or {},
        }
        self._write_manifest(manifest)
        return path

    def save_moe_layer(self, layer: int, moe: Any) -> int:
        """Persist every expert of a ``MixtureOfExperts`` layer. Returns count."""
        n = 0
        for expert in moe.experts:
            self.save_expert(
                layer,
                int(expert.expert_id),
                {
                    "gate_proj": expert.gate_proj,
                    "up_proj": expert.up_proj,
                    "down_proj": expert.down_proj,
                },
                meta={"specialization": expert.specialization},
            )
            n += 1
        return n

    def load(self, layer: int, expert: int, mmap: bool = True) -> Dict[str, np.ndarray]:
        """Load one expert's weights (mmap'd by default). Counts bytes moved."""
        path = self._path(layer, expert)
        if not path.exists():
            raise FileNotFoundError(f"expert not in cold store: layer={layer} expert={expert}")
        mode = "r" if mmap else None
        with np.load(path, mmap_mode=mode, allow_pickle=False) as zf:
            arrays = {k: zf[k] for k in WEIGHT_KEYS}
        entry = self._read_manifest().get(f"{layer}:{expert}", {})
        self.bytes_read += int(entry.get("nbytes", sum(a.nbytes for a in arrays.values())))
        self.loads += 1
        return arrays

    def expert_nbytes(self, layer: int, expert: int) -> int:
        entry = self._read_manifest().get(f"{layer}:{expert}", {})
        return int(entry.get("nbytes", 0))

    def verify(self, layer: int, expert: int) -> bool:
        """Re-hash an expert's weights against the manifest digest."""
        entry = self._read_manifest().get(f"{layer}:{expert}")
        if not entry:
            return False
        arrays = self.load(layer, expert, mmap=False)
        # don't double-count a verification read as a demand load
        self.bytes_read -= int(entry.get("nbytes", 0))
        self.loads -= 1
        return _sha256_arrays(arrays) == entry["sha256"]

    def _read_manifest(self) -> Dict[str, Any]:
        mp = self.root / "manifest.json"
        if not mp.exists():
            return {}
        return json.loads(mp.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
