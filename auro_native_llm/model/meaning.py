"""Meaning engines — Latin, Sanskrit, and Nahuatl cosmic lexica.

First-class embedding tables that inject *named meaning* into the Auro token
stream. Reuses MESIE cosmology (22 layers / Teotl) and classical root glosses.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from auro_native_llm.model.phi_math import GOLDEN_ANGLE_RAD, PHI, PHI_INV, phi_init

LATIN_ROOTS: Dict[str, str] = {
    "ratio": "reason measure reckoning",
    "lumen": "light clarity illumination",
    "ordo": "order structure rank",
    "mens": "mind measure intention",
    "spectre": "appearance image spectrum",
    "veritas": "truth verification",
    "potentia": "power potential capacity",
    "forma": "form shape pattern",
    "nexus": "bond connection link",
    "memoria": "memory retention",
    "signum": "sign token signal",
    "numerus": "number count measure",
    "fluxus": "flow stream current",
    "resonare": "resound resonate echo",
    "computare": "compute reckon sum",
    "scientia": "knowledge science",
    "ars": "art skill craft",
    "lex": "law rule governance",
    "spiritus": "breath spirit life",
    "corpus": "body collection corpus",
}

SANSKRIT_ROOTS: Dict[str, str] = {
    "rta": "cosmic order truth rhythm",
    "satya": "truth reality essence",
    "akasha": "ether space sky field",
    "prana": "life breath energy",
    "manas": "mind thought sense",
    "buddhi": "intellect discernment",
    "citta": "consciousness store mind",
    "shakti": "power force energy",
    "nada": "sound vibration tone",
    "bindu": "point seed singularity",
    "mandala": "circle sacred geometry",
    "yantra": "instrument diagram engine",
    "sutra": "thread formula rule",
    "veda": "knowledge sacred lore",
    "dharma": "law duty structure",
    "karma": "action consequence work",
    "yoga": "union discipline join",
    "om": "primordial sound whole",
    "jyoti": "light radiance flame",
    "kala": "time art black continuum",
}


def _nahuatl_layers() -> Dict[str, str]:
    try:
        from mesie.cosmology.layers import _HEAVEN_NAMES, _UNDERWORLD_NAMES

        out: Dict[str, str] = {}
        for i, name in enumerate(_UNDERWORLD_NAMES):
            key = name.lower().replace(" ", "_")
            out[key] = f"underworld layer {i + 1} low-frequency stratum {name}"
        for i, name in enumerate(_HEAVEN_NAMES):
            key = name.lower().replace(" ", "_")
            out[key] = f"heaven layer {i + 1} high-frequency stratum {name}"
        out["teotl"] = "divine energy continuous transformation"
        out["omeyocan"] = "place of duality highest heaven"
        return out
    except Exception:
        return {
            "teotl": "divine energy continuous transformation",
            "mictlan": "underworld low-frequency domain",
            "ilhuicatl": "heaven high-frequency domain",
            "omeyocan": "place of duality highest heaven",
        }


NAHUATL_ROOTS = _nahuatl_layers()


def _stable_vec(text: str, dim: int, salt: str = "") -> np.ndarray:
    h = hashlib.sha256((salt + "|" + text).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float64)
    idx = np.arange(dim, dtype=np.float64)
    v = v + 0.15 * np.sin(idx * GOLDEN_ANGLE_RAD + len(text) * PHI)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


@dataclass
class MeaningHit:
    root: str
    engine: str
    gloss: str
    score: float
    vector: np.ndarray


@dataclass
class MeaningEngine:
    name: str
    roots: Dict[str, str]
    dim: int
    table: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.table:
            for root, gloss in self.roots.items():
                self.table[root] = _stable_vec(f"{root}:{gloss}", self.dim, salt=self.name)

    def match(self, text: str, top_k: int = 3) -> List[MeaningHit]:
        t = text.lower()
        hits: List[MeaningHit] = []
        for root, gloss in self.roots.items():
            pat = re.escape(root.replace("_", " "))
            score = 0.0
            if re.search(rf"\b{pat}\b", t):
                score = 1.0
            elif root in t or any(w in t for w in gloss.split()[:2]):
                score = 0.45
            if score > 0:
                hits.append(
                    MeaningHit(
                        root=root,
                        engine=self.name,
                        gloss=gloss,
                        score=score,
                        vector=self.table[root],
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def embed_text(self, text: str) -> np.ndarray:
        hits = self.match(text, top_k=5)
        if not hits:
            return np.zeros(self.dim, dtype=np.float64)
        acc = np.zeros(self.dim, dtype=np.float64)
        wsum = 0.0
        for h in hits:
            acc += h.score * h.vector
            wsum += h.score
        return acc / max(wsum, 1e-9)


class MultiMeaningField:
    """Fuses Latin + Sanskrit + Nahuatl meaning into one d_model vector."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.latin = MeaningEngine("latin", LATIN_ROOTS, dim)
        self.sanskrit = MeaningEngine("sanskrit", SANSKRIT_ROOTS, dim)
        self.nahuatl = MeaningEngine("nahuatl", NAHUATL_ROOTS, dim)
        # Geometric prior in φ proportions
        blend = np.array([1.0, PHI_INV, 1.0 / PHI], dtype=np.float64)
        self.blend = blend / blend.sum()
        self.inject = phi_init((dim, dim), seed=7, layer=0) * 0.1

    def embed(self, text: str) -> np.ndarray:
        parts = [
            self.latin.embed_text(text),
            self.sanskrit.embed_text(text),
            self.nahuatl.embed_text(text),
        ]
        fused = (
            self.blend[0] * parts[0]
            + self.blend[1] * parts[1]
            + self.blend[2] * parts[2]
        )
        out = self.inject @ fused
        n = float(np.linalg.norm(out)) or 1.0
        return out / n

    def annotate(self, text: str) -> List[Dict[str, object]]:
        notes: List[Dict[str, object]] = []
        for eng in (self.latin, self.sanskrit, self.nahuatl):
            for h in eng.match(text, top_k=2):
                notes.append(
                    {
                        "engine": h.engine,
                        "root": h.root,
                        "gloss": h.gloss,
                        "score": h.score,
                    }
                )
        return notes
