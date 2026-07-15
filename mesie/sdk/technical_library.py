"""Technical engineering library — signal, robotics, power, orbital, seismic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class TechnicalDomain(str, Enum):
    SIGNAL_PROCESSING = "signal_processing"
    ROBOTICS = "robotics"
    POWER_SYSTEMS = "power_systems"
    ORBITAL = "orbital_mechanics"
    SEISMIC = "seismic"
    STRUCTURAL = "structural"
    SPECTRAL_ML = "spectral_ml"
    TIME_FREQUENCY = "time_frequency"
    ANN_RETRIEVAL = "ann_retrieval"


@dataclass
class TechnicalConcept:
    name: str
    domain: TechnicalDomain
    description: str
    key_metrics: List[str]
    mesie_module: str
    typical_freq_hz: tuple[float, float] = (0.01, 1e12)
    tags: List[str] = field(default_factory=list)

    def to_embedding(self) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(self.name)) % (2**31))
        base = rng.standard_normal(32)
        dom = list(TechnicalDomain).index(self.domain) / max(len(TechnicalDomain), 1)
        base[0] = dom
        base[1] = np.log10(max(self.typical_freq_hz[0], 1e-12))
        base[2] = np.log10(self.typical_freq_hz[1])
        return base.astype(np.float64)


_TECHNICAL: List[TechnicalConcept] = [
    TechnicalConcept("STFT Spectrogram", TechnicalDomain.TIME_FREQUENCY, "Short-time Fourier transform for non-stationary signals.", ["frame_ms", "hop", "n_fft"], "mesie.signal.time_frequency", (0.1, 1e5), ["tf", "laptop"]),
    TechnicalConcept("Salient TF Peaks", TechnicalDomain.SIGNAL_PROCESSING, "Landmark maxima in time-frequency maps for fingerprinting.", ["n_peaks", "threshold_pct"], "mesie.signal.salient", (0.01, 1e4), ["fingerprint"]),
    TechnicalConcept("LSH Spectral Hash", TechnicalDomain.ANN_RETRIEVAL, "Locality-sensitive hashing for compact bucket lookup.", ["n_planes", "bucket"], "mesie.embeddings.lsh", tags=["fast"]),
    TechnicalConcept("ANN Cosine Rerank", TechnicalDomain.ANN_RETRIEVAL, "Approximate NN with LSH pre-filter and exact rerank.", ["top_k", "metric"], "mesie.embeddings.ann", tags=["fast"]),
    TechnicalConcept("Pump Vibration Baseline", TechnicalDomain.ROBOTICS, "Healthy rotating machinery PSD/FAS reference.", ["rpm", "bearing"], "data.reference.vibration_monitoring", (1, 1e4), ["plc", "edge"]),
    TechnicalConcept("Anomaly vs Baseline", TechnicalDomain.ROBOTICS, "Spectral deviation scoring for fault alerts.", ["threshold"], "mesie.cognitive.agent_state_adapter", tags=["control"]),
    TechnicalConcept("Schumann Eco-Hz", TechnicalDomain.POWER_SYSTEMS, "Earth-ionosphere cavity resonances for timing.", ["7.83_Hz", "modes"], "data.spectral_library.schumann", (7, 50), ["power"]),
    TechnicalConcept("EM Band Ladder", TechnicalDomain.POWER_SYSTEMS, "ELF through SHF band definitions (ITU/IEEE).", ["tier", "center_Hz"], "mesie.edge.hz_ladder", (3, 300e9)),
    TechnicalConcept("LEO Contact Window", TechnicalDomain.ORBITAL, "Ground pass duration from orbital mechanics.", ["altitude_km", "period_s"], "mesie.edge.satellite_nodes", (1e-5, 1e-2)),
    TechnicalConcept("Orbital Edge Gate", TechnicalDomain.ORBITAL, "Phase-gated spectral transients at harmonic periods.", ["period_days", "threshold"], "scripts.orbital_edge_50d_analysis", (1e-8, 1.0)),
    TechnicalConcept("Earthquake PSD Anchor", TechnicalDomain.SEISMIC, "Broadband seismic coupling reference spectrum.", ["magnitude", "distance_km"], "data.reference.earthquake_psd", (0.01, 50)),
    TechnicalConcept("RotDNN Orientation", TechnicalDomain.SEISMIC, "Orientation-dependent spectral intensity metric.", ["period_s"], "data.reference.rotdnn", (0.1, 10)),
    TechnicalConcept("Structural FAS", TechnicalDomain.STRUCTURAL, "Frequency-dependent site/building amplification.", ["Vs30", "site_class"], "data.reference.structural_fas", (0.1, 50)),
    TechnicalConcept("Spectral Vectorizer", TechnicalDomain.SPECTRAL_ML, "Band-energy + statistics embedding.", ["n_bands"], "mesie.embeddings.vectorizers", tags=["core"]),
    TechnicalConcept("Fingerprint Pipeline", TechnicalDomain.SPECTRAL_ML, "TF → salient → LSH → ANN end-to-end.", ["dim"], "mesie.embeddings.fingerprint", tags=["core", "fast"]),
    TechnicalConcept("Octopus Multi-Arm Control", TechnicalDomain.ROBOTICS, "Eight-arm orchestration across MESIE engines.", ["arms"], "mesie.octopus.controller", tags=["workflow"]),
    TechnicalConcept("Internal API Bus", TechnicalDomain.SIGNAL_PROCESSING, "Cross-engine message routing on laptop.", ["topic"], "mesie.internal_api.bus", tags=["architecture"]),
    TechnicalConcept("Cross-Domain Transfer", TechnicalDomain.SPECTRAL_ML, "Align embeddings across seismic/vibration/audio.", ["CORAL", "MMD"], "mesie.transfer.cross_domain", tags=["research"]),
    TechnicalConcept("Intelligence Protocol", TechnicalDomain.SPECTRAL_ML, "Reasoning layer over spectral embeddings.", ["conclusion", "confidence"], "mesie.ai.intelligence_protocols"),
    TechnicalConcept("Hz Virtual Chip", TechnicalDomain.POWER_SYSTEMS, "On-device sub-ms spectral decisions without cloud.", ["compares_per_sec"], "docs.laptop_virtual_chip", tags=["product"]),
]


def get_technical_library() -> List[TechnicalConcept]:
    return list(_TECHNICAL)


def get_technical_by_domain(domain: TechnicalDomain) -> List[TechnicalConcept]:
    return [t for t in _TECHNICAL if t.domain == domain]


def get_technical_matrix() -> np.ndarray:
    return np.stack([t.to_embedding() for t in _TECHNICAL])