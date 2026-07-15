"""Research knowledge base — methods, fields, and MESIE module links."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class ResearchField(str, Enum):
    SEISMOLOGY = "seismology"
    SIGNAL_PROCESSING = "signal_processing"
    MACHINE_LEARNING = "machine_learning"
    ROBOTICS = "robotics"
    SPACE_SYSTEMS = "space_systems"
    POWER_ELECTROMAGNETICS = "power_electromagnetics"
    COGNITIVE_AI = "cognitive_ai"
    STRUCTURAL_ENGINEERING = "structural_engineering"
    SPECTRAL_THEORY = "spectral_theory"


@dataclass
class ResearchEntry:
    title: str
    field: ResearchField
    summary: str
    methods: List[str]
    mesie_hooks: List[str]
    citations: List[str] = field(default_factory=list)
    year: int = 2020

    def to_embedding(self) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(self.title)) % (2**31))
        v = rng.standard_normal(32)
        v[0] = list(ResearchField).index(self.field) / max(len(ResearchField), 1)
        v[1] = (self.year - 1900) / 200.0
        v[2] = len(self.methods) / 20.0
        return v.astype(np.float64)


_RESEARCH: List[ResearchEntry] = [
    ResearchEntry("PSD Site Response", ResearchField.SEISMOLOGY, "Power spectral density for ground motion and site effects.", ["FFT", "PSD", "damping"], ["mesie.generation.psd", "data.reference.earthquake_psd"], ["Boore et al. NGA"], 2014),
    ResearchEntry("RotDNN Period Ranges", ResearchField.SEISMOLOGY, "Rotation-dependent spectral intensity over period bands.", ["ROTDNN", "orientation"], ["data.reference.rotdnn"], ["Boore & Kishida 2017"], 2017),
    ResearchEntry("FAS Structural Amplification", ResearchField.STRUCTURAL_ENGINEERING, "Fourier amplitude spectrum for building response.", ["FAS", "transfer_function"], ["mesie.generation.fas", "data.reference.structural_fas"], ["Campbell & Bozorgnia"], 2014),
    ResearchEntry("STFT Non-Stationary Analysis", ResearchField.SIGNAL_PROCESSING, "Time-varying spectral content via windowed FFT.", ["STFT", "spectrogram"], ["mesie.signal.time_frequency"], ["Allen 1977"], 1977),
    ResearchEntry("Salient Feature Matching", ResearchField.SIGNAL_PROCESSING, "Landmark-based alignment in TF domain (Shazam-class).", ["peak_picking", "constellation"], ["mesie.signal.salient", "mesie.embeddings.fingerprint"], ["Wang 2003"], 2003),
    ResearchEntry("LSH Approximate Retrieval", ResearchField.MACHINE_LEARNING, "Sublinear similarity search via random hyperplanes.", ["LSH", "cosine"], ["mesie.embeddings.lsh"], ["Charikar 2002"], 2002),
    ResearchEntry("ANN Vector Databases", ResearchField.MACHINE_LEARNING, "Billion-scale embedding search (FAISS/HNSW).", ["ANN", "HNSW", "IVF"], ["mesie.embeddings.ann"], ["Johnson et al. 2019"], 2019),
    ResearchEntry("Domain Adaptation CORAL", ResearchField.MACHINE_LEARNING, "Align second-order statistics across domains.", ["CORAL", "transfer"], ["mesie.transfer.cross_domain"], ["Sun et al. 2016"], 2016),
    ResearchEntry("MMD Domain Divergence", ResearchField.MACHINE_LEARNING, "Maximum mean discrepancy for transfer feasibility.", ["MMD", "kernel"], ["mesie.transfer.cross_domain"], ["Gretton et al. 2012"], 2012),
    ResearchEntry("Condition Monitoring Vibration", ResearchField.ROBOTICS, "Machinery fault detection from vibration spectra.", ["envelope", "order_tracking"], ["data.reference.vibration_monitoring"], ["ISO 10816"], 2016),
    ResearchEntry("Schumann Resonances", ResearchField.POWER_ELECTROMAGNETICS, "ELF cavity modes of Earth-ionosphere waveguide.", ["mode_n", "Q_factor"], ["data.spectral_library.schumann"], ["Schumann 1952"], 1952),
    ResearchEntry("Satellite Link Budget", ResearchField.SPACE_SYSTEMS, "FSPL, Doppler, and contact windows for LEO/MEO.", ["Kepler", "FSPL", "Doppler"], ["mesie.edge.satellite_nodes"], ["ITU-R"], 2020),
    ResearchEntry("Orbital Harmonic Coupling", ResearchField.SPACE_SYSTEMS, "Long/short period gates for spectral edge events.", ["phase_gate", "forecast"], ["scripts.orbital_edge_50d_analysis"], ["MESIE internal"], 2026),
    ResearchEntry("Hz Ladder Propagation", ResearchField.POWER_ELECTROMAGNETICS, "Vertical tier model ELF→optical for comm paths.", ["tier", "latency"], ["mesie.edge.hz_ladder"], ["MESIE internal"], 2026),
    ResearchEntry("Connectome Spectral Binding", ResearchField.COGNITIVE_AI, "Brain-region propagation of spectral observations.", ["connectome", "delay"], ["mesie.sdk.neuroaix_engine", "mesie.connectome"], ["MESIE Paper II"], 2026),
    ResearchEntry("Intelligence Protocols", ResearchField.COGNITIVE_AI, "Adaptive reasoning over spectral memory buffers.", ["observe", "reason"], ["mesie.ai.intelligence_protocols"], ["MESIE v0.2"], 2026),
    ResearchEntry("Multi-Element Spectral Records", ResearchField.SPECTRAL_THEORY, "Unified schema for PSD/FAS/multi-component data.", ["validation", "lineage"], ["mesie.core.records", "mesie.validation"], ["MESIE spec"], 2026),
    ResearchEntry("Helix Vector Retrieval", ResearchField.SPECTRAL_THEORY, "Helical embedding geometry for structured search.", ["helix", "traversal"], ["mesie.helix"], ["MESIE v0.2"], 2026),
    ResearchEntry("Octopus Engineering", ResearchField.ROBOTICS, "Multi-arm control/movement/workflow on internal API.", ["arms", "bus"], ["mesie.octopus", "mesie.internal_api"], ["MESIE v0.2.1"], 2026),
    ResearchEntry("Laptop Virtual Chip", ResearchField.SIGNAL_PROCESSING, "Sub-ms on-device spectral ALU for agents.", ["embed", "match"], ["docs.laptop_virtual_chip", "mesie.embeddings"], ["MESIE product"], 2026),
    ResearchEntry("Bundled Benchmark Corpora", ResearchField.MACHINE_LEARNING, "Classification and embedding training spectra.", ["benchmark", "train"], ["data.benchmarks"], ["MESIE data pkg"], 2026),
    ResearchEntry("Cloudflare Edge API", ResearchField.SIGNAL_PROCESSING, "Worker validation/match at the edge.", ["worker", "REST"], ["workers.mesie-api"], ["MESIE deploy"], 2026),
    ResearchEntry("Deterministic Generation", ResearchField.SIGNAL_PROCESSING, "Seeded synthetic PSD/FAS for reproducible tests.", ["seed", "RNG"], ["mesie.generation", "scripts.determinism_benchmark"], ["MESIE QA"], 2026),
    ResearchEntry("USIT Universal Integration", ResearchField.SPECTRAL_THEORY, "Laws, elements, biology as one spectral substrate.", ["USIT", "MAESI"], ["mesie.sdk"], ["Paper III"], 2026),
]


def get_research_catalog() -> List[ResearchEntry]:
    return list(_RESEARCH)


def get_research_by_field(field: ResearchField) -> List[ResearchEntry]:
    return [r for r in _RESEARCH if r.field == field]


def search_research(query: str, top_k: int = 5) -> List[ResearchEntry]:
    """Keyword search over titles and summaries."""
    q = query.lower()
    scored = []
    for r in _RESEARCH:
        text = f"{r.title} {r.summary} {' '.join(r.methods)}".lower()
        score = sum(1 for w in q.split() if w in text)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


def get_research_matrix() -> np.ndarray:
    return np.stack([r.to_embedding() for r in _RESEARCH])