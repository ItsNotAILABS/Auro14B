"""Biological systems encoded as spectral entities for MAESI/NeuroAIX.

Living organisms are spectral engines. Every biological process — from
ATP hydrolysis to neural firing to photosynthesis — has a characteristic
spectral signature in frequency space. This module encodes biological
systems as first-class spectral citizens within the MESIE/NeuroAIX
cognitive architecture.

Systems Encoded
---------------
- Cellular respiration (mitochondrial electron transport chain)
- Photosynthesis (chloroplast light reactions)
- Neural signaling (action potentials, synaptic transmission)
- Cardiac electrophysiology (ECG spectral signatures)
- Muscular contraction (EMG spectral signatures)
- DNA replication and transcription
- Immune response (cytokine signaling cascades)
- Endocrine signaling (hormonal oscillations)
- Circadian rhythms (suprachiasmatic nucleus oscillations)
- Metabolic networks (glycolysis, Krebs cycle)

Nomenclature follows:
- IUPAC biochemistry nomenclature
- Gene Ontology (GO) terms
- Human Protein Atlas classifications
- ICD-11 physiological systems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class BiologicalScale(Enum):
    """Scale of biological organization."""

    MOLECULAR = "molecular"
    ORGANELLE = "organelle"
    CELLULAR = "cellular"
    TISSUE = "tissue"
    ORGAN = "organ"
    SYSTEM = "system"
    ORGANISM = "organism"
    POPULATION = "population"
    ECOSYSTEM = "ecosystem"


class PhysiologicalDomain(Enum):
    """ICD-11 aligned physiological system classification."""

    NERVOUS = "nervous_system"
    CARDIOVASCULAR = "cardiovascular_system"
    RESPIRATORY = "respiratory_system"
    MUSCULOSKELETAL = "musculoskeletal_system"
    ENDOCRINE = "endocrine_system"
    IMMUNE = "immune_system"
    DIGESTIVE = "digestive_system"
    INTEGUMENTARY = "integumentary_system"
    REPRODUCTIVE = "reproductive_system"
    EXCRETORY = "excretory_system"
    METABOLIC = "metabolic_processes"
    GENETIC = "genetic_information"


@dataclass
class BiologicalSystem:
    """A biological system encoded as a spectral entity.

    Attributes
    ----------
    name : str
        Official name (GO/IUPAC nomenclature).
    system_id : str
        Unique identifier (GO:XXXXXXX format where applicable).
    domain : PhysiologicalDomain
        Physiological system classification.
    scale : BiologicalScale
        Organizational scale.
    description : str
        Functional description.
    characteristic_frequencies_hz : List[float]
        Dominant oscillation frequencies of this biological process.
    spectral_signature : np.ndarray
        32-dimensional spectral encoding.
    energy_consumption_atp_per_s : float
        ATP turnover rate (molecules/second).
    timescale_s : Tuple[float, float]
        Operating timescale range in seconds.
    key_molecules : List[str]
        Key molecular participants (IUPAC names).
    connectome_binding : List[str]
        NeuroAIX brain regions that model this system.
    """

    name: str
    system_id: str
    domain: PhysiologicalDomain
    scale: BiologicalScale
    description: str
    characteristic_frequencies_hz: List[float] = field(default_factory=list)
    spectral_signature: np.ndarray = field(default_factory=lambda: np.zeros(32))
    energy_consumption_atp_per_s: float = 0.0
    timescale_s: Tuple[float, float] = (1e-3, 1.0)
    key_molecules: List[str] = field(default_factory=list)
    connectome_binding: List[str] = field(default_factory=list)

    def to_embedding(self) -> np.ndarray:
        """Produce 64-dim embedding for connectome injection."""
        scale_vec = np.zeros(9)
        scale_vec[list(BiologicalScale).index(self.scale)] = 1.0
        domain_vec = np.zeros(12)
        domain_vec[list(PhysiologicalDomain).index(self.domain)] = 1.0
        meta = np.array([
            np.log10(self.energy_consumption_atp_per_s + 1),
            np.log10(self.timescale_s[0] + 1e-10),
            np.log10(self.timescale_s[1] + 1e-10),
            len(self.key_molecules) / 20.0,
            np.mean(self.characteristic_frequencies_hz) / 1e6 if self.characteristic_frequencies_hz else 0.0,
        ])
        padding = np.zeros(6)
        return np.concatenate([self.spectral_signature, scale_vec, domain_vec, meta, padding])


def _bio_sig(seed: int) -> np.ndarray:
    """Deterministic biological spectral signature."""
    return np.random.default_rng(seed).uniform(0, 1, 32).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# BIOLOGICAL SYSTEMS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEMS: List[BiologicalSystem] = [
    BiologicalSystem(
        name="Mitochondrial Electron Transport Chain",
        system_id="GO:0022900",
        domain=PhysiologicalDomain.METABOLIC,
        scale=BiologicalScale.ORGANELLE,
        description="Oxidative phosphorylation generating ATP via electron transfer through Complexes I-IV and ATP synthase.",
        characteristic_frequencies_hz=[1e3, 1e4, 1e5],  # Molecular vibration bands
        spectral_signature=_bio_sig(900),
        energy_consumption_atp_per_s=1e8,
        timescale_s=(1e-6, 1e-3),
        key_molecules=["NADH", "Ubiquinone (Coenzyme Q10)", "Cytochrome c", "ATP Synthase (Complex V)"],
        connectome_binding=["INS", "HYP", "ACC"],
    ),
    BiologicalSystem(
        name="Photosystem II — Light-Dependent Reactions",
        system_id="GO:0009523",
        domain=PhysiologicalDomain.METABOLIC,
        scale=BiologicalScale.ORGANELLE,
        description="Water-splitting complex in chloroplasts; captures photons at 680nm to drive electron flow.",
        characteristic_frequencies_hz=[4.41e14, 4.62e14],  # 680nm, 649nm absorption
        spectral_signature=_bio_sig(901),
        energy_consumption_atp_per_s=0,  # Produces energy
        timescale_s=(1e-12, 1e-6),  # Picosecond photochemistry
        key_molecules=["Chlorophyll a (P680)", "Pheophytin", "Plastoquinone", "Manganese Cluster (OEC)"],
        connectome_binding=["V1", "V2"],
    ),
    BiologicalSystem(
        name="Action Potential Propagation",
        system_id="GO:0019228",
        domain=PhysiologicalDomain.NERVOUS,
        scale=BiologicalScale.CELLULAR,
        description="Voltage-gated Na⁺/K⁺ channel-mediated depolarization wave along axonal membrane.",
        characteristic_frequencies_hz=[1.0, 10.0, 40.0, 100.0, 600.0],  # Neural firing rates
        spectral_signature=_bio_sig(902),
        energy_consumption_atp_per_s=1e9,
        timescale_s=(1e-3, 5e-3),  # 1-5ms action potential duration
        key_molecules=["Na⁺/K⁺-ATPase", "Voltage-gated Na⁺ channel (Nav1.6)", "Voltage-gated K⁺ channel (Kv1.2)"],
        connectome_binding=["M1", "S1", "DLPFC", "HPC"],
    ),
    BiologicalSystem(
        name="Synaptic Neurotransmission",
        system_id="GO:0007268",
        domain=PhysiologicalDomain.NERVOUS,
        scale=BiologicalScale.CELLULAR,
        description="Chemical synaptic transmission via vesicle fusion, neurotransmitter release, and receptor binding.",
        characteristic_frequencies_hz=[5.0, 20.0, 50.0],  # Synaptic event rates
        spectral_signature=_bio_sig(903),
        energy_consumption_atp_per_s=5e8,
        timescale_s=(5e-4, 5e-2),
        key_molecules=["Glutamate", "GABA (γ-Aminobutyric acid)", "Acetylcholine", "SNARE Complex (Syntaxin/SNAP-25)"],
        connectome_binding=["DLPFC", "HPC", "AMY", "VTA"],
    ),
    BiologicalSystem(
        name="Cardiac Electrophysiology",
        system_id="GO:0086001",
        domain=PhysiologicalDomain.CARDIOVASCULAR,
        scale=BiologicalScale.ORGAN,
        description="SA node pacemaker activity driving cardiac depolarization through the His-Purkinje system.",
        characteristic_frequencies_hz=[1.0, 1.2, 2.4, 4.8],  # Heart rate harmonics ~72bpm
        spectral_signature=_bio_sig(904),
        energy_consumption_atp_per_s=2e10,
        timescale_s=(0.2, 1.0),
        key_molecules=["HCN4 (Funny Channel)", "L-type Ca²⁺ Channel (Cav1.2)", "Ryanodine Receptor (RyR2)", "Troponin C"],
        connectome_binding=["INS", "ACC", "AMY"],
    ),
    BiologicalSystem(
        name="Skeletal Muscle Contraction",
        system_id="GO:0006936",
        domain=PhysiologicalDomain.MUSCULOSKELETAL,
        scale=BiologicalScale.TISSUE,
        description="Actin-myosin cross-bridge cycling driven by Ca²⁺ release from sarcoplasmic reticulum.",
        characteristic_frequencies_hz=[10.0, 20.0, 50.0, 100.0],  # Motor unit firing
        spectral_signature=_bio_sig(905),
        energy_consumption_atp_per_s=1e11,
        timescale_s=(1e-2, 1.0),
        key_molecules=["Myosin Heavy Chain", "Actin (F-actin)", "Troponin-Tropomyosin Complex", "Ca²⁺-ATPase (SERCA)"],
        connectome_binding=["M1", "SMA", "PMC", "CB"],
    ),
    BiologicalSystem(
        name="DNA Replication",
        system_id="GO:0006260",
        domain=PhysiologicalDomain.GENETIC,
        scale=BiologicalScale.MOLECULAR,
        description="Semi-conservative replication of genomic DNA by the replisome complex.",
        characteristic_frequencies_hz=[1e3, 1e4],  # Nucleotide incorporation rate
        spectral_signature=_bio_sig(906),
        energy_consumption_atp_per_s=1e7,
        timescale_s=(3600.0, 28800.0),  # Hours for complete replication
        key_molecules=["DNA Polymerase III Holoenzyme", "Helicase (MCM2-7)", "Primase (DnaG)", "PCNA (Sliding Clamp)"],
        connectome_binding=["DLPFC", "HPC"],
    ),
    BiologicalSystem(
        name="Adaptive Immune Response — T Cell Activation",
        system_id="GO:0002250",
        domain=PhysiologicalDomain.IMMUNE,
        scale=BiologicalScale.CELLULAR,
        description="MHC-TCR recognition triggering clonal expansion and effector differentiation.",
        characteristic_frequencies_hz=[0.001, 0.01, 0.1],  # Cell division rate
        spectral_signature=_bio_sig(907),
        energy_consumption_atp_per_s=5e9,
        timescale_s=(3600.0, 604800.0),  # Hours to weeks
        key_molecules=["T Cell Receptor (TCR αβ)", "MHC Class I/II", "CD28 (Costimulatory)", "Interleukin-2 (IL-2)"],
        connectome_binding=["INS", "HYP", "ACC"],
    ),
    BiologicalSystem(
        name="Circadian Rhythm Oscillation",
        system_id="GO:0007623",
        domain=PhysiologicalDomain.ENDOCRINE,
        scale=BiologicalScale.SYSTEM,
        description="~24h transcription-translation feedback loop in SCN driving sleep-wake cycles.",
        characteristic_frequencies_hz=[1.157e-5, 2.314e-5],  # 1/86400 Hz and harmonic
        spectral_signature=_bio_sig(908),
        energy_consumption_atp_per_s=1e6,
        timescale_s=(43200.0, 86400.0),  # 12-24 hours
        key_molecules=["CLOCK Protein", "BMAL1", "PER1/PER2 (Period)", "CRY1/CRY2 (Cryptochrome)"],
        connectome_binding=["HYP", "SCN", "PIN"],
    ),
    BiologicalSystem(
        name="Glycolysis — Embden-Meyerhof-Parnas Pathway",
        system_id="GO:0006096",
        domain=PhysiologicalDomain.METABOLIC,
        scale=BiologicalScale.CELLULAR,
        description="Anaerobic conversion of glucose to pyruvate yielding 2 ATP and 2 NADH per glucose.",
        characteristic_frequencies_hz=[100.0, 1000.0],  # Enzyme turnover
        spectral_signature=_bio_sig(909),
        energy_consumption_atp_per_s=0,  # Net producer
        timescale_s=(1e-3, 1.0),
        key_molecules=["Hexokinase", "Phosphofructokinase-1 (PFK-1)", "Pyruvate Kinase", "Glucose-6-phosphate"],
        connectome_binding=["INS", "HYP"],
    ),
    BiologicalSystem(
        name="Citric Acid Cycle (Krebs Cycle)",
        system_id="GO:0006099",
        domain=PhysiologicalDomain.METABOLIC,
        scale=BiologicalScale.ORGANELLE,
        description="Central metabolic hub oxidizing acetyl-CoA to CO₂, producing NADH/FADH₂ for ETC.",
        characteristic_frequencies_hz=[500.0, 5000.0],
        spectral_signature=_bio_sig(910),
        energy_consumption_atp_per_s=0,
        timescale_s=(1e-2, 1.0),
        key_molecules=["Citrate Synthase", "Isocitrate Dehydrogenase", "α-Ketoglutarate Dehydrogenase", "Succinate Dehydrogenase"],
        connectome_binding=["INS", "HYP"],
    ),
    BiologicalSystem(
        name="Long-Term Potentiation (LTP)",
        system_id="GO:0060291",
        domain=PhysiologicalDomain.NERVOUS,
        scale=BiologicalScale.CELLULAR,
        description="Activity-dependent strengthening of synaptic transmission underlying learning and memory.",
        characteristic_frequencies_hz=[5.0, 8.0, 40.0, 100.0],  # Theta-gamma coupling
        spectral_signature=_bio_sig(911),
        energy_consumption_atp_per_s=1e8,
        timescale_s=(1e-1, 3600.0),
        key_molecules=["NMDA Receptor (GluN2B)", "AMPA Receptor (GluA1)", "CaMKII", "CREB (Transcription Factor)"],
        connectome_binding=["HPC", "DLPFC", "ACC"],
    ),
    BiologicalSystem(
        name="Respiratory Gas Exchange",
        system_id="GO:0007585",
        domain=PhysiologicalDomain.RESPIRATORY,
        scale=BiologicalScale.ORGAN,
        description="O₂/CO₂ diffusion across alveolar-capillary membrane driven by partial pressure gradients.",
        characteristic_frequencies_hz=[0.2, 0.25, 0.5],  # Breathing rate ~15/min
        spectral_signature=_bio_sig(912),
        energy_consumption_atp_per_s=5e9,
        timescale_s=(2.0, 6.0),
        key_molecules=["Hemoglobin (Hb)", "Carbonic Anhydrase", "Surfactant Protein (SP-A/SP-B)", "2,3-BPG"],
        connectome_binding=["BS", "INS"],
    ),
    BiologicalSystem(
        name="Hypothalamic-Pituitary-Adrenal (HPA) Axis",
        system_id="GO:0061179",
        domain=PhysiologicalDomain.ENDOCRINE,
        scale=BiologicalScale.SYSTEM,
        description="Neuroendocrine stress response axis: CRH → ACTH → Cortisol with negative feedback.",
        characteristic_frequencies_hz=[2.78e-4, 1.16e-5],  # Ultradian pulses, circadian
        spectral_signature=_bio_sig(913),
        energy_consumption_atp_per_s=1e7,
        timescale_s=(60.0, 86400.0),
        key_molecules=["Corticotropin-Releasing Hormone (CRH)", "ACTH (Corticotropin)", "Cortisol", "Glucocorticoid Receptor (NR3C1)"],
        connectome_binding=["HYP", "AMY", "HPC", "ACC"],
    ),
    BiologicalSystem(
        name="Visual Phototransduction",
        system_id="GO:0007601",
        domain=PhysiologicalDomain.NERVOUS,
        scale=BiologicalScale.CELLULAR,
        description="Rod/cone photoreceptor G-protein cascade converting photons to neural signals.",
        characteristic_frequencies_hz=[4.3e14, 5.5e14, 6.5e14],  # Red, green, blue cone peaks
        spectral_signature=_bio_sig(914),
        energy_consumption_atp_per_s=1e8,
        timescale_s=(1e-3, 0.2),
        key_molecules=["Rhodopsin", "Transducin (Gαt)", "Phosphodiesterase 6 (PDE6)", "Retinal (11-cis/all-trans)"],
        connectome_binding=["V1", "V2", "V3", "LGN"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANISM-LEVEL SPECTRAL PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrganismSpectralProfile:
    """Complete spectral profile of a living organism.

    Integrates all subsystems into a unified spectral representation
    for the NeuroAIX connectome to reason about as a whole entity.
    """

    name: str
    taxonomy: str  # Linnean classification
    systems: List[str]  # Names of active BiologicalSystems
    dominant_frequencies_hz: List[float]
    metabolic_rate_watts: float
    neural_complexity: float  # 0-1 scale
    spectral_signature: np.ndarray = field(default_factory=lambda: np.zeros(64))

    def to_embedding(self) -> np.ndarray:
        """Produce 96-dim organism embedding."""
        meta = np.array([
            self.metabolic_rate_watts / 100.0,
            self.neural_complexity,
            len(self.systems) / 15.0,
            np.mean(self.dominant_frequencies_hz) / 1e6 if self.dominant_frequencies_hz else 0.0,
        ])
        padding = np.zeros(28)
        return np.concatenate([self.spectral_signature, meta, padding])


_ORGANISMS: List[OrganismSpectralProfile] = [
    OrganismSpectralProfile(
        name="Homo sapiens",
        taxonomy="Animalia > Chordata > Mammalia > Primates > Hominidae > Homo > H. sapiens",
        systems=[s.name for s in _SYSTEMS],
        dominant_frequencies_hz=[1.0, 10.0, 40.0, 100.0, 1.2, 0.25],
        metabolic_rate_watts=80.0,
        neural_complexity=1.0,
        spectral_signature=np.random.default_rng(42).uniform(0, 1, 64),
    ),
    OrganismSpectralProfile(
        name="Mus musculus (Laboratory Mouse)",
        taxonomy="Animalia > Chordata > Mammalia > Rodentia > Muridae > Mus > M. musculus",
        systems=["Action Potential Propagation", "Cardiac Electrophysiology",
                 "Glycolysis — Embden-Meyerhof-Parnas Pathway", "Circadian Rhythm Oscillation"],
        dominant_frequencies_hz=[8.0, 30.0, 80.0, 10.0, 0.3],
        metabolic_rate_watts=0.5,
        neural_complexity=0.3,
        spectral_signature=np.random.default_rng(43).uniform(0, 1, 64),
    ),
    OrganismSpectralProfile(
        name="Arabidopsis thaliana (Thale Cress)",
        taxonomy="Plantae > Tracheophyta > Magnoliopsida > Brassicales > Brassicaceae > Arabidopsis",
        systems=["Photosystem II — Light-Dependent Reactions", "Circadian Rhythm Oscillation",
                 "DNA Replication"],
        dominant_frequencies_hz=[4.41e14, 1.157e-5, 1e3],
        metabolic_rate_watts=0.001,
        neural_complexity=0.0,
        spectral_signature=np.random.default_rng(44).uniform(0, 1, 64),
    ),
    OrganismSpectralProfile(
        name="Escherichia coli",
        taxonomy="Bacteria > Proteobacteria > Gammaproteobacteria > Enterobacterales > Enterobacteriaceae > Escherichia",
        systems=["Glycolysis — Embden-Meyerhof-Parnas Pathway", "DNA Replication",
                 "Citric Acid Cycle (Krebs Cycle)"],
        dominant_frequencies_hz=[1e3, 1e4, 5e-4],  # Division ~20min
        metabolic_rate_watts=1e-12,
        neural_complexity=0.0,
        spectral_signature=np.random.default_rng(45).uniform(0, 1, 64),
    ),
    OrganismSpectralProfile(
        name="Caenorhabditis elegans",
        taxonomy="Animalia > Nematoda > Chromadorea > Rhabditida > Rhabditidae > Caenorhabditis",
        systems=["Action Potential Propagation", "Synaptic Neurotransmission",
                 "Skeletal Muscle Contraction"],
        dominant_frequencies_hz=[1.0, 5.0, 10.0],
        metabolic_rate_watts=1e-6,
        neural_complexity=0.01,  # 302 neurons
        spectral_signature=np.random.default_rng(46).uniform(0, 1, 64),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════════

def get_biological_systems() -> List[BiologicalSystem]:
    """Return all registered biological systems."""
    return list(_SYSTEMS)


def get_system_by_name(name: str) -> Optional[BiologicalSystem]:
    """Look up a biological system by name."""
    for s in _SYSTEMS:
        if s.name == name:
            return s
    return None


def get_systems_by_domain(domain: PhysiologicalDomain) -> List[BiologicalSystem]:
    """Filter biological systems by physiological domain."""
    return [s for s in _SYSTEMS if s.domain == domain]


def get_organism_profile(name: str) -> Optional[OrganismSpectralProfile]:
    """Look up an organism spectral profile by species name."""
    for o in _ORGANISMS:
        if name.lower() in o.name.lower():
            return o
    return None


def get_all_organism_profiles() -> List[OrganismSpectralProfile]:
    """Return all organism profiles."""
    return list(_ORGANISMS)


def get_biological_embedding_matrix() -> np.ndarray:
    """Return (N, 64) embedding matrix of all biological systems."""
    return np.stack([s.to_embedding() for s in _SYSTEMS])
