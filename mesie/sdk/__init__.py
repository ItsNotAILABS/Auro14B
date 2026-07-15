"""MAESI — Multi-Agent Embodied Spectral Intelligence.

Powered by NeuroAIX™ — The Connectome Intelligence Engine.
"""

__sdk_version__ = "1.1.0"
__brand__ = "MAESI Powered by NeuroAIX"
__engine__ = "MESIE — Multi-Element Spectral Intelligence Engine"

from mesie.sdk.constants import (
    PLANCK_SPECTRAL,
    BOLTZMANN_SPECTRAL,
    SPEED_OF_LIGHT_SPECTRAL,
    GRAVITATIONAL_SPECTRAL,
    AVOGADRO_SPECTRAL,
    FINE_STRUCTURE_SPECTRAL,
    UniversalSpectralConstant,
    ALL_CONSTANTS,
)
from mesie.sdk.physical_laws import (
    PhysicalLaw,
    get_fundamental_laws,
    get_law_by_name,
    SpectralLawRegistry,
)
from mesie.sdk.chemical_elements import (
    SpectralElement,
    get_periodic_table,
    get_element_by_symbol,
    get_elements_by_group,
)
from mesie.sdk.biological_systems import (
    BiologicalSystem,
    OrganismSpectralProfile,
    get_biological_systems,
    get_organism_profile,
)
from mesie.sdk.technical_library import (
    TechnicalConcept,
    TechnicalDomain,
    get_technical_library,
    get_technical_by_domain,
    get_technical_matrix,
)
from mesie.sdk.research_knowledge import (
    ResearchEntry,
    ResearchField,
    get_research_catalog,
    get_research_by_field,
    search_research,
    get_research_matrix,
)
from mesie.sdk.fast_compute import FastSpectralCompute, SpeedBenchmark
from mesie.sdk.maesi_client import MAESIClient, MAESIQueryResult, MAESIRunReport, KnowledgeStats
from mesie.sdk.neuroaix_engine import (
    NeuroAIXEngine,
    MAESIObservationEncoder,
    CognitiveIntegrationLoop,
    MAESIObservation,
)
from mesie.sdk.solus import (
    SDKSolusOrganism,
    SolusLogicProver,
    SolusPatternForge,
    SOLUS_BRAND,
    LOCAL_ENGINE,
)

__all__ = [
    "__sdk_version__",
    "__brand__",
    "__engine__",
    "ALL_CONSTANTS",
    "PLANCK_SPECTRAL",
    "BOLTZMANN_SPECTRAL",
    "SPEED_OF_LIGHT_SPECTRAL",
    "GRAVITATIONAL_SPECTRAL",
    "AVOGADRO_SPECTRAL",
    "FINE_STRUCTURE_SPECTRAL",
    "UniversalSpectralConstant",
    "PhysicalLaw",
    "get_fundamental_laws",
    "get_law_by_name",
    "SpectralLawRegistry",
    "SpectralElement",
    "get_periodic_table",
    "get_element_by_symbol",
    "get_elements_by_group",
    "BiologicalSystem",
    "OrganismSpectralProfile",
    "get_biological_systems",
    "get_organism_profile",
    "TechnicalConcept",
    "TechnicalDomain",
    "get_technical_library",
    "get_technical_by_domain",
    "get_technical_matrix",
    "ResearchEntry",
    "ResearchField",
    "get_research_catalog",
    "get_research_by_field",
    "search_research",
    "get_research_matrix",
    "FastSpectralCompute",
    "SpeedBenchmark",
    "MAESIClient",
    "MAESIQueryResult",
    "MAESIRunReport",
    "KnowledgeStats",
    "NeuroAIXEngine",
    "MAESIObservationEncoder",
    "CognitiveIntegrationLoop",
    "MAESIObservation",
    "SDKSolusOrganism",
    "SolusLogicProver",
    "SolusPatternForge",
    "SOLUS_BRAND",
    "LOCAL_ENGINE",
]

from mesie.sdk.intelligence_sdk import SpectralIntelligenceSDK
from mesie.sdk.universal_lab_sdk import UniversalLabSDK

__all__.append("SpectralIntelligenceSDK")
__all__.append("UniversalLabSDK")
