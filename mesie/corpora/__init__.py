"""Multi-domain spectral corpora abstraction.

Defines domain types and corpus base classes for loading spectral data
from heterogeneous sources (seismic, vibration, EEG/ECG, audio, RF/EM, financial).
"""

from mesie.corpora.domains import SpectralDomain, DOMAIN_REGISTRY
from mesie.corpora.base import SpectralCorpus, CorpusRecord

__all__ = [
    "CorpusRecord",
    "DOMAIN_REGISTRY",
    "SpectralCorpus",
    "SpectralDomain",
]
