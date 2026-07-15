"""Spectral domain definitions for multi-corpus pretraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class SpectralDomain:
    """Definition of a spectral domain with characteristic parameters.

    Attributes:
        name: Human-readable domain name.
        key: Unique domain identifier.
        frequency_range: Typical (min_hz, max_hz) frequency range.
        typical_sampling_rate: Common sampling rate in Hz.
        amplitude_units: Typical amplitude units.
        description: Domain description.
    """

    name: str
    key: str
    frequency_range: Tuple[float, float]
    typical_sampling_rate: float
    amplitude_units: str = "linear"
    description: str = ""


# Registry of known spectral domains
DOMAIN_REGISTRY: Dict[str, SpectralDomain] = {
    "seismic": SpectralDomain(
        name="Seismic",
        key="seismic",
        frequency_range=(0.01, 50.0),
        typical_sampling_rate=100.0,
        amplitude_units="g",
        description="Earthquake ground motion spectra (FAS, response spectra).",
    ),
    "vibration": SpectralDomain(
        name="Structural Vibration",
        key="vibration",
        frequency_range=(0.1, 1000.0),
        typical_sampling_rate=2048.0,
        amplitude_units="m/s^2",
        description="Structural health monitoring and mechanical vibration spectra.",
    ),
    "eeg": SpectralDomain(
        name="EEG",
        key="eeg",
        frequency_range=(0.5, 100.0),
        typical_sampling_rate=256.0,
        amplitude_units="uV",
        description="Electroencephalography power spectral density.",
    ),
    "ecg": SpectralDomain(
        name="ECG",
        key="ecg",
        frequency_range=(0.05, 150.0),
        typical_sampling_rate=500.0,
        amplitude_units="mV",
        description="Electrocardiography frequency-domain representations.",
    ),
    "audio": SpectralDomain(
        name="Audio",
        key="audio",
        frequency_range=(20.0, 20000.0),
        typical_sampling_rate=44100.0,
        amplitude_units="dB",
        description="Audio spectra, mel-frequency, and spectrograms.",
    ),
    "rf": SpectralDomain(
        name="RF/EM",
        key="rf",
        frequency_range=(1e3, 1e9),
        typical_sampling_rate=1e6,
        amplitude_units="dBm",
        description="Radio frequency and electromagnetic spectral sweeps.",
    ),
    "financial": SpectralDomain(
        name="Financial Frequency",
        key="financial",
        frequency_range=(1e-6, 1.0),
        typical_sampling_rate=1.0,
        amplitude_units="normalized",
        description="Fourier-transformed price/volume series and cyclical signatures.",
    ),
    "satellite_edge": SpectralDomain(
        name="Satellite Edge Communication",
        key="satellite_edge",
        frequency_range=(3.0, 3e14),
        typical_sampling_rate=1e9,
        amplitude_units="dBW",
        description="Hz-ladder satellite edge spectra spanning ELF to optical ISL.",
    ),
    "eco_hz": SpectralDomain(
        name="Eco-Hz / Earth Resonance",
        key="eco_hz",
        frequency_range=(1e-8, 50.0),
        typical_sampling_rate=100.0,
        amplitude_units="pT",
        description="Schumann resonances and geophysical oscillations for eco-aware timing.",
    ),
}
