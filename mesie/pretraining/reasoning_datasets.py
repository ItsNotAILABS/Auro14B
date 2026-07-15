"""Reasoning training datasets — real physics, mathematics, and geometry.

These are structured training sets containing actual scientific knowledge
that MESIE can use for spectral reasoning, embedding calibration, and
transfer learning. Every entry is grounded in real physical law.

Domains covered:
- Fundamental physics constants and spectral relationships
- Mathematical transforms (Fourier, Laplace, wavelet)
- Geometry of wave propagation and resonance
- Thermodynamics / statistical mechanics
- Quantum mechanical spectral lines
- Electromagnetic spectrum structure
- Acoustics and vibration physics
- Signal processing theory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Physics Constants (CODATA 2022 exact values)
# =============================================================================

PHYSICS_CONSTANTS: Dict[str, Dict[str, Any]] = {
    "speed_of_light": {
        "symbol": "c",
        "value": 299_792_458.0,
        "unit": "m/s",
        "uncertainty": 0.0,
        "domain": "electromagnetism",
        "spectral_relevance": "Relates frequency to wavelength: λ = c/f",
    },
    "planck_constant": {
        "symbol": "h",
        "value": 6.62607015e-34,
        "unit": "J⋅s",
        "uncertainty": 0.0,
        "domain": "quantum_mechanics",
        "spectral_relevance": "Energy of spectral line: E = hf",
    },
    "boltzmann_constant": {
        "symbol": "k_B",
        "value": 1.380649e-23,
        "unit": "J/K",
        "uncertainty": 0.0,
        "domain": "thermodynamics",
        "spectral_relevance": "Thermal noise floor: PSD ∝ k_B * T",
    },
    "gravitational_constant": {
        "symbol": "G",
        "value": 6.67430e-11,
        "unit": "m³/(kg⋅s²)",
        "uncertainty": 2.2e-5,
        "domain": "gravitation",
        "spectral_relevance": "Gravitational wave frequency: f_GW = (1/π)√(GM/r³)",
    },
    "electron_mass": {
        "symbol": "m_e",
        "value": 9.1093837015e-31,
        "unit": "kg",
        "uncertainty": 3.0e-10,
        "domain": "quantum_mechanics",
        "spectral_relevance": "Compton wavelength: λ_C = h/(m_e * c)",
    },
    "elementary_charge": {
        "symbol": "e",
        "value": 1.602176634e-19,
        "unit": "C",
        "uncertainty": 0.0,
        "domain": "electromagnetism",
        "spectral_relevance": "Photon energy from voltage: E = eV → f = eV/h",
    },
    "avogadro_number": {
        "symbol": "N_A",
        "value": 6.02214076e23,
        "unit": "1/mol",
        "uncertainty": 0.0,
        "domain": "chemistry",
        "spectral_relevance": "Molar absorption: A = ε * c * l (Beer-Lambert)",
    },
    "stefan_boltzmann": {
        "symbol": "σ",
        "value": 5.670374419e-8,
        "unit": "W/(m²⋅K⁴)",
        "uncertainty": 0.0,
        "domain": "thermodynamics",
        "spectral_relevance": "Black-body total power: P = σ * A * T⁴",
    },
    "rydberg_constant": {
        "symbol": "R_∞",
        "value": 10_973_731.568160,
        "unit": "1/m",
        "uncertainty": 2.1e-12,
        "domain": "atomic_physics",
        "spectral_relevance": "Hydrogen spectral lines: 1/λ = R_∞(1/n₁² - 1/n₂²)",
    },
    "fine_structure_constant": {
        "symbol": "α",
        "value": 7.2973525693e-3,
        "unit": "dimensionless",
        "uncertainty": 1.5e-10,
        "domain": "quantum_electrodynamics",
        "spectral_relevance": "Spectral line splitting: ΔE ∝ α² * E_n",
    },
}


# =============================================================================
# Mathematical Transform Relationships
# =============================================================================

TRANSFORM_RELATIONS: List[Dict[str, Any]] = [
    {
        "name": "fourier_transform_pair",
        "formula": "F(ω) = ∫ f(t) e^{-iωt} dt",
        "inverse": "f(t) = (1/2π) ∫ F(ω) e^{iωt} dω",
        "domain": "signal_processing",
        "properties": [
            "Linearity: F[af + bg] = aF[f] + bF[g]",
            "Time shift: F[f(t-τ)] = e^{-iωτ} F(ω)",
            "Frequency shift: F[e^{iω₀t} f(t)] = F(ω - ω₀)",
            "Convolution: F[f*g] = F[f] · F[g]",
            "Parseval: ∫|f(t)|² dt = (1/2π) ∫|F(ω)|² dω",
        ],
        "spectral_use": "Time→frequency decomposition of signals",
    },
    {
        "name": "laplace_transform",
        "formula": "F(s) = ∫₀^∞ f(t) e^{-st} dt,  s = σ + iω",
        "domain": "control_theory",
        "properties": [
            "Derivative: L[f'] = sF(s) - f(0)",
            "Integration: L[∫f] = F(s)/s",
            "Final value: lim_{t→∞} f(t) = lim_{s→0} sF(s)",
            "Transfer function: H(s) = Y(s)/X(s)",
        ],
        "spectral_use": "System response and pole-zero analysis",
    },
    {
        "name": "wavelet_transform",
        "formula": "W(a,b) = (1/√a) ∫ f(t) ψ*((t-b)/a) dt",
        "domain": "time_frequency_analysis",
        "properties": [
            "Scale a: dilation (frequency resolution)",
            "Translation b: time localization",
            "Admissibility: ∫|Ψ(ω)|²/|ω| dω < ∞",
            "Multi-resolution: coarse + fine simultaneously",
        ],
        "spectral_use": "Non-stationary spectral analysis with time localization",
    },
    {
        "name": "hilbert_transform",
        "formula": "H[f](t) = (1/π) P.V. ∫ f(τ)/(t-τ) dτ",
        "domain": "signal_processing",
        "properties": [
            "Analytic signal: z(t) = f(t) + iH[f](t)",
            "Instantaneous amplitude: A(t) = |z(t)|",
            "Instantaneous frequency: ω(t) = d/dt arg(z(t))",
            "Phase shift: H shifts all frequencies by -π/2",
        ],
        "spectral_use": "Envelope detection and instantaneous frequency",
    },
    {
        "name": "z_transform",
        "formula": "X(z) = Σ x[n] z^{-n}",
        "domain": "discrete_systems",
        "properties": [
            "Convolution: Z[x*h] = X(z)H(z)",
            "Time delay: Z[x[n-k]] = z^{-k} X(z)",
            "Stability: all poles inside unit circle",
            "DFT is Z-transform on unit circle: z = e^{j2πk/N}",
        ],
        "spectral_use": "Discrete spectral analysis and filter design",
    },
]


# =============================================================================
# Wave Physics and Geometry
# =============================================================================

WAVE_PHYSICS: List[Dict[str, Any]] = [
    {
        "name": "standing_wave_resonance",
        "formula": "f_n = n * v / (2L)",
        "description": "Resonant frequencies of a string/pipe of length L",
        "parameters": {"n": "harmonic number", "v": "wave speed", "L": "length"},
        "geometry": "1D bounded",
        "spectral_pattern": "Harmonic series with integer spacing",
    },
    {
        "name": "circular_membrane_modes",
        "formula": "f_{mn} = (α_{mn} / (2πa)) * √(T/σ)",
        "description": "Vibration modes of a circular drum head",
        "parameters": {
            "α_mn": "Bessel function zeros",
            "a": "radius",
            "T": "tension",
            "σ": "surface density",
        },
        "geometry": "2D circular",
        "spectral_pattern": "Non-harmonic — zeros of Bessel functions",
    },
    {
        "name": "rectangular_plate_modes",
        "formula": "f_{mn} = (π/2) * √(D/(ρh)) * ((m/a)² + (n/b)²)",
        "description": "Flexural modes of a rectangular plate",
        "parameters": {
            "D": "flexural rigidity",
            "ρ": "density",
            "h": "thickness",
            "a,b": "dimensions",
        },
        "geometry": "2D rectangular",
        "spectral_pattern": "Sum-of-squares frequency distribution",
    },
    {
        "name": "spherical_harmonics",
        "formula": "Y_l^m(θ,φ) = N * P_l^m(cos θ) * e^{imφ}",
        "description": "Angular modes on a sphere",
        "parameters": {"l": "degree", "m": "order", "N": "normalization"},
        "geometry": "2D spherical surface",
        "spectral_pattern": "Degeneracy 2l+1 per angular momentum l",
    },
    {
        "name": "doppler_shift",
        "formula": "f_obs = f_src * (v + v_obs) / (v + v_src)",
        "description": "Frequency shift from relative motion",
        "parameters": {
            "v": "wave speed",
            "v_obs": "observer velocity",
            "v_src": "source velocity",
        },
        "geometry": "radial",
        "spectral_pattern": "Uniform frequency scaling",
    },
    {
        "name": "diffraction_grating",
        "formula": "d * sin(θ) = m * λ",
        "description": "Spectral decomposition by periodic structure",
        "parameters": {"d": "grating spacing", "m": "order", "θ": "angle"},
        "geometry": "periodic 1D",
        "spectral_pattern": "Angular separation proportional to wavelength",
    },
    {
        "name": "wave_interference",
        "formula": "I = I₁ + I₂ + 2√(I₁I₂) cos(Δφ)",
        "description": "Superposition intensity pattern",
        "parameters": {"I₁,I₂": "individual intensities", "Δφ": "phase difference"},
        "geometry": "superposition",
        "spectral_pattern": "Constructive/destructive bands at Δφ = 2nπ / (2n+1)π",
    },
]


# =============================================================================
# Quantum Spectral Lines (Hydrogen-like)
# =============================================================================

HYDROGEN_SERIES: Dict[str, Dict[str, Any]] = {
    "lyman": {
        "n_lower": 1,
        "transitions": [(1, n) for n in range(2, 8)],
        "wavelength_range_nm": (91.2, 121.6),
        "region": "ultraviolet",
        "formula": "1/λ = R_∞ * (1 - 1/n²)",
    },
    "balmer": {
        "n_lower": 2,
        "transitions": [(2, n) for n in range(3, 9)],
        "wavelength_range_nm": (364.6, 656.3),
        "region": "visible",
        "formula": "1/λ = R_∞ * (1/4 - 1/n²)",
    },
    "paschen": {
        "n_lower": 3,
        "transitions": [(3, n) for n in range(4, 10)],
        "wavelength_range_nm": (820.4, 1875.1),
        "region": "near_infrared",
        "formula": "1/λ = R_∞ * (1/9 - 1/n²)",
    },
    "brackett": {
        "n_lower": 4,
        "transitions": [(4, n) for n in range(5, 11)],
        "wavelength_range_nm": (1458.0, 4051.0),
        "region": "infrared",
        "formula": "1/λ = R_∞ * (1/16 - 1/n²)",
    },
}


# =============================================================================
# Electromagnetic Spectrum Bands
# =============================================================================

EM_SPECTRUM_BANDS: List[Dict[str, Any]] = [
    {"name": "gamma_rays", "freq_min_hz": 3e19, "freq_max_hz": 3e24, "wavelength": "<10 pm", "energy_ev": ">124 keV"},
    {"name": "x_rays", "freq_min_hz": 3e16, "freq_max_hz": 3e19, "wavelength": "10 pm – 10 nm", "energy_ev": "124 eV – 124 keV"},
    {"name": "ultraviolet", "freq_min_hz": 7.5e14, "freq_max_hz": 3e16, "wavelength": "10 nm – 400 nm", "energy_ev": "3.1 – 124 eV"},
    {"name": "visible", "freq_min_hz": 4.3e14, "freq_max_hz": 7.5e14, "wavelength": "400 – 700 nm", "energy_ev": "1.77 – 3.1 eV"},
    {"name": "infrared", "freq_min_hz": 3e11, "freq_max_hz": 4.3e14, "wavelength": "700 nm – 1 mm", "energy_ev": "1.24 meV – 1.77 eV"},
    {"name": "microwave", "freq_min_hz": 3e8, "freq_max_hz": 3e11, "wavelength": "1 mm – 1 m", "energy_ev": "1.24 μeV – 1.24 meV"},
    {"name": "radio", "freq_min_hz": 3.0, "freq_max_hz": 3e8, "wavelength": "1 m – 100 Mm", "energy_ev": "<1.24 μeV"},
]


# =============================================================================
# Geometry: Key Relationships for Spectral Analysis
# =============================================================================

GEOMETRY_RELATIONS: List[Dict[str, Any]] = [
    {
        "name": "nyquist_sampling",
        "formula": "f_s > 2 * f_max",
        "description": "Minimum sampling rate to avoid aliasing",
        "implication": "Spectral content above f_s/2 folds back (aliases)",
    },
    {
        "name": "uncertainty_principle",
        "formula": "Δt * Δf ≥ 1/(4π)",
        "description": "Time-frequency resolution tradeoff",
        "implication": "Cannot have arbitrarily precise time AND frequency simultaneously",
    },
    {
        "name": "parseval_energy",
        "formula": "∫|x(t)|² dt = ∫|X(f)|² df",
        "description": "Total energy is same in time and frequency domains",
        "implication": "PSD integral equals total signal power",
    },
    {
        "name": "convolution_theorem",
        "formula": "F{f * g} = F{f} · F{g}",
        "description": "Convolution in time = multiplication in frequency",
        "implication": "Filtering is multiplication of spectra",
    },
    {
        "name": "euler_formula",
        "formula": "e^{iθ} = cos(θ) + i·sin(θ)",
        "description": "Bridge between exponential and trigonometric",
        "implication": "Complex exponentials are the natural basis for spectral analysis",
    },
    {
        "name": "orthogonality_of_sinusoids",
        "formula": "∫₀^T sin(mωt)sin(nωt)dt = (T/2)δ_{mn}",
        "description": "Sinusoids at different harmonics are orthogonal",
        "implication": "Fourier coefficients are independent projections",
    },
    {
        "name": "golden_ratio_in_spectra",
        "formula": "φ = (1 + √5) / 2 ≈ 1.618",
        "description": "Appears in quasi-periodic spectral structures",
        "implication": "Penrose tilings, quasicrystals have φ-spaced diffraction peaks",
    },
    {
        "name": "fractal_dimension_spectrum",
        "formula": "S(f) ∝ f^{-β},  D = (5 - β) / 2",
        "description": "Power-law spectrum relates to fractal dimension",
        "implication": "β=2 (Brownian) → D=1.5, β=0 (white) → D=2.5",
    },
]


# =============================================================================
# Thermodynamics / Statistical Mechanics for Spectral Systems
# =============================================================================

THERMODYNAMICS_SPECTRAL: List[Dict[str, Any]] = [
    {
        "name": "planck_radiation_law",
        "formula": "B(ν,T) = (2hν³/c²) / (e^{hν/kT} - 1)",
        "description": "Spectral radiance of a black body",
        "peak_law": "Wien's displacement: λ_max * T = 2.898×10⁻³ m·K",
        "total_power": "Stefan-Boltzmann: P = σ T⁴",
    },
    {
        "name": "johnson_nyquist_noise",
        "formula": "V_rms = √(4 k_B T R Δf)",
        "description": "Thermal noise voltage in a resistor",
        "psd": "S_V(f) = 4 k_B T R  [V²/Hz]  (flat/white up to ~THz)",
        "spectral_relevance": "Fundamental noise floor for any measurement",
    },
    {
        "name": "equipartition_theorem",
        "formula": "⟨E_mode⟩ = (1/2) k_B T  per quadratic degree of freedom",
        "description": "Each spectral mode carries same average energy",
        "spectral_relevance": "Explains flat PSD of thermal noise at low frequencies",
    },
    {
        "name": "fluctuation_dissipation",
        "formula": "S_x(ω) = (2 k_B T / ω) Im[χ(ω)]",
        "description": "Links noise spectrum to system response function",
        "spectral_relevance": "Measured PSD reveals system susceptibility",
    },
]


# =============================================================================
# Signal Processing Fundamentals
# =============================================================================

SIGNAL_PROCESSING: List[Dict[str, Any]] = [
    {
        "name": "power_spectral_density",
        "formula": "S_xx(f) = lim_{T→∞} (1/T)|X_T(f)|²",
        "description": "Power per unit frequency of a stationary process",
        "estimators": ["Periodogram", "Welch", "Bartlett", "Multitaper"],
        "units": "V²/Hz or m²/s⁴/Hz (acceleration PSD)",
    },
    {
        "name": "transfer_function",
        "formula": "H(f) = Y(f) / X(f)",
        "description": "Frequency-domain input-output relationship",
        "properties": ["Gain: |H(f)|", "Phase: arg(H(f))", "Group delay: -dφ/dω"],
    },
    {
        "name": "window_functions",
        "formula": "X_w(f) = X(f) * W(f)  (spectral leakage)",
        "description": "Windowing controls sidelobe/mainlobe tradeoff",
        "common_windows": [
            {"name": "Hanning", "sidelobe_db": -31, "mainlobe_width": "4/N"},
            {"name": "Hamming", "sidelobe_db": -42, "mainlobe_width": "4/N"},
            {"name": "Blackman", "sidelobe_db": -58, "mainlobe_width": "6/N"},
            {"name": "Kaiser", "sidelobe_db": "adjustable via β", "mainlobe_width": "variable"},
        ],
    },
    {
        "name": "coherence",
        "formula": "γ²(f) = |S_xy(f)|² / (S_xx(f) * S_yy(f))",
        "description": "Linear correlation between two signals at each frequency",
        "range": "[0, 1] — 1 means perfect linear relationship at frequency f",
    },
    {
        "name": "cepstral_analysis",
        "formula": "c(τ) = F⁻¹{log|F{x(t)}|}",
        "description": "Inverse FT of log spectrum — separates source and filter",
        "applications": ["Speech (formants)", "Seismology (echoes)", "Machine diagnostics"],
    },
]


# =============================================================================
# Vibration & Structural Dynamics
# =============================================================================

VIBRATION_PHYSICS: List[Dict[str, Any]] = [
    {
        "name": "sdof_resonance",
        "formula": "f_n = (1/2π) √(k/m)",
        "description": "Natural frequency of single-degree-of-freedom oscillator",
        "damped_frequency": "f_d = f_n √(1 - ζ²)",
        "half_power_bandwidth": "Δf = 2ζf_n",
    },
    {
        "name": "modal_superposition",
        "formula": "x(t) = Σ φ_r * q_r(t)",
        "description": "Response as sum of modal contributions",
        "spectral_form": "S_x(f) = Σ |φ_r|² * |H_r(f)|² * S_F(f)",
    },
    {
        "name": "response_spectrum",
        "formula": "S_a(T, ζ) = max|ẍ(t) + ẍ_g(t)|  for oscillator period T",
        "description": "Peak response of SDOF oscillators to ground motion",
        "applications": ["Seismic design", "Structural engineering", "MESIE matching"],
    },
    {
        "name": "random_vibration_fatigue",
        "formula": "E[D] = ν₀⁺ T * ∫ p(S) / N(S) dS",
        "description": "Expected fatigue damage from PSD via Dirlik/Narrow-band",
        "spectral_moments": "m_n = ∫ f^n * S(f) df",
    },
]


# =============================================================================
# Dataset Generator Functions
# =============================================================================


@dataclass
class TrainingExample:
    """One training example for reasoning/embedding calibration.

    Attributes:
        domain: Physics domain this belongs to.
        concept: Specific concept being trained.
        input_data: Input spectral/numeric data.
        expected_output: Expected reasoning output.
        difficulty: 1-5 scale.
        metadata: Additional context.
    """

    domain: str
    concept: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    difficulty: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_harmonic_series_dataset(n_examples: int = 100) -> List[TrainingExample]:
    """Generate training examples for harmonic series recognition.

    Creates synthetic spectra with known harmonic structure and
    trains the system to identify fundamental frequency, harmonic
    count, and decay pattern.
    """
    rng = np.random.default_rng(42)
    examples: List[TrainingExample] = []

    for i in range(n_examples):
        f0 = rng.uniform(10.0, 500.0)  # Fundamental frequency
        n_harmonics = rng.integers(3, 15)
        decay_rate = rng.uniform(0.3, 0.9)  # Amplitude decay per harmonic

        freqs = np.array([f0 * (n + 1) for n in range(n_harmonics)])
        amps = np.array([decay_rate**n for n in range(n_harmonics)])
        # Add noise
        amps += rng.normal(0, 0.02, n_harmonics)
        amps = np.clip(amps, 0, None)

        examples.append(TrainingExample(
            domain="acoustics",
            concept="harmonic_series",
            input_data={
                "frequency": freqs.tolist(),
                "amplitude": amps.tolist(),
            },
            expected_output={
                "fundamental_hz": f0,
                "n_harmonics": n_harmonics,
                "decay_rate": decay_rate,
                "pattern": "exponential_decay",
            },
            difficulty=min(5, 1 + n_harmonics // 4),
            metadata={"generator": "harmonic_series", "index": i},
        ))

    return examples


def generate_resonance_dataset(n_examples: int = 100) -> List[TrainingExample]:
    """Generate training examples for resonance identification.

    Creates spectra with peaks at resonant frequencies and trains
    detection of natural frequencies, damping ratios, and mode shapes.
    """
    rng = np.random.default_rng(123)
    examples: List[TrainingExample] = []

    for i in range(n_examples):
        n_modes = rng.integers(1, 6)
        f_n = np.sort(rng.uniform(1.0, 100.0, n_modes))
        zeta = rng.uniform(0.01, 0.15, n_modes)  # Damping ratios

        # Generate PSD with resonance peaks
        f = np.linspace(0.1, 200.0, 512)
        psd = np.ones_like(f) * 1e-6  # Noise floor
        for fn, z in zip(f_n, zeta):
            # SDOF transfer function magnitude squared
            r = f / fn
            H2 = 1.0 / ((1 - r**2)**2 + (2 * z * r)**2)
            psd += H2 * 1e-4

        examples.append(TrainingExample(
            domain="structural_dynamics",
            concept="modal_resonance",
            input_data={
                "frequency": f.tolist(),
                "amplitude": psd.tolist(),
            },
            expected_output={
                "natural_frequencies_hz": f_n.tolist(),
                "damping_ratios": zeta.tolist(),
                "n_modes": n_modes,
            },
            difficulty=min(5, n_modes + 1),
            metadata={"generator": "resonance", "index": i},
        ))

    return examples


def generate_transform_pair_dataset(n_examples: int = 100) -> List[TrainingExample]:
    """Generate Fourier transform pair training examples.

    Teaches the relationship between time-domain signals and their
    frequency-domain representations.
    """
    rng = np.random.default_rng(456)
    examples: List[TrainingExample] = []

    signal_types = ["gaussian_pulse", "exponential_decay", "rect_pulse", "sinc", "chirp"]

    for i in range(n_examples):
        sig_type = signal_types[i % len(signal_types)]
        t = np.linspace(0, 1, 256)
        f = np.fft.rfftfreq(256, d=t[1] - t[0])

        if sig_type == "gaussian_pulse":
            sigma = rng.uniform(0.01, 0.1)
            t0 = 0.5
            x = np.exp(-((t - t0)**2) / (2 * sigma**2))
            # FT of Gaussian is Gaussian: width ∝ 1/sigma
            bandwidth = 1.0 / (2 * np.pi * sigma)
        elif sig_type == "exponential_decay":
            alpha = rng.uniform(5.0, 50.0)
            x = np.exp(-alpha * t) * (t >= 0)
            bandwidth = alpha / (2 * np.pi)
        elif sig_type == "rect_pulse":
            width = rng.uniform(0.05, 0.3)
            x = ((t >= 0.5 - width / 2) & (t <= 0.5 + width / 2)).astype(float)
            bandwidth = 1.0 / width
        elif sig_type == "sinc":
            bw = rng.uniform(10.0, 50.0)
            x = np.sinc(bw * (t - 0.5))
            bandwidth = bw
        else:  # chirp
            f0 = rng.uniform(1.0, 10.0)
            f1 = rng.uniform(50.0, 100.0)
            x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))
            bandwidth = f1

        X = np.abs(np.fft.rfft(x))
        X /= X.max() if X.max() > 0 else 1.0

        examples.append(TrainingExample(
            domain="mathematics",
            concept="fourier_transform_pair",
            input_data={
                "time_signal": x.tolist(),
                "frequency": f.tolist(),
                "spectrum_magnitude": X.tolist(),
            },
            expected_output={
                "signal_type": sig_type,
                "bandwidth_hz": float(bandwidth),
                "relationship": f"Time width ∝ 1/bandwidth for {sig_type}",
            },
            difficulty=2 + (i % 3),
            metadata={"generator": "transform_pair", "index": i},
        ))

    return examples


def generate_wave_geometry_dataset(n_examples: int = 80) -> List[TrainingExample]:
    """Generate wave propagation geometry training examples.

    Teaches spatial-spectral relationships: how geometry determines
    resonant frequencies, mode shapes, and spectral patterns.
    """
    rng = np.random.default_rng(789)
    examples: List[TrainingExample] = []

    for i in range(n_examples):
        # Rectangular cavity modes
        Lx = rng.uniform(1.0, 10.0)
        Ly = rng.uniform(1.0, 10.0)
        Lz = rng.uniform(1.0, 10.0)
        c = 343.0  # Speed of sound in air

        modes: List[Tuple[int, int, int]] = []
        mode_freqs: List[float] = []
        for nx in range(1, 5):
            for ny in range(1, 5):
                for nz in range(1, 5):
                    f_mode = (c / 2) * np.sqrt((nx / Lx)**2 + (ny / Ly)**2 + (nz / Lz)**2)
                    modes.append((nx, ny, nz))
                    mode_freqs.append(float(f_mode))

        # Sort by frequency
        sorted_idx = np.argsort(mode_freqs)
        mode_freqs = [mode_freqs[j] for j in sorted_idx[:10]]
        modes = [modes[j] for j in sorted_idx[:10]]

        examples.append(TrainingExample(
            domain="geometry",
            concept="cavity_modes",
            input_data={
                "dimensions_m": [Lx, Ly, Lz],
                "speed_of_sound": c,
                "resonant_frequencies_hz": mode_freqs,
            },
            expected_output={
                "mode_indices": modes,
                "fundamental_hz": mode_freqs[0],
                "n_modes_below_500hz": sum(1 for f in mode_freqs if f < 500),
                "geometry_type": "rectangular_cavity",
            },
            difficulty=3,
            metadata={"generator": "wave_geometry", "index": i},
        ))

    return examples


def generate_quantum_lines_dataset(n_examples: int = 50) -> List[TrainingExample]:
    """Generate hydrogen spectral line training examples.

    Real physics: compute actual wavelengths from the Rydberg formula
    and train identification of series and transitions.
    """
    R_inf = 10_973_731.568160  # Rydberg constant, 1/m
    examples: List[TrainingExample] = []

    series_names = list(HYDROGEN_SERIES.keys())
    rng = np.random.default_rng(321)

    for i in range(n_examples):
        series_name = series_names[i % len(series_names)]
        series = HYDROGEN_SERIES[series_name]
        n_lower = series["n_lower"]

        # Compute actual wavelengths
        wavelengths_nm: List[float] = []
        transitions: List[Tuple[int, int]] = []
        for n1, n2 in series["transitions"]:
            inv_lambda = R_inf * (1.0 / n1**2 - 1.0 / n2**2)
            if inv_lambda > 0:
                lam_m = 1.0 / inv_lambda
                wavelengths_nm.append(lam_m * 1e9)
                transitions.append((n1, n2))

        # Add measurement noise
        noise = rng.normal(0, 0.01, len(wavelengths_nm))
        measured = [w + n for w, n in zip(wavelengths_nm, noise)]

        examples.append(TrainingExample(
            domain="quantum_mechanics",
            concept="hydrogen_spectral_lines",
            input_data={
                "measured_wavelengths_nm": measured,
                "series_hint": series["region"],
            },
            expected_output={
                "series_name": series_name,
                "n_lower": n_lower,
                "transitions": transitions,
                "exact_wavelengths_nm": wavelengths_nm,
            },
            difficulty=2,
            metadata={"generator": "quantum_lines", "index": i},
        ))

    return examples


def generate_power_law_dataset(n_examples: int = 80) -> List[TrainingExample]:
    """Generate power-law (1/f^β) spectral training examples.

    Teaches identification of spectral slope, fractal dimension,
    and physical process type from PSD shape.
    """
    rng = np.random.default_rng(654)
    examples: List[TrainingExample] = []

    process_types = {
        0.0: ("white_noise", "Random, uncorrelated"),
        1.0: ("pink_noise", "1/f flicker noise, ubiquitous in nature"),
        1.5: ("fractional_brownian_H075", "Persistent random walk"),
        2.0: ("brownian_motion", "Random walk, integrated white noise"),
        3.0: ("black_noise", "Highly correlated, natural disasters"),
    }

    for i in range(n_examples):
        beta = rng.uniform(0.0, 3.5)
        # Find closest known process
        closest_beta = min(process_types.keys(), key=lambda b: abs(b - beta))
        process_name, description = process_types[closest_beta]

        f = np.logspace(-2, 2, 256)
        psd = f**(-beta)
        # Add noise
        psd *= rng.lognormal(0, 0.1, len(psd))

        fractal_dim = (5 - beta) / 2  # For 1D signal embedded in 2D

        examples.append(TrainingExample(
            domain="physics",
            concept="power_law_spectrum",
            input_data={
                "frequency": f.tolist(),
                "amplitude": psd.tolist(),
                "log_frequency": np.log10(f).tolist(),
                "log_psd": np.log10(psd).tolist(),
            },
            expected_output={
                "spectral_exponent_beta": float(beta),
                "fractal_dimension": float(fractal_dim),
                "process_type": process_name,
                "description": description,
            },
            difficulty=2 + int(abs(beta - closest_beta) > 0.3),
            metadata={"generator": "power_law", "index": i},
        ))

    return examples


# =============================================================================
# Master Dataset Builder
# =============================================================================


@dataclass
class ReasoningDatasetManifest:
    """Manifest describing all available reasoning training datasets."""

    datasets: Dict[str, List[TrainingExample]] = field(default_factory=dict)
    total_examples: int = 0
    domains_covered: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "n_datasets": len(self.datasets),
            "domains": self.domains_covered,
            "datasets": {k: len(v) for k, v in self.datasets.items()},
        }


def build_reasoning_datasets(
    *,
    n_harmonic: int = 100,
    n_resonance: int = 100,
    n_transform: int = 100,
    n_geometry: int = 80,
    n_quantum: int = 50,
    n_power_law: int = 80,
) -> ReasoningDatasetManifest:
    """Build all reasoning training datasets.

    Returns a manifest with real physics/math/geometry training data
    suitable for embedding calibration and reasoning training.

    Args:
        n_harmonic: Number of harmonic series examples.
        n_resonance: Number of resonance examples.
        n_transform: Number of Fourier transform pair examples.
        n_geometry: Number of wave geometry examples.
        n_quantum: Number of quantum spectral line examples.
        n_power_law: Number of power-law spectrum examples.

    Returns:
        ReasoningDatasetManifest with all datasets.
    """
    datasets = {
        "harmonic_series": generate_harmonic_series_dataset(n_harmonic),
        "modal_resonance": generate_resonance_dataset(n_resonance),
        "fourier_transform_pairs": generate_transform_pair_dataset(n_transform),
        "wave_geometry": generate_wave_geometry_dataset(n_geometry),
        "quantum_spectral_lines": generate_quantum_lines_dataset(n_quantum),
        "power_law_spectra": generate_power_law_dataset(n_power_law),
    }

    total = sum(len(v) for v in datasets.values())
    domains = sorted(set(ex.domain for ds in datasets.values() for ex in ds))

    return ReasoningDatasetManifest(
        datasets=datasets,
        total_examples=total,
        domains_covered=domains,
    )
