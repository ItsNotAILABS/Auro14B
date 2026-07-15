"""Physics Lab — simulation wrappers, unit conversions, constants, and mechanics."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from mesie.labs.base_lab import BaseLab, LabConfig, LabResult


# Fundamental physical constants (CODATA 2018)
CONSTANTS: Dict[str, Dict[str, Any]] = {
    "c": {"value": 299792458.0, "unit": "m/s", "name": "Speed of light"},
    "h": {"value": 6.62607015e-34, "unit": "J·s", "name": "Planck constant"},
    "hbar": {"value": 1.054571817e-34, "unit": "J·s", "name": "Reduced Planck constant"},
    "k_B": {"value": 1.380649e-23, "unit": "J/K", "name": "Boltzmann constant"},
    "N_A": {"value": 6.02214076e23, "unit": "1/mol", "name": "Avogadro number"},
    "e": {"value": 1.602176634e-19, "unit": "C", "name": "Elementary charge"},
    "G": {"value": 6.67430e-11, "unit": "m³/(kg·s²)", "name": "Gravitational constant"},
    "epsilon_0": {"value": 8.8541878128e-12, "unit": "F/m", "name": "Vacuum permittivity"},
    "mu_0": {"value": 1.25663706212e-6, "unit": "N/A²", "name": "Vacuum permeability"},
    "m_e": {"value": 9.1093837015e-31, "unit": "kg", "name": "Electron mass"},
    "m_p": {"value": 1.67262192369e-27, "unit": "kg", "name": "Proton mass"},
    "R": {"value": 8.314462618, "unit": "J/(mol·K)", "name": "Gas constant"},
    "sigma": {"value": 5.670374419e-8, "unit": "W/(m²·K⁴)", "name": "Stefan-Boltzmann constant"},
}

# Unit conversion factors
UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
    "length": {"m_to_ft": 3.28084, "m_to_in": 39.3701, "km_to_mi": 0.621371, "m_to_nm": 1e9},
    "mass": {"kg_to_lb": 2.20462, "kg_to_g": 1000.0, "kg_to_amu": 6.022e26},
    "energy": {"J_to_eV": 6.242e18, "J_to_cal": 0.239006, "J_to_kWh": 2.778e-7},
    "temperature": {"K_to_C_offset": -273.15, "C_to_F_scale": 1.8, "C_to_F_offset": 32.0},
    "pressure": {"Pa_to_atm": 9.8692e-6, "Pa_to_bar": 1e-5, "Pa_to_psi": 0.000145038},
}


class PhysicsLab(BaseLab):
    """Lab for physics calculations, simulations, and unit conversions.

    Provides physical constants, unit conversions, classical mechanics
    calculations, wave physics, and thermodynamics.
    """

    def _default_config(self) -> LabConfig:
        return LabConfig(
            name="Physics Lab",
            domain="physics",
            capabilities=[
                "constants",
                "unit_convert",
                "kinematics",
                "wave_physics",
                "thermodynamics",
                "oscillator",
                "blackbody",
            ],
        )

    def run(self, operation: str, **kwargs: Any) -> LabResult:
        start = time.time()
        try:
            if operation == "constants":
                data = self._constants(**kwargs)
            elif operation == "unit_convert":
                data = self._unit_convert(**kwargs)
            elif operation == "kinematics":
                data = self._kinematics(**kwargs)
            elif operation == "wave_physics":
                data = self._wave_physics(**kwargs)
            elif operation == "thermodynamics":
                data = self._thermodynamics(**kwargs)
            elif operation == "oscillator":
                data = self._oscillator(**kwargs)
            elif operation == "blackbody":
                data = self._blackbody(**kwargs)
            else:
                return LabResult(
                    lab=self.name, operation=operation,
                    status="error", error=f"Unknown operation: {operation}",
                )
            return LabResult(
                lab=self.name, operation=operation, data=data,
                duration_seconds=time.time() - start,
            )
        except Exception as exc:
            return LabResult(
                lab=self.name, operation=operation,
                status="error", error=str(exc),
                duration_seconds=time.time() - start,
            )

    def _constants(self, name: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
        """Look up physical constants."""
        if name:
            entry = CONSTANTS.get(name)
            if entry:
                return {"name": name, **entry}
            return {"error": f"Unknown constant: {name}"}
        return {"constants": CONSTANTS}

    def _unit_convert(
        self, value: float = 0.0, from_unit: str = "", to_unit: str = "", **kw: Any
    ) -> Dict[str, Any]:
        """Convert between physical units."""
        key = f"{from_unit}_to_{to_unit}"
        for category, conversions in UNIT_CONVERSIONS.items():
            if key in conversions:
                factor = conversions[key]
                if "offset" in key:
                    result = value + factor
                else:
                    result = value * factor
                return {
                    "input": value,
                    "from": from_unit,
                    "to": to_unit,
                    "result": result,
                    "category": category,
                }
        return {"error": f"Conversion {from_unit} → {to_unit} not found"}

    def _kinematics(
        self,
        v0: float = 0.0,
        a: float = 0.0,
        t: float = 0.0,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Classical kinematics: compute position, velocity, and displacement."""
        v = v0 + a * t
        s = v0 * t + 0.5 * a * t ** 2
        return {
            "initial_velocity": v0,
            "acceleration": a,
            "time": t,
            "final_velocity": v,
            "displacement": s,
        }

    def _wave_physics(
        self, frequency: float = 0.0, wavelength: float = 0.0, medium_speed: float = 343.0, **kw: Any
    ) -> Dict[str, Any]:
        """Wave physics: relate frequency, wavelength, and speed."""
        if frequency > 0 and wavelength == 0:
            wavelength = medium_speed / frequency
        elif wavelength > 0 and frequency == 0:
            frequency = medium_speed / wavelength
        period = 1.0 / frequency if frequency > 0 else 0.0
        energy = CONSTANTS["h"]["value"] * frequency
        return {
            "frequency_hz": frequency,
            "wavelength_m": wavelength,
            "speed_m_s": medium_speed,
            "period_s": period,
            "photon_energy_J": energy,
        }

    def _thermodynamics(
        self,
        T: float = 300.0,
        n_moles: float = 1.0,
        volume: float = 0.0224,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Ideal gas law and thermodynamic quantities."""
        R = CONSTANTS["R"]["value"]
        P = n_moles * R * T / volume if volume > 0 else 0.0
        internal_energy = 1.5 * n_moles * R * T  # Monatomic ideal gas
        entropy_approx = n_moles * R * math.log(T / 273.15) if T > 0 else 0.0
        return {
            "temperature_K": T,
            "n_moles": n_moles,
            "volume_m3": volume,
            "pressure_Pa": P,
            "internal_energy_J": internal_energy,
            "entropy_approx_J_K": entropy_approx,
        }

    def _oscillator(
        self,
        mass: float = 1.0,
        spring_constant: float = 1.0,
        amplitude: float = 1.0,
        damping: float = 0.0,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Simple/damped harmonic oscillator properties."""
        omega_0 = math.sqrt(spring_constant / mass) if mass > 0 else 0.0
        freq = omega_0 / (2 * math.pi)
        period = 1.0 / freq if freq > 0 else 0.0
        gamma = damping / (2 * mass) if mass > 0 else 0.0
        q_factor = omega_0 / (2 * gamma) if gamma > 0 else float("inf")
        max_energy = 0.5 * spring_constant * amplitude ** 2
        return {
            "natural_frequency_hz": freq,
            "angular_frequency_rad_s": omega_0,
            "period_s": period,
            "damping_ratio": gamma / omega_0 if omega_0 > 0 else 0.0,
            "quality_factor": q_factor,
            "max_energy_J": max_energy,
        }

    def _blackbody(self, temperature: float = 5778.0, **kw: Any) -> Dict[str, Any]:
        """Blackbody radiation: peak wavelength and total power."""
        sigma = CONSTANTS["sigma"]["value"]
        # Wien's displacement law
        b = 2.897771955e-3  # Wien displacement constant (m·K)
        peak_wavelength = b / temperature if temperature > 0 else 0.0
        # Stefan-Boltzmann total radiance
        total_power_per_area = sigma * temperature ** 4
        return {
            "temperature_K": temperature,
            "peak_wavelength_m": peak_wavelength,
            "peak_wavelength_nm": peak_wavelength * 1e9,
            "total_power_W_m2": total_power_per_area,
        }
