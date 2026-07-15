"""Earth Lab — seismic analysis, geospatial utilities, and climate data tools."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesie.labs.base_lab import BaseLab, LabConfig, LabResult


class EarthLab(BaseLab):
    """Lab for earth science: seismology, geospatial analysis, and climate.

    Provides seismic magnitude conversions, distance calculations,
    coordinate transforms, and basic climate/atmosphere models.
    """

    def _default_config(self) -> LabConfig:
        return LabConfig(
            name="Earth Science Lab",
            domain="earth",
            capabilities=[
                "haversine_distance",
                "coordinate_convert",
                "seismic_magnitude",
                "atmosphere_model",
                "soil_classification",
                "earthquake_energy",
                "plate_velocity",
            ],
        )

    def run(self, operation: str, **kwargs: Any) -> LabResult:
        start = time.time()
        try:
            if operation == "haversine_distance":
                data = self._haversine_distance(**kwargs)
            elif operation == "coordinate_convert":
                data = self._coordinate_convert(**kwargs)
            elif operation == "seismic_magnitude":
                data = self._seismic_magnitude(**kwargs)
            elif operation == "atmosphere_model":
                data = self._atmosphere_model(**kwargs)
            elif operation == "soil_classification":
                data = self._soil_classification(**kwargs)
            elif operation == "earthquake_energy":
                data = self._earthquake_energy(**kwargs)
            elif operation == "plate_velocity":
                data = self._plate_velocity(**kwargs)
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

    def _haversine_distance(
        self,
        lat1: float = 0.0, lon1: float = 0.0,
        lat2: float = 0.0, lon2: float = 0.0,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Compute great-circle distance between two coordinates."""
        R = 6371.0  # Earth radius in km
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = R * c
        return {
            "point_a": {"lat": lat1, "lon": lon1},
            "point_b": {"lat": lat2, "lon": lon2},
            "distance_km": round(distance_km, 3),
            "distance_mi": round(distance_km * 0.621371, 3),
        }

    def _coordinate_convert(
        self, lat: float = 0.0, lon: float = 0.0, to_format: str = "utm", **kw: Any
    ) -> Dict[str, Any]:
        """Convert geographic coordinates (simplified UTM approximation)."""
        if to_format == "utm":
            zone = int((lon + 180) / 6) + 1
            # Simplified UTM (not full projection)
            easting = 500000 + (lon - (zone * 6 - 183)) * 111320 * math.cos(math.radians(lat))
            northing = lat * 110540
            return {
                "lat": lat, "lon": lon,
                "utm_zone": zone,
                "easting_m": round(easting, 1),
                "northing_m": round(northing, 1),
            }
        elif to_format == "radians":
            return {
                "lat_deg": lat, "lon_deg": lon,
                "lat_rad": math.radians(lat),
                "lon_rad": math.radians(lon),
            }
        return {"error": f"Unsupported format: {to_format}"}

    def _seismic_magnitude(
        self, magnitude: float = 0.0, scale: str = "Mw", **kw: Any
    ) -> Dict[str, Any]:
        """Seismic magnitude classification and energy estimate."""
        # Gutenberg-Richter energy-magnitude relation
        log_energy = 1.5 * magnitude + 4.8  # in Joules (log10)
        energy_joules = 10 ** log_energy
        # Classification
        if magnitude < 2.0:
            classification = "Micro"
        elif magnitude < 4.0:
            classification = "Minor"
        elif magnitude < 5.0:
            classification = "Light"
        elif magnitude < 6.0:
            classification = "Moderate"
        elif magnitude < 7.0:
            classification = "Strong"
        elif magnitude < 8.0:
            classification = "Major"
        else:
            classification = "Great"
        return {
            "magnitude": magnitude,
            "scale": scale,
            "classification": classification,
            "energy_joules": energy_joules,
            "energy_tnt_tons": energy_joules / 4.184e9,
        }

    def _atmosphere_model(
        self, altitude_m: float = 0.0, **kw: Any
    ) -> Dict[str, Any]:
        """US Standard Atmosphere 1976 (troposphere approximation)."""
        T0 = 288.15  # Sea level temperature K
        P0 = 101325.0  # Sea level pressure Pa
        L = 0.0065  # Lapse rate K/m
        g = 9.80665
        M = 0.0289644  # Molar mass of air
        R = 8.31447

        if altitude_m <= 11000:
            T = T0 - L * altitude_m
            P = P0 * (T / T0) ** (g * M / (R * L))
        else:
            # Simplified stratosphere (constant T)
            T = 216.65
            P11 = P0 * (216.65 / T0) ** (g * M / (R * L))
            P = P11 * math.exp(-g * M * (altitude_m - 11000) / (R * T))

        rho = P * M / (R * T)
        return {
            "altitude_m": altitude_m,
            "temperature_K": round(T, 2),
            "temperature_C": round(T - 273.15, 2),
            "pressure_Pa": round(P, 2),
            "density_kg_m3": round(rho, 4),
        }

    def _soil_classification(
        self, vs30: float = 0.0, **kw: Any
    ) -> Dict[str, Any]:
        """NEHRP site classification based on Vs30."""
        if vs30 >= 1500:
            site_class, description = "A", "Hard rock"
        elif vs30 >= 760:
            site_class, description = "B", "Rock"
        elif vs30 >= 360:
            site_class, description = "C", "Very dense soil / soft rock"
        elif vs30 >= 180:
            site_class, description = "D", "Stiff soil"
        else:
            site_class, description = "E", "Soft soil"
        return {
            "vs30_m_s": vs30,
            "site_class": site_class,
            "description": description,
            "standard": "NEHRP",
        }

    def _earthquake_energy(self, magnitude: float = 0.0, **kw: Any) -> Dict[str, Any]:
        """Compute seismic energy and equivalent comparisons."""
        log_energy = 1.5 * magnitude + 4.8
        energy_j = 10 ** log_energy
        hiroshima_equiv = energy_j / 6.3e13  # ~15 kt TNT
        return {
            "magnitude": magnitude,
            "energy_joules": energy_j,
            "energy_tnt_kg": energy_j / 4.184e6,
            "hiroshima_equivalents": round(hiroshima_equiv, 4),
        }

    def _plate_velocity(
        self, plate: str = "", **kw: Any
    ) -> Dict[str, Any]:
        """Approximate tectonic plate velocities (mm/yr)."""
        plates = {
            "pacific": {"velocity_mm_yr": 75, "direction": "NW"},
            "north_american": {"velocity_mm_yr": 25, "direction": "W"},
            "eurasian": {"velocity_mm_yr": 21, "direction": "E"},
            "african": {"velocity_mm_yr": 21, "direction": "NE"},
            "antarctic": {"velocity_mm_yr": 10, "direction": "N"},
            "indo_australian": {"velocity_mm_yr": 65, "direction": "NE"},
            "south_american": {"velocity_mm_yr": 30, "direction": "W"},
            "nazca": {"velocity_mm_yr": 76, "direction": "E"},
        }
        if plate:
            entry = plates.get(plate.lower().replace(" ", "_"))
            if entry:
                return {"plate": plate, **entry}
            return {"error": f"Unknown plate: {plate}", "available": list(plates.keys())}
        return {"plates": plates}
