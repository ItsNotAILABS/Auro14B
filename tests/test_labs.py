"""Tests for the labs module: all domain-specific labs."""

import pytest
import numpy as np

from mesie.labs.base_lab import BaseLab, LabConfig, LabRegistry, LabResult, build_default_lab_registry
from mesie.labs.spectral_lab import SpectralLab
from mesie.labs.chemistry_lab import ChemistryLab
from mesie.labs.physics_lab import PhysicsLab
from mesie.labs.bio_lab import BioLab
from mesie.labs.earth_lab import EarthLab


class TestLabRegistry:
    def test_build_default_registry(self):
        registry = build_default_lab_registry()
        assert "spectral" in registry.domains
        assert "chemistry" in registry.domains
        assert "physics" in registry.domains
        assert "bio" in registry.domains
        assert "earth" in registry.domains

    def test_list_labs(self):
        registry = build_default_lab_registry()
        labs = registry.list_labs()
        assert len(labs) == 5
        assert all("capabilities" in lab for lab in labs)


class TestChemistryLab:
    def test_molecular_fingerprint(self):
        lab = ChemistryLab()
        result = lab.run("molecular_fingerprint", smiles="CCO", bits=512)
        assert result.status == "success"
        assert result.data["bits"] == 512
        assert 0 < result.data["density"] < 1

    def test_similarity_search(self):
        lab = ChemistryLab()
        result = lab.run("similarity_search", query_smiles="CCO")
        assert result.status == "success"
        assert len(result.data["results"]) > 0

    def test_property_predict(self):
        lab = ChemistryLab()
        result = lab.run("property_predict", smiles="CCO")
        assert result.status == "success"
        assert "molecular_weight" in result.data["predictions"]

    def test_formula_parse(self):
        lab = ChemistryLab()
        result = lab.run("formula_parse", formula="H2O")
        assert result.status == "success"
        assert result.data["elements"]["H"] == 2
        assert result.data["elements"]["O"] == 1

    def test_compound_lookup(self):
        lab = ChemistryLab()
        result = lab.run("compound_lookup", name="ethanol")
        assert result.status == "success"
        assert result.data["formula"] == "C2H5OH"

    def test_unknown_operation(self):
        lab = ChemistryLab()
        result = lab.run("nonexistent")
        assert result.status == "error"


class TestPhysicsLab:
    def test_constants_lookup(self):
        lab = PhysicsLab()
        result = lab.run("constants", name="c")
        assert result.status == "success"
        assert result.data["value"] == 299792458.0

    def test_kinematics(self):
        lab = PhysicsLab()
        result = lab.run("kinematics", v0=0.0, a=9.8, t=2.0)
        assert result.status == "success"
        assert abs(result.data["final_velocity"] - 19.6) < 0.01
        assert abs(result.data["displacement"] - 19.6) < 0.01

    def test_wave_physics(self):
        lab = PhysicsLab()
        result = lab.run("wave_physics", frequency=440.0)
        assert result.status == "success"
        assert result.data["wavelength_m"] > 0

    def test_blackbody(self):
        lab = PhysicsLab()
        result = lab.run("blackbody", temperature=5778.0)
        assert result.status == "success"
        # Sun's peak should be ~502nm
        assert 400 < result.data["peak_wavelength_nm"] < 600

    def test_oscillator(self):
        lab = PhysicsLab()
        result = lab.run("oscillator", mass=1.0, spring_constant=4.0)
        assert result.status == "success"
        # omega_0 = sqrt(4/1) = 2, freq = 2/(2pi) ≈ 0.318
        assert abs(result.data["angular_frequency_rad_s"] - 2.0) < 0.001


class TestBioLab:
    def test_gc_content(self):
        lab = BioLab()
        result = lab.run("gc_content", sequence="ATCGGCTA")
        assert result.status == "success"
        assert result.data["gc_count"] == 4
        assert result.data["gc_content"] == 0.5

    def test_translate(self):
        lab = BioLab()
        result = lab.run("translate", rna_sequence="AUGGCUUAA")
        assert result.status == "success"
        assert result.data["protein"] == "MA"

    def test_complement(self):
        lab = BioLab()
        result = lab.run("complement", sequence="ATCG", reverse=True)
        assert result.status == "success"
        assert result.data["complement"] == "CGAT"

    def test_kmer_frequency(self):
        lab = BioLab()
        result = lab.run("kmer_frequency", sequence="ATCGATCG", k=2)
        assert result.status == "success"
        assert result.data["total_kmers"] == 7

    def test_alignment_score(self):
        lab = BioLab()
        result = lab.run("alignment_score", seq_a="ATCG", seq_b="ATCG")
        assert result.status == "success"
        assert result.data["identity"] == 1.0


class TestEarthLab:
    def test_haversine_distance(self):
        lab = EarthLab()
        # NYC to London approx 5570 km
        result = lab.run("haversine_distance", lat1=40.7, lon1=-74.0, lat2=51.5, lon2=-0.1)
        assert result.status == "success"
        assert 5500 < result.data["distance_km"] < 5700

    def test_seismic_magnitude(self):
        lab = EarthLab()
        result = lab.run("seismic_magnitude", magnitude=6.5)
        assert result.status == "success"
        assert result.data["classification"] == "Strong"

    def test_atmosphere_model(self):
        lab = EarthLab()
        result = lab.run("atmosphere_model", altitude_m=0.0)
        assert result.status == "success"
        assert abs(result.data["temperature_K"] - 288.15) < 0.1
        assert abs(result.data["pressure_Pa"] - 101325.0) < 1.0

    def test_soil_classification(self):
        lab = EarthLab()
        result = lab.run("soil_classification", vs30=400.0)
        assert result.status == "success"
        assert result.data["site_class"] == "C"

    def test_plate_velocity(self):
        lab = EarthLab()
        result = lab.run("plate_velocity", plate="pacific")
        assert result.status == "success"
        assert result.data["velocity_mm_yr"] == 75
