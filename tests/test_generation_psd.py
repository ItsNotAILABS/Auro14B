"""Tests for PSD generation."""

import numpy as np
import pytest

from mesie.core.config import GenerationConfig
from mesie.generation.psd import generate_psd
from mesie.generation.fas import generate_fas
from mesie.generation.rotdnn import generate_rotdnn
from mesie.generation.single_component import generate_single


class TestPSDGeneration:
    def test_generate_psd_basic(self):
        cfg = GenerationConfig(seed=3)
        rec = generate_psd(cfg)
        assert rec.representation == "psd"
        assert rec.components[0].units == "psd"
        assert np.all(rec.components[0].amplitude >= 0.0)

    def test_generate_psd_with_shape(self):
        cfg = GenerationConfig(seed=5, amplitude_shape="power_law")
        rec = generate_psd(cfg)
        assert len(rec.components[0].frequency) > 0
        assert rec.representation == "psd"

    def test_generate_psd_reproducible(self):
        cfg = GenerationConfig(seed=42)
        rec1 = generate_psd(cfg)
        rec2 = generate_psd(cfg)
        np.testing.assert_array_equal(rec1.components[0].amplitude, rec2.components[0].amplitude)

    def test_generate_psd_custom_grid(self):
        grid = np.linspace(0.1, 50.0, 100)
        cfg = GenerationConfig(seed=1, target_frequency=grid)
        rec = generate_psd(cfg)
        assert len(rec.components[0].frequency) == 100


class TestFASGeneration:
    def test_generate_fas_basic(self):
        cfg = GenerationConfig(seed=7)
        rec = generate_fas(cfg)
        assert rec.representation == "fas"
        assert rec.components[0].units == "fas"
        assert np.all(rec.components[0].amplitude >= 0.0)

    def test_generate_fas_gaussian(self):
        cfg = GenerationConfig(seed=2, amplitude_shape="gaussian")
        rec = generate_fas(cfg)
        assert len(rec.components) == 1


class TestRotDnnGeneration:
    def test_generate_rotdnn_basic(self):
        cfg = GenerationConfig(seed=10)
        rec = generate_rotdnn(cfg)
        assert rec.representation == "rotdnn"
        assert len(rec.components) >= 2

    def test_generate_rotdnn_custom_blend(self):
        cfg = GenerationConfig(
            seed=11,
            multi_element_blending={"RotD0": 0.3, "RotD50": 0.5, "RotD100": 0.2},
        )
        rec = generate_rotdnn(cfg)
        assert len(rec.components) == 3
        names = {c.name for c in rec.components}
        assert "RotD50" in names


class TestSingleGeneration:
    def test_generate_single_flat(self):
        cfg = GenerationConfig(seed=1, amplitude_shape="flat")
        rec = generate_single(cfg)
        assert rec.representation == "single"

    def test_generate_single_broadband(self):
        cfg = GenerationConfig(seed=2, amplitude_shape="broadband")
        rec = generate_single(cfg)
        assert len(rec.components[0].amplitude) > 0
