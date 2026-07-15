"""Tests for core record data structures."""

import numpy as np
import pytest

from mesie.core.records import MultiElementRecord, SpectralComponent, SpectralMetadata
from mesie.core.components import FrequencyGrid, RecordLineage
from mesie.core.config import GenerationConfig


class TestSpectralComponent:
    def test_create_basic_component(self):
        comp = SpectralComponent(
            name="test",
            frequency=np.array([1.0, 2.0, 3.0]),
            amplitude=np.array([0.1, 0.2, 0.3]),
        )
        assert comp.name == "test"
        assert len(comp.frequency) == 3
        assert len(comp.amplitude) == 3
        assert comp.domain == "frequency"
        assert comp.units == "linear"

    def test_component_with_phase(self):
        comp = SpectralComponent(
            name="phased",
            frequency=np.array([1.0, 2.0]),
            amplitude=np.array([0.5, 0.6]),
            phase=np.array([0.0, 1.57]),
        )
        assert comp.phase is not None
        assert len(comp.phase) == 2

    def test_component_metadata(self):
        comp = SpectralComponent(
            name="meta",
            frequency=np.array([1.0]),
            amplitude=np.array([0.5]),
            metadata={"source": "sensor_1"},
        )
        assert comp.metadata["source"] == "sensor_1"


class TestMultiElementRecord:
    def test_create_single_component_record(self):
        comp = SpectralComponent(
            name="c0",
            frequency=np.array([1.0, 2.0, 3.0]),
            amplitude=np.array([0.1, 0.2, 0.3]),
        )
        record = MultiElementRecord(record_id="r1", components=[comp])
        assert record.record_id == "r1"
        assert len(record.components) == 1
        assert record.representation == "single"

    def test_create_multi_component_record(self):
        c1 = SpectralComponent(name="a", frequency=np.array([1.0, 2.0]), amplitude=np.array([0.1, 0.2]))
        c2 = SpectralComponent(name="b", frequency=np.array([1.0, 2.0]), amplitude=np.array([0.3, 0.4]))
        record = MultiElementRecord(record_id="r2", components=[c1, c2], representation="multi")
        assert len(record.components) == 2
        assert record.representation == "multi"

    def test_record_lineage(self):
        comp = SpectralComponent(name="c", frequency=np.array([1.0]), amplitude=np.array([0.5]))
        record = MultiElementRecord(record_id="r3", components=[comp], lineage=["source", "processed"])
        assert record.lineage == ["source", "processed"]


class TestFrequencyGrid:
    def test_linear_grid(self):
        grid = FrequencyGrid.linear(0.1, 100.0, 50)
        assert len(grid.values) == 50
        assert grid.spacing == "linear"
        assert grid.values[0] == pytest.approx(0.1)

    def test_logarithmic_grid(self):
        grid = FrequencyGrid.logarithmic(0.1, 100.0, 50)
        assert len(grid.values) == 50
        assert grid.spacing == "log"


class TestGenerationConfig:
    def test_default_config(self):
        cfg = GenerationConfig()
        assert cfg.amplitude_shape == "flat"
        assert cfg.seed is None
        assert cfg.physical_min_amplitude == 1e-12

    def test_config_with_seed(self):
        cfg = GenerationConfig(seed=42, amplitude_shape="gaussian")
        assert cfg.seed == 42
        assert cfg.amplitude_shape == "gaussian"
