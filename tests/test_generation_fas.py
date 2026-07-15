"""Tests for FAS generation (additional coverage)."""

import numpy as np
import pytest

from mesie.core.config import GenerationConfig
from mesie.generation.fas import generate_fas


class TestFASAdditional:
    def test_fas_positive_amplitudes(self):
        cfg = GenerationConfig(seed=99)
        rec = generate_fas(cfg)
        assert np.all(rec.components[0].amplitude >= 0.0)

    def test_fas_lineage(self):
        cfg = GenerationConfig(seed=5)
        rec = generate_fas(cfg)
        assert "fas" in rec.lineage

    def test_fas_with_perturbation(self):
        cfg = GenerationConfig(seed=8, stochastic_perturbation=0.1)
        rec = generate_fas(cfg)
        assert len(rec.components[0].amplitude) > 0
