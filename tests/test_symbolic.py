"""Symbolic compression + future 2/4B specs."""

from __future__ import annotations

from auro_native_llm.symbolic.compress import (
    SymbolicCompressor,
    future_2b_spec,
    future_4b_spec,
)
import numpy as np


def test_symbolic_match_and_compress():
    s = SymbolicCompressor()
    hits = s.match("open many websites parallel DOM multi site", top_k=3)
    assert hits
    assert any(p.domain == "web" for p in hits)
    ctx = s.expand_context("REST API sqlite auth")
    assert "SYMBOLIC_COMPRESS" in ctx
    v = np.random.randn(256)
    c = s.compress_vector(v, code_dim=32)
    assert c["code_dim"] == 32
    assert c["compression_ratio"] < 1.0
    eff = s.effective_intelligence(17_000_000, retrieval_docs=1000, tools=25)
    assert eff["effective_units"] > eff["neural_params"]


def test_future_specs():
    a = future_2b_spec()
    b = future_4b_spec()
    assert a.parameter_target == 2_000_000_000
    assert b.parameter_target == 4_000_000_000
    assert a.symbolic_program_budget >= 10_000
    assert "neural" in a.thesis.lower() or "2B" in a.thesis
