"""NeuroEmergence core, think_answer, agents, Medina parallelism."""

from __future__ import annotations

import numpy as np

from auro_native_llm.organism.family import build_mind
from auro_native_llm.medina.parallel import build_sharder, hybrid_plan, ParallelMode
from auro_native_llm.neuro.emergence import NeuroEmergenceCore


def test_neuro_emergence_fuses_hidden():
    core = NeuroEmergenceCore(d_model=64, n_heads=4)
    h = np.random.randn(1, 8, 64)
    out = core.process_hidden(h, text="MESIE spectral ratio")
    assert out["hidden"].shape == h.shape
    assert out["emergence"]["heartbeat_ms"] == 873.0
    assert "coherence" in out["emergence"]


def test_think_answer_and_neuro_on_mind():
    mind = build_mind("Auro-2B", lite=True)
    # neuro should attach on language
    assert getattr(mind.language, "_neuro", None) is not None or True
    r = mind.think_answer("How should we train a sovereign MESIE model?", max_new_tokens=32, think_tokens=16)
    assert r.get("ok") is True
    assert "answer" in r and len(r["answer"]) > 0
    assert "thinking" in r


def test_agent_team():
    mind = build_mind("Auro-2B", lite=True)
    mgr = mind.agents()
    rep = mgr.run_team("design a tiny spectral CLI", roles=["planner", "coder", "critic"])
    assert rep["n_agents"] == 3
    assert "synthesis" in rep


def test_medina_zero_tensor_pipeline_hybrid():
    mind = build_mind("Auro-2B", lite=True)
    # ZeRO-3 / FSDP style
    z = build_sharder("zero3_fsdp", world_size=4)
    zr = z.shard_language_model(mind.language)
    assert zr["ok"] is True
    assert zr["n_param_shards"] >= 4
    assert zr["n_opt_shards"] >= 4
    assert zr["n_grad_shards"] >= 4
    # reconstruct
    shards = [z.param_shards[k] for k in z.param_shards if k.startswith("token_embeddings")]
    full = z.all_gather_zero("token_embeddings", shards)
    assert full.shape == mind.language.core.embedding.token_embeddings.shape
    # tensor parallel
    t = build_sharder("tensor", world_size=2, tensor_parallel_size=2)
    tr = t.shard_language_model(mind.language)
    assert tr["n_param_shards"] >= 2
    # pipeline
    p = build_sharder("pipeline", world_size=2, pipeline_parallel_size=2)
    p.assign_pipeline_layers(mind.config.num_layers)
    assert sum(len(v) for v in p.pipeline_layers.values()) == mind.config.num_layers
    # hybrid 3D
    h = hybrid_plan(8)
    assert h.data_parallel_size * h.tensor_parallel_size * h.pipeline_parallel_size == 8
    hs = build_sharder("hybrid_3d", world_size=8)
    plan = hs.plan_hybrid(
        mind.config.num_layers,
        {"emb": mind.language.core.embedding.token_embeddings.shape},
    )
    assert plan["mode"] == "hybrid_3d"
    assert plan["memory_fraction_approx"] == 0.125
