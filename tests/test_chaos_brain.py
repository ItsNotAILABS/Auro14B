"""ChaosCUDA + mini brains/hearts teaching suite."""

from __future__ import annotations

from auro_native_llm.chaos_cuda.plane import get_chaos_cuda
from auro_native_llm.brain.organs import build_brain_cluster
from auro_native_llm.polyglot.cuda_plane import get_cuda_plane
from auro_native_llm.organism.family import build_mind


def test_chaos_cuda_works_here():
    c = get_chaos_cuda(refresh=True)
    info = c.info()
    assert info["backend"] == "chaos_cuda"
    assert info["cuda_available"] is True
    bench = c.benchmark(128)
    assert bench["ok"] is True
    # plane detection prefers chaos on this host
    p = get_cuda_plane(refresh=True)
    assert p.backend == "chaos_cuda"
    assert p.cuda_available is True


def test_brain_cluster_curriculum():
    cl = build_brain_cluster()
    info = cl.info()
    assert set(info["domains"]) == {"code", "research", "math"}
    lessons = cl.teacher.lesson_batch(6)
    assert len(lessons) == 6
    assert any("MINI_BRAIN" in L for L in lessons)
    vitals = cl.heart.pulse(1.0)
    assert vitals["coherence"] >= 0
    assert "BRAIN-AI" in " ".join(cl.lineage) or "NeuroEmergence" in " ".join(cl.lineage)


def test_mind_teach_domains():
    mind = build_mind("Auro-2B", lite=True)
    assert mind.organs.brains is not None
    before = mind.language.train_steps
    rep = mind.teach_domains(steps_per_lesson=1)
    assert rep["ok"] is True
    assert rep["lessons"] == 6
    assert mind.language.train_steps > before
    assert rep["cuda"]["backend"] == "chaos_cuda"
    hp = mind.heart_pulse()
    assert "vitals" in hp and "thoughts" in hp
