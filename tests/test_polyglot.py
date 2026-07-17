"""Polyglot organ: Python + Julia + Haskell + CUDA plane."""

from __future__ import annotations

from auro_native_llm.polyglot.cuda_plane import get_cuda_plane
from auro_native_llm.polyglot.organ import PolyglotOrgan
from auro_native_llm.polyglot.runtimes import PolyglotRuntime


def test_cuda_plane_detects():
    p = get_cuda_plane(refresh=True)
    info = p.info()
    assert info["backend"] in (
        "torch_cuda",
        "torch_mps",
        "torch_cpu",
        "cupy_cuda",
        "numpy",
    )
    a = __import__("numpy").random.randn(32, 32)
    b = __import__("numpy").random.randn(32, 32)
    c = p.matmul(a, b)
    assert c.shape == (32, 32)


def test_runtime_status():
    rt = PolyglotRuntime()
    st = rt.status()
    assert st["python"]["available"] is True
    # Julia should be present on this machine
    assert "julia" in st
    assert "haskell" in st


def test_julia_kernels():
    rt = PolyglotRuntime()
    h = rt.julia_call("health")
    assert h.get("ok") is True, h
    assert h.get("lang") == "julia"
    e = rt.julia_call("spectral_energy", {"x": [1.0, 0, -1, 0, 1, 0, -1]})
    assert e.get("ok") is True, e
    assert float(e["energy"]) > 0
    p = rt.julia_call("phi_powers", {"n": 6})
    assert p.get("ok") is True, p
    assert abs(float(p["sum"]) - sum(((1 + 5**0.5) / 2) ** i for i in range(1, 7))) < 1e-6


def test_haskell_kernels():
    rt = PolyglotRuntime()
    h = rt.haskell_call("health")
    assert h.get("ok") is True, h
    # native or semantics both ok
    assert "haskell" in h.get("lang", "")
    e = rt.haskell_call("spectral_energy", "1,0,-1,0,1,0,-1")
    assert e.get("ok") is True, e
    p = rt.haskell_call("phi_powers", "8")
    assert p.get("ok") is True and float(p["sum"]) > 0
    emb = rt.haskell_call("multi_fft_embed", "MESIE")
    assert emb.get("ok") is True and int(emb["dim"]) > 0


def test_polyglot_suite_and_parity():
    organ = PolyglotOrgan()
    r = organ.suite()
    assert r.ok is True, r.to_dict()
    # spectral parity across langs
    s = organ.spectral_energy_all()
    assert s.ok is True, s.to_dict()
    assert s.results["parity_spread"] < 1e-3
    info = organ.info()
    assert info["langs"]["python"] is True
    assert info["ok_rate"] == 1.0


def test_mind_polyglot_organ():
    from auro_native_llm.organism.family import build_mind

    mind = build_mind("Auro-2B", lite=True)
    assert mind.organs.polyglot is not None
    r = mind.polyglot("phi", n=5)
    assert r.ok is True, r.error
    assert "polyglot" in mind.info()["capabilities"]
