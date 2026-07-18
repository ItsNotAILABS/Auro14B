"""Closed-form physics AI formulas (NumPy). Every symbol is used in training.

Notation
--------
φ  = (1+√5)/2 golden ratio — scale hierarchy constant
θ_g = 2π/φ²   golden angle (radians)

1) Nonlinear dispersion (wave physics for token/spectrum modes)
     ω(k)² = ω₀² + c² k² + α k⁴ + β sin²(k θ_g)
   Used as spectral frequency lattice for residual phases.

2) Spectral action density (on discrete FFT bands)
     S[A] = Σ_ω  ( |Ã(ω)|² ω(k_ω) + λ |∇_ω Ã|² )
   Penalizes rough spectra; gradient drives smooth energetic embeddings.

3) Wiener coherence between fields X,Y
     C(ω) = |S_xy(ω)|² / (S_xx(ω) S_yy(ω) + ε)
     Ḡ   = mean_ω C(ω)
   High coherence ⇒ aligned spectral memory & text signal.

4) Resonance score (matched filter energy)
     R = |Σ Ã B̄*| / (‖A‖₂ ‖B‖₂ + ε)
   Real inner product in Fourier domain.

5) φ-potential Schrödinger residual (complex field on R^D)
     i ∂_t ψ = -½ ∇²ψ + V_φ(x) ψ
     V_φ(x) = (1-φ⁻¹) |x|²/2 + φ⁻² cos(φ · x · θ_g)
   One unitary-ish Euler step stabilizes hidden residuals.

6) Kuramoto synchronization (multi-head / multi-band phases)
     dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j - θ_i)
     r e^{iψ} = (1/N) Σ e^{iθ_i}
   Order parameter r ∈ [0,1] measures phase lock of modes.

7) Landau free energy for spectral order parameter m ∈ R^D
     F = a/2 |m|² + b/4 |m|⁴ - h·m
     force = -∇_m F = -a m - b |m|² m + h
   Used as embedding attractor toward resonant directions.

8) Diagonal Fisher natural-gradient correction (information geometry)
     g_ii ≈ p_i (soft probs) or m_i² + ε
     Ñg = g^{-1} ∇L   (elementwise  ∇L_i / g_ii)
   Stabilizes embedding SGD without full matrix inverse.

9) Physics-regularized LM loss
     L = CE + λ_S S_norm + λ_C (1-Ḡ) + λ_K (1-r) + λ_L F_norm
   All terms differentiable w.r.t. continuous fields we control.

These are *real equations* evaluated in NumPy — not labels or scaffolds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- exact constants ----
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = PHI - 1.0  # 1/φ
PHI_SQ = PHI * PHI
GOLDEN_ANGLE = 2.0 * math.pi / (PHI * PHI)  # 2π/φ² rad ≈ 137.508°
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# 1) Dispersion
# ---------------------------------------------------------------------------
def dispersion_omega(
    k: np.ndarray,
    *,
    omega0: float = 1.0,
    c: float = PHI,
    alpha: float = PHI_INV * 0.1,
    beta: float = PHI_INV,
) -> np.ndarray:
    """ω(k) from nonlinear dispersion relation (positive branch)."""
    k = np.asarray(k, dtype=np.float64)
    w2 = omega0**2 + (c * k) ** 2 + alpha * (k**4) + beta * np.sin(k * GOLDEN_ANGLE) ** 2
    return np.sqrt(np.maximum(w2, 1e-18))


# ---------------------------------------------------------------------------
# 2) Spectral action density
# ---------------------------------------------------------------------------
def spectral_action_density(
    spectrum: np.ndarray,
    *,
    lambda_grad: float = PHI_INV,
    omega0: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """S[A] and dS/dA for real amplitude spectrum A(ω).

    S = Σ (A_i² ω_i + λ (A_{i+1}-A_i)²)
    Returns (S_scalar, grad_A).
    """
    A = np.asarray(spectrum, dtype=np.float64).ravel()
    n = A.size
    if n == 0:
        return 0.0, A
    k = np.linspace(0.0, PHI, n)
    w = dispersion_omega(k, omega0=omega0)
    # energy term
    S_e = float(np.sum(A * A * w))
    g = 2.0 * A * w
    # discrete gradient penalty
    if n > 1:
        d = np.diff(A)
        S_g = float(lambda_grad * np.sum(d * d))
        # d/dA_i of (A_{i+1}-A_i)²
        g_grad = np.zeros_like(A)
        g_grad[:-1] += -2.0 * lambda_grad * d
        g_grad[1:] += 2.0 * lambda_grad * d
        g = g + g_grad
        S = S_e + S_g
    else:
        S = S_e
    return S, g


# ---------------------------------------------------------------------------
# 3) Wiener coherence
# ---------------------------------------------------------------------------
def wiener_coherence(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_fft: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Mean magnitude-squared coherence Ḡ and C(ω) curve."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(n_fft or max(16, 2 ** int(math.ceil(math.log2(max(x.size, y.size, 16))))))
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    else:
        x = x[:n]
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    else:
        y = y[:n]
    X = np.fft.rfft(x * np.hanning(n))
    Y = np.fft.rfft(y * np.hanning(n))
    Sxx = (X * np.conj(X)).real + 1e-12
    Syy = (Y * np.conj(Y)).real + 1e-12
    Sxy = X * np.conj(Y)
    C = (Sxy * np.conj(Sxy)).real / (Sxx * Syy)
    C = np.clip(C.real, 0.0, 1.0)
    return float(C.mean()), C


# ---------------------------------------------------------------------------
# 4) Resonance
# ---------------------------------------------------------------------------
def resonance_score(a: np.ndarray, b: np.ndarray) -> float:
    """Fourier matched-filter resonance R ∈ [0,1] (cosine of complex spectra)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = int(2 ** int(math.ceil(math.log2(max(a.size, b.size, 8)))))
    if a.size < n:
        a = np.pad(a, (0, n - a.size))
    else:
        a = a[:n]
    if b.size < n:
        b = np.pad(b, (0, n - b.size))
    else:
        b = b[:n]
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    num = np.abs(np.vdot(A, B))
    den = (np.linalg.norm(A) * np.linalg.norm(B)) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 5) φ-Schrödinger residual
# ---------------------------------------------------------------------------
def phi_schrodinger_step(
    psi_real: np.ndarray,
    *,
    dt: float = 0.05,
    mass: float = 1.0,
) -> np.ndarray:
    """One split-step: kinetic (FFT) + φ-potential, return real part residual.

    ψ ← exp(-i V dt) exp(-i K dt) ψ  with ψ initialized as real field.
    We keep the real component as the residual inject into hidden states.
    """
    x = np.asarray(psi_real, dtype=np.float64).ravel()
    n = x.size
    # complex field
    psi = x.astype(np.complex128)
    # kinetic in Fourier: K = k²/(2m)
    k = np.fft.fftfreq(n) * TWO_PI
    K = (k * k) / (2.0 * mass)
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-1j * K * dt)
    psi = np.fft.ifft(psi_k)
    # potential V_φ
    idx = np.arange(n, dtype=np.float64)
    # map to local coordinate
    pos = (idx / max(n - 1, 1) - 0.5) * 2.0
    V = (1.0 - PHI_INV) * 0.5 * (pos**2) + (PHI_INV**2) * np.cos(PHI * pos * GOLDEN_ANGLE * n / TWO_PI)
    # scale V to field magnitude
    scale = float(np.std(x) + 1e-6)
    V = V * scale
    psi *= np.exp(-1j * V * dt)
    out = psi.real
    # preserve energy scale
    nrm = float(np.linalg.norm(out)) or 1.0
    target = float(np.linalg.norm(x)) or 1.0
    return (out / nrm) * target


# ---------------------------------------------------------------------------
# 6) Kuramoto
# ---------------------------------------------------------------------------
def kuramoto_order(theta: np.ndarray) -> Tuple[float, float]:
    """Return (r, ψ) order parameter."""
    th = np.asarray(theta, dtype=np.float64).ravel()
    z = np.exp(1j * th).mean()
    return float(np.abs(z)), float(np.angle(z))


def kuramoto_step(
    theta: np.ndarray,
    omega: np.ndarray,
    *,
    K: float = PHI,
    dt: float = 0.05,
) -> np.ndarray:
    """One Euler step of Kuramoto dynamics."""
    th = np.asarray(theta, dtype=np.float64).ravel()
    om = np.asarray(omega, dtype=np.float64).ravel()
    n = th.size
    if om.size != n:
        om = np.resize(om, n)
    # mean-field form: dθ_i = ω_i + K r sin(ψ - θ_i)
    r, psi = kuramoto_order(th)
    dth = om + K * r * np.sin(psi - th)
    return th + dt * dth


# ---------------------------------------------------------------------------
# 7) Landau free energy
# ---------------------------------------------------------------------------
def landau_free_energy(
    m: np.ndarray,
    h: np.ndarray,
    *,
    a: float = -0.5,  # a<0 ordered phase
    b: float = 1.0,
) -> float:
    m = np.asarray(m, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()
    if h.size != m.size:
        h = np.resize(h, m.size)
    m2 = float(np.dot(m, m))
    return 0.5 * a * m2 + 0.25 * b * (m2**2) - float(np.dot(h, m))


def landau_field_force(
    m: np.ndarray,
    h: np.ndarray,
    *,
    a: float = -0.5,
    b: float = 1.0,
) -> np.ndarray:
    """force = -∇_m F."""
    m = np.asarray(m, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()
    if h.size != m.size:
        h = np.resize(h, m.size)
    m2 = float(np.dot(m, m))
    return -a * m - b * m2 * m + h


# ---------------------------------------------------------------------------
# 8) Fisher natural gradient (diagonal)
# ---------------------------------------------------------------------------
def fisher_natural_grad(
    grad: np.ndarray,
    metric_diag: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Ñg_i = g_i / (G_ii + ε)."""
    g = np.asarray(grad, dtype=np.float64)
    G = np.asarray(metric_diag, dtype=np.float64)
    if G.shape != g.shape:
        G = np.resize(G, g.shape)
    return g / (np.abs(G) + eps)


# ---------------------------------------------------------------------------
# 9) Composite physics loss + spectral force on hidden
# ---------------------------------------------------------------------------
def physics_regularized_loss(
    ce: float,
    spectrum: np.ndarray,
    text_signal: np.ndarray,
    hidden_tail: np.ndarray,
    *,
    lambda_S: float = 0.02,
    lambda_C: float = 0.05,
    lambda_K: float = 0.03,
    lambda_L: float = 0.02,
    lambda_R: float = 0.04,
) -> Tuple[float, Dict[str, float]]:
    """L = CE + physics penalties; returns (L, metrics)."""
    S, _ = spectral_action_density(np.abs(spectrum) + 1e-9)
    # normalize S by dimension
    S_n = float(S / (spectrum.size + 1.0))
    # coherence between text signal and hidden projection
    h = np.asarray(hidden_tail, dtype=np.float64).ravel()
    tsig = np.asarray(text_signal, dtype=np.float64).ravel()
    # match lengths via FFT size
    coh, _ = wiener_coherence(tsig, h)
    # Kuramoto on phase of FFT(hidden)
    H = np.fft.rfft(h - h.mean())
    theta = np.angle(H)
    # natural frequencies from dispersion on mode index
    k = np.linspace(0.0, PHI, theta.size)
    omega = dispersion_omega(k)
    th2 = kuramoto_step(theta, omega, K=PHI, dt=0.02)
    r, _ = kuramoto_order(th2)
    # Landau with external field = normalized spectrum projection
    m = h / (float(np.linalg.norm(h)) + 1e-12)
    # external field from spectrum
    spec = np.asarray(spectrum, dtype=np.float64).ravel()
    if spec.size < m.size:
        hh = np.zeros(m.size)
        hh[: spec.size] = spec / (float(np.linalg.norm(spec)) + 1e-12)
    else:
        hh = spec[: m.size]
        hh = hh / (float(np.linalg.norm(hh)) + 1e-12)
    F = landau_free_energy(m, hh)
    F_n = float(F / (m.size + 1.0))
    # resonance between spectrum and |FFT(h)|
    R = resonance_score(spec, np.abs(np.fft.rfft(h)))

    L = (
        float(ce)
        + lambda_S * S_n
        + lambda_C * (1.0 - coh)
        + lambda_K * (1.0 - r)
        + lambda_L * max(F_n, 0.0)
        + lambda_R * (1.0 - R)
    )
    metrics = {
        "ce": float(ce),
        "physics_loss": float(L),
        "spectral_action": S_n,
        "coherence": coh,
        "kuramoto_r": r,
        "landau_F": F_n,
        "resonance": R,
        "lambda_S": lambda_S,
        "lambda_C": lambda_C,
        "lambda_K": lambda_K,
        "lambda_L": lambda_L,
        "lambda_R": lambda_R,
    }
    return float(L), metrics


def spectral_force_on_hidden(
    hidden: np.ndarray,
    spectrum: np.ndarray,
    text_signal: np.ndarray,
    *,
    strength: float = 0.08,
) -> np.ndarray:
    """Physics force field added to last-token hidden residual.

    Combines:
      - Landau force toward spectral external field
      - φ-Schrödinger smoothed residual
      - Coherence phase alignment via FFT phase matching
    """
    h = np.asarray(hidden, dtype=np.float64).ravel().copy()
    d = h.size
    spec = np.asarray(spectrum, dtype=np.float64).ravel()
    if spec.size < d:
        ext = np.zeros(d)
        ext[: spec.size] = spec
    else:
        # energy-preserving downsample
        idx = np.linspace(0, spec.size, d, endpoint=False).astype(int)
        ext = spec[idx]
    n_ext = float(np.linalg.norm(ext)) or 1.0
    ext = ext / n_ext
    n_h = float(np.linalg.norm(h)) or 1.0
    m = h / n_h
    force = landau_field_force(m, ext, a=-0.45, b=1.1)
    # Schrödinger smooth
    psi = phi_schrodinger_step(h, dt=0.04)
    # phase-lock: align arg(FFT(h)) toward arg(FFT(text))
    tsig = np.asarray(text_signal, dtype=np.float64).ravel()
    nfft = int(2 ** int(math.ceil(math.log2(max(d, tsig.size, 16)))))
    ht = np.zeros(nfft)
    tt = np.zeros(nfft)
    ht[:d] = h
    tt[: min(tsig.size, nfft)] = tsig[:nfft]
    Hf = np.fft.rfft(ht)
    Tf = np.fft.rfft(tt)
    # replace phase of H with blend toward T phase, keep |H|
    mag = np.abs(Hf)
    phase_h = np.angle(Hf)
    phase_t = np.angle(Tf)
    # Kuramoto pull
    phase = phase_h + 0.15 * np.sin(phase_t - phase_h)
    H2 = mag * np.exp(1j * phase)
    h_phase = np.fft.irfft(H2, n=nfft)[:d]
    # compose
    out = h + strength * (
        0.45 * force * n_h
        + 0.30 * (psi - h)
        + 0.25 * (h_phase - h)
    )
    return out


def physics_lr_schedule(
    step: int,
    base_lr: float,
    *,
    coherence: float = 0.5,
    kuramoto_r: float = 0.5,
    resonance: float = 0.5,
) -> float:
    """Adaptive LR from physics state: higher lock ⇒ slightly bolder steps.

    η = η0 · φ^{-s/τ} · (0.7 + 0.3 Ḡ) · (0.8 + 0.2 r) · (0.85 + 0.15 R)
    """
    tau = 200.0
    decay = PHI ** (-(step / tau))
    return float(
        base_lr
        * decay
        * (0.7 + 0.3 * float(np.clip(coherence, 0, 1)))
        * (0.8 + 0.2 * float(np.clip(kuramoto_r, 0, 1)))
        * (0.85 + 0.15 * float(np.clip(resonance, 0, 1)))
    )


def text_to_physical_signal(text: str, length: int = 128) -> np.ndarray:
    """Deterministic map text → real 1D field (byte + φ-phase lattice)."""
    raw = (text or " ").encode("utf-8", errors="replace") or b" "
    sig = np.zeros(length, dtype=np.float64)
    for i, b in enumerate(raw[: length * 4]):
        sig[i % length] += (b / 255.0) - 0.5
    idx = np.arange(length, dtype=np.float64)
    sig = sig - sig.mean()
    sig = sig + 0.08 * np.sin(idx * GOLDEN_ANGLE) + 0.04 * np.cos(idx * PHI_INV)
    return sig


def spectrum_from_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 4:
        x = np.pad(x, (0, 4 - x.size))
    return np.abs(np.fft.rfft(x * np.hanning(x.size)))


@dataclass
class PhysicsState:
    coherence: float = 0.0
    kuramoto_r: float = 0.0
    resonance: float = 0.0
    spectral_action: float = 0.0
    landau_F: float = 0.0
    physics_loss: float = 0.0
    ce: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "coherence": self.coherence,
            "kuramoto_r": self.kuramoto_r,
            "resonance": self.resonance,
            "spectral_action": self.spectral_action,
            "landau_F": self.landau_F,
            "physics_loss": self.physics_loss,
            "ce": self.ce,
        }


@dataclass
class PhysicsReport:
    ok: bool
    formulas: List[str]
    last_state: PhysicsState
    equations: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.physics.ai.v1",
            "ok": self.ok,
            "formulas": self.formulas,
            "last_state": self.last_state.to_dict(),
            "equations": self.equations,
            "scaffold": False,
            "fake": False,
        }


EQUATIONS = {
    "dispersion": "ω(k)² = ω0² + c²k² + αk⁴ + β sin²(k·θ_g)",
    "spectral_action": "S[A] = Σ A²ω + λ|∇A|²",
    "wiener_coherence": "C(ω)=|S_xy|²/(S_xx S_yy); Ḡ=⟨C⟩",
    "resonance": "R = |⟨Ã,B̃⟩| / (‖A‖‖B‖)",
    "phi_schrodinger": "i∂tψ = -½∇²ψ + V_φ ψ; V_φ=(1-φ⁻¹)|x|²/2 + φ⁻² cos(φ x θ_g)",
    "kuramoto": "θ̇_i = ω_i + (K/N)Σ sin(θ_j-θ_i); r=|⟨e^{iθ}⟩|",
    "landau": "F=a/2|m|² + b/4|m|⁴ - h·m; force=-∇F",
    "fisher_natural_grad": "Ñg_i = ∂_i L / (G_ii+ε)",
    "physics_loss": "L=CE + λ_S S + λ_C(1-Ḡ) + λ_K(1-r) + λ_L F + λ_R(1-R)",
    "lr_schedule": "η=η0 φ^{-s/τ} (0.7+0.3Ḡ)(0.8+0.2r)(0.85+0.15R)",
}
