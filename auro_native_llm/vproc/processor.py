"""Virtual processor: every prompt is a measurable work call.

Metrics (inspectable runtime):
  bytes_processed, nova_cycles, entropy, spectral_buckets,
  coherence, resonance, kuramoto_r, routing, receipt tip, hash-chain state
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.ghost.receipts import GhostReceiptChain
from auro_native_llm.physics.formulas import (
    PHI,
    dispersion_omega,
    kuramoto_order,
    kuramoto_step,
    resonance_score,
    spectral_action_density,
    spectrum_from_signal,
    text_to_physical_signal,
    wiener_coherence,
)


@dataclass
class WorkMetrics:
    """Fully inspectable work-call measurement."""

    bytes_processed: int = 0
    nova_cycles: float = 0.0  # abstract compute units
    entropy: float = 0.0
    spectral_buckets: List[float] = field(default_factory=list)
    coherence: float = 0.0
    resonance: float = 0.0
    kuramoto_r: float = 0.0
    spectral_action: float = 0.0
    dispersion_mean_omega: float = 0.0
    routing: str = "mesie_only"  # mesie_only | escalate_llm | dual_brain
    latency_ms: float = 0.0
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bytes_processed": self.bytes_processed,
            "nova_cycles": self.nova_cycles,
            "entropy": self.entropy,
            "spectral_buckets": self.spectral_buckets,
            "n_buckets": len(self.spectral_buckets),
            "coherence": self.coherence,
            "resonance": self.resonance,
            "kuramoto_r": self.kuramoto_r,
            "spectral_action": self.spectral_action,
            "dispersion_mean_omega": self.dispersion_mean_omega,
            "routing": self.routing,
            "latency_ms": self.latency_ms,
            "deterministic": self.deterministic,
        }


@dataclass
class WorkCall:
    """One measurable prompt → work unit."""

    call_id: str
    prompt: str
    metrics: WorkMetrics
    features: Dict[str, Any]
    receipt_tip: Optional[str] = None
    chain_id: Optional[str] = None
    ok: bool = True
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.vproc.work_call.v1",
            "call_id": self.call_id,
            "prompt": self.prompt[:500],
            "metrics": self.metrics.to_dict(),
            "features": self.features,
            "receipt_tip": self.receipt_tip,
            "chain_id": self.chain_id,
            "ok": self.ok,
            "result": self.result,
            "philosophy": (
                "MESIE/Ghost default path is deterministic. "
                "LLM only when escalation justified by cleaned spectral features."
            ),
        }


def _shannon_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.abs(x)
    s = float(x.sum()) + 1e-12
    p = x / s
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum() / math.log(2))


def _bucket_energy(spec: np.ndarray, n_buckets: int = 16) -> List[float]:
    spec = np.asarray(spec, dtype=np.float64).ravel()
    if spec.size == 0:
        return [0.0] * n_buckets
    edges = np.linspace(0, spec.size, n_buckets + 1).astype(int)
    out = []
    for i in range(n_buckets):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out.append(float(spec[lo:hi].mean()) if hi > lo else 0.0)
    total = sum(out) + 1e-12
    return [v / total for v in out]


class MesieVirtualProcessor:
    """Inspectable MESIE virtual processor for the computing family."""

    def __init__(self, mind: Any = None) -> None:
        self.mind = mind
        self.calls: List[WorkCall] = []
        self.total_bytes = 0
        self.total_nova = 0.0
        self.chain = GhostReceiptChain(chain_id=f"vproc-{uuid.uuid4().hex[:10]}")

    def measure_prompt(self, prompt: str) -> WorkMetrics:
        """Convert prompt → physical signal → full metric suite (no LLM)."""
        t0 = time.perf_counter()
        raw = (prompt or "").encode("utf-8", errors="replace")
        nbytes = len(raw)
        sig = text_to_physical_signal(prompt or " ", length=128)
        spec = spectrum_from_signal(sig)
        buckets = _bucket_energy(spec, 16)
        ent = _shannon_entropy(spec)
        S, _ = spectral_action_density(spec)
        k = np.linspace(0.0, float(PHI), 64)
        w = dispersion_omega(k)
        # phase dynamics
        phase = np.angle(np.fft.rfft(sig - sig.mean()))
        if phase.size < 8:
            phase = np.linspace(0, 2 * math.pi, 16)
        om = w[: phase.size] if w.size >= phase.size else np.resize(w, phase.size)
        th2 = kuramoto_step(phase, om, K=float(PHI), dt=0.05)
        r, _ = kuramoto_order(th2)
        coh, _ = wiener_coherence(sig, np.sin(np.arange(sig.size) * 0.17))
        R = resonance_score(spec, np.abs(np.fft.rfft(sig)))
        # NOVA cycles: abstract = bytes * entropy_scale * spectral_cost
        nova = float(nbytes) * (1.0 + ent / 8.0) * (1.0 + float(S) / (spec.size + 1.0))
        return WorkMetrics(
            bytes_processed=nbytes,
            nova_cycles=nova,
            entropy=ent,
            spectral_buckets=buckets,
            coherence=float(coh),
            resonance=float(R),
            kuramoto_r=float(r),
            spectral_action=float(S / (spec.size + 1.0)),
            dispersion_mean_omega=float(w.mean()),
            routing="mesie_only",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            deterministic=True,
        )

    def execute_mesie_path(self, prompt: str, metrics: WorkMetrics) -> Dict[str, Any]:
        """Deterministic MESIE/Ghost work — filter, features, shadow sim, stream metrics."""
        t0 = time.perf_counter()
        sig = text_to_physical_signal(prompt, 128)
        # real-time-ish smoothing (exponential moving average filter)
        alpha = 0.35
        smooth = np.zeros_like(sig)
        smooth[0] = sig[0]
        for i in range(1, sig.size):
            smooth[i] = alpha * sig[i] + (1 - alpha) * smooth[i - 1]
        # frequency-domain features
        spec = spectrum_from_signal(smooth)
        # shadow execution: simulate one step of φ-dispersion prediction
        k = np.linspace(0.0, float(PHI), min(32, spec.size))
        omega = dispersion_omega(k)
        # shadow state: phase advance
        shadow = spec[: k.size] * np.cos(omega * 0.05)
        # ghost node if available
        ghost_out: Dict[str, Any] = {}
        try:
            from auro_native_llm.ghost.node import MesieGhostNode

            node = MesieGhostNode(self.mind)
            ghost_out = {
                "spectral": node.run("spectral", {"text": prompt}).to_dict(),
                "embed": node.run("embed", {"text": prompt}).to_dict(),
                "helix": node.run("helix", {"text": prompt}).to_dict(),
            }
        except Exception as exc:
            ghost_out = {"note": str(exc)[:120]}

        # dual brain optional (Julia) — still deterministic physics
        brain: Dict[str, Any] = {}
        try:
            from auro_native_llm.dual import DualOrganism

            dual = DualOrganism(self.mind, n_cores=4)
            # only brain health + short think if julia quick — use measure already
            brain = dual.brain_call("brain_cycle", intent=prompt[:200], steps=2)
        except Exception as exc:
            brain = {"ok": False, "error": str(exc)[:120]}

        features = {
            "smooth_energy": float(np.sum(smooth**2)),
            "spec_peak_bin": int(np.argmax(spec)),
            "shadow_l2": float(np.linalg.norm(shadow)),
            "buckets": metrics.spectral_buckets,
            "coherence": metrics.coherence,
            "resonance": metrics.resonance,
            "kuramoto_r": metrics.kuramoto_r,
        }
        return {
            "path": "mesie_ghost_deterministic",
            "features": features,
            "ghost": ghost_out,
            "brain": {
                "ok": brain.get("ok"),
                "focus_band": brain.get("focus_band"),
                "mean_resonance": brain.get("mean_resonance"),
                "mean_kuramoto_r": brain.get("mean_kuramoto_r"),
                "lang": brain.get("lang"),
                "distributed": brain.get("distributed"),
            },
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "deterministic": True,
            "llm_used": False,
        }

    def should_escalate_llm(self, prompt: str, metrics: WorkMetrics, mesie_result: Dict[str, Any]) -> tuple[bool, str]:
        """LLM only when justified — not every step.

        Escalate if:
          - explicit plan/explain/strategy language AND low resonance lock
          - OR mesie path incomplete
          - OR operator-tagged escalate
        Do NOT escalate pure spectral/match/filter/stream work.
        """
        low = (prompt or "").lower()
        pure_signal = any(
            k in low
            for k in (
                "filter",
                "smooth",
                "psd",
                "fas",
                "fft",
                "spectral",
                "embed",
                "match",
                "coherence",
                "stream",
                "sensor",
                "measure",
            )
        )
        wants_language = any(
            k in low
            for k in (
                "explain",
                "plan",
                "strategy",
                "write a",
                "summarize",
                "why should",
                "design",
                "justify",
            )
        )
        if "escalate" in low or "use llm" in low:
            return True, "operator_requested_escalation"
        if pure_signal and not wants_language:
            return False, "deterministic_signal_path_sufficient"
        feats = mesie_result.get("features") or {}
        R = float(feats.get("resonance") or metrics.resonance)
        r = float(feats.get("kuramoto_r") or metrics.kuramoto_r)
        # high lock + cleaned features → skip LLM
        if R >= 0.45 and r >= 0.2 and not wants_language:
            return False, "spectral_lock_sufficient_skip_llm"
        if wants_language:
            return True, "language_planning_requires_llm"
        if not mesie_result.get("deterministic", True):
            return True, "mesie_path_non_deterministic"
        return False, "default_mesie_only"

    def escalate_llm(self, prompt: str, mesie_result: Dict[str, Any]) -> Dict[str, Any]:
        """Higher-level reasoning only — fed cleaned spectral features, not raw chaos."""
        feats = mesie_result.get("features") or {}
        cleaned = (
            f"[ESCALATION — MESIE features first, not raw prompt alone]\n"
            f"coherence={feats.get('coherence')} resonance={feats.get('resonance')} "
            f"kuramoto_r={feats.get('kuramoto_r')} peak_bin={feats.get('spec_peak_bin')}\n"
            f"shadow_l2={feats.get('shadow_l2')} smooth_E={feats.get('smooth_energy')}\n"
            f"Intent: {prompt}\n"
            f"Produce a short plan. Label model inferences as uncertain.\n"
        )
        text = ""
        used = False
        if self.mind is not None and hasattr(self.mind, "language"):
            try:
                lang = self.mind.language
                prev = getattr(self.mind, "train_every_act", None)
                if hasattr(self.mind, "train_every_act"):
                    self.mind.train_every_act = False
                try:
                    g = lang.generate(cleaned, max_new_tokens=48)
                    text = str(getattr(g, "text", g))[:1500]
                    used = True
                finally:
                    if prev is not None:
                        self.mind.train_every_act = prev
            except Exception as exc:
                text = f"[llm_error] {exc}"
        else:
            # still return structured plan without giant model
            text = (
                f"Plan from spectral features only: "
                f"focus peak_bin={feats.get('spec_peak_bin')}, "
                f"improve lock (r={feats.get('kuramoto_r')}), "
                f"act on high-energy buckets. No LLM weights loaded."
            )
            used = False
        return {
            "path": "llm_escalation",
            "text": text,
            "used_model": used,
            "fed_cleaned_features": True,
            "deterministic": False,
        }

    def work(self, prompt: str, *, force_mesie_only: bool = False) -> WorkCall:
        """Full inspectable work call: measure → MESIE → maybe escalate → receipt."""
        call_id = f"wc-{uuid.uuid4().hex[:12]}"
        metrics = self.measure_prompt(prompt)
        self.chain.append(
            "intent",
            {"initiator": "operator", "prompt": prompt[:300], "call_id": call_id},
            actor="operator",
        )
        mesie = self.execute_mesie_path(prompt, metrics)
        self.chain.append(
            "mesie",
            {
                "engine": "mesie_virtual_processor",
                "path": mesie["path"],
                "features": mesie.get("features"),
                "latency_ms": mesie.get("latency_ms"),
                "change": "deterministic_spectral_work",
            },
            actor="vproc.mesie",
            ok=True,
        )

        escalate, reason = (False, "force_mesie_only")
        llm_part: Dict[str, Any] = {"used": False}
        if not force_mesie_only:
            escalate, reason = self.should_escalate_llm(prompt, metrics, mesie)
        if escalate:
            metrics.routing = "escalate_llm"
            llm_part = self.escalate_llm(prompt, mesie)
            llm_part["reason"] = reason
            self.chain.append(
                "llm",
                {
                    "tool": "llm_escalate",
                    "reason": reason,
                    "change": "escalation",
                    "chars": len(str(llm_part.get("text") or "")),
                },
                actor="vproc.llm",
                model_id=getattr(self.mind, "model_id", None),
            )
        else:
            metrics.routing = "mesie_only"
            self.chain.append(
                "validate",
                {"check": "skip_llm", "reason": reason, "change": "mesie_sufficient"},
                actor="vproc.router",
                ok=True,
            )

        self.chain.append(
            "result",
            {
                "ok": True,
                "routing": metrics.routing,
                "nova_cycles": metrics.nova_cycles,
                "bytes": metrics.bytes_processed,
                "change": "work_call_complete",
            },
            actor="vproc",
            ok=True,
        )

        metrics.latency_ms = metrics.latency_ms + float(mesie.get("latency_ms") or 0)
        self.total_bytes += metrics.bytes_processed
        self.total_nova += metrics.nova_cycles

        call = WorkCall(
            call_id=call_id,
            prompt=prompt,
            metrics=metrics,
            features=mesie.get("features") or {},
            receipt_tip=self.chain.tip_hash,
            chain_id=self.chain.chain_id,
            ok=True,
            result={
                "mesie": mesie,
                "llm": llm_part,
                "escalate": escalate,
                "escalate_reason": reason,
                "custody": self.chain.custody_answers(),
            },
        )
        self.calls.append(call)
        return call

    def health(self) -> Dict[str, Any]:
        return {
            "schema": "auro.vproc.health.v1",
            "n_calls": len(self.calls),
            "total_bytes": self.total_bytes,
            "total_nova_cycles": self.total_nova,
            "chain_id": self.chain.chain_id,
            "tip_hash": self.chain.tip_hash,
            "doctrine": {
                "default": "mesie_ghost_deterministic",
                "llm": "escalate_only_when_justified",
                "benefits": [
                    "faster_cheaper_skip_llm",
                    "auditable_deterministic_math",
                    "reduced_hallucinations",
                    "edge_low_latency",
                    "binary_envelope_fit",
                ],
            },
        }


def run_work_call(prompt: str, mind: Any = None, **kw: Any) -> Dict[str, Any]:
    proc = MesieVirtualProcessor(mind)
    call = proc.work(prompt, **kw)
    return {"call": call.to_dict(), "processor": proc.health()}
