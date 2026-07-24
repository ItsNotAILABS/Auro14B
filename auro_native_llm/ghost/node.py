"""MESIE Ghost Node — deterministic spectral/math layer (no LLM).

Handles: embed, match/metrics, helix, spectral FFT, intelligence observe,
connectome pulse. Fast, auditable, local-first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class NodeResult:
    ok: bool
    engine: str
    action: str
    output: Dict[str, Any]
    latency_ms: float
    deterministic: bool = True
    evidence_ref: str = ""
    claim_kind: str = "deterministic_mesie"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "engine": self.engine,
            "action": self.action,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "deterministic": self.deterministic,
            "evidence_ref": self.evidence_ref,
            "claim_kind": self.claim_kind,
        }


class MesieGhostNode:
    """Deterministic MESIE work unit for Ghost Agents."""

    def __init__(self, mind: Any = None) -> None:
        self.mind = mind
        self._rt = getattr(mind, "mesie_runtime", None) if mind is not None else None
        self.calls = 0

    def _runtime(self) -> Any:
        if self._rt is not None:
            return self._rt
        try:
            from auro_native_llm.mesie_runtime import get_mesie_runtime

            mid = getattr(self.mind, "model_id", "Auro-14B") if self.mind else "Auro-14B"
            self._rt = get_mesie_runtime(mid, lite=True)
        except Exception:
            self._rt = None
        return self._rt

    def run(self, action: str, payload: Optional[Dict[str, Any]] = None) -> NodeResult:
        payload = payload or {}
        t0 = time.perf_counter()
        self.calls += 1
        action = (action or "embed").lower()
        try:
            if action in ("embed", "embedding", "vectorize"):
                return self._embed(payload, t0)
            if action in ("helix", "helix_encode"):
                return self._helix(payload, t0)
            if action in ("spectral", "fft", "metrics"):
                return self._spectral(payload, t0)
            if action in ("intelligence", "reason_proto"):
                return self._intelligence(payload, t0)
            if action in ("connectome", "brain"):
                return self._connectome(payload, t0)
            if action in ("match", "validate"):
                return self._match_validate(payload, t0, action)
            if action in ("probe", "health", "capabilities"):
                return self._health(t0)
            # default: embed signal from text
            return self._embed(payload, t0)
        except Exception as exc:
            return NodeResult(
                ok=False,
                engine="mesie_ghost_node",
                action=action,
                output={"error": str(exc)[:400]},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                deterministic=True,
                evidence_ref="mesie_ghost_node:error",
            )

    def _embed(self, payload: Dict[str, Any], t0: float) -> NodeResult:
        text = str(payload.get("text") or payload.get("signal") or "MESIE ghost")
        dim = int(payload.get("dim") or 64)
        rt = self._runtime()
        if rt is not None:
            vec = rt.embed_text(text, dim=dim)
            ref = f"mesie_runtime.embed:{getattr(rt, 'mesie_version', '?')}"
        else:
            # pure numpy fallback deterministic
            raw = text.encode("utf-8")
            vec = [((b / 255.0) - 0.5) for b in raw[:dim]]
            while len(vec) < dim:
                vec.append(0.0)
            n = float(np.linalg.norm(vec)) or 1.0
            vec = (np.asarray(vec) / n).tolist()
            ref = "numpy.spectral_fallback"
        return NodeResult(
            ok=True,
            engine="mesie_ghost_node",
            action="embed",
            output={"embedding_dim": len(vec), "embedding_head": vec[:8], "text_len": len(text)},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref=ref,
        )

    def _helix(self, payload: Dict[str, Any], t0: float) -> NodeResult:
        vals = payload.get("values") or payload.get("vector")
        if vals is None:
            text = str(payload.get("text") or "helix")
            vals = [((b / 255.0) - 0.5) for b in text.encode("utf-8")[:32]]
        rt = self._runtime()
        if rt is not None:
            out = rt.helix_encode(list(vals))
        else:
            out = {"ok": False, "error": "helix unbound"}
        return NodeResult(
            ok=bool(out.get("ok", True)),
            engine="mesie.helix",
            action="helix",
            output=out if isinstance(out, dict) else {"result": str(out)[:500]},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref="mesie.helix.HelixEncoder",
        )

    def _spectral(self, payload: Dict[str, Any], t0: float) -> NodeResult:
        from auro_native_llm.mesie_compute import spectral_fft_metrics, text_to_signal

        text = str(payload.get("text") or "spectral")
        sig = text_to_signal(text, int(payload.get("length") or 128))
        metrics = spectral_fft_metrics(sig, bands=int(payload.get("bands") or 16))
        return NodeResult(
            ok=True,
            engine="mesie.spectral_fft",
            action="spectral",
            output={"metrics": metrics, "signal_len": int(sig.size)},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref="mesie_compute.spectral_fft_metrics",
        )

    def _intelligence(self, payload: Dict[str, Any], t0: float) -> NodeResult:
        rt = self._runtime()
        obs = payload.get("observation") or {"signal": 0.5, "intent_len": float(len(str(payload.get("text") or "")))}
        if rt is not None:
            out = rt.intelligence_reason(obs if isinstance(obs, dict) else {"v": 1.0})
        else:
            out = {"ok": False, "error": "intelligence unbound"}
        return NodeResult(
            ok=bool(out.get("ok", True)),
            engine="mesie.intelligence",
            action="intelligence",
            output=out if isinstance(out, dict) else {"result": str(out)[:500]},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref="mesie.IntelligenceProtocol",
            # intelligence protocol is structured but not pure math — still local
            deterministic=True,
        )

    def _connectome(self, payload: Dict[str, Any], t0: float) -> NodeResult:
        rt = self._runtime()
        if rt is not None:
            out = rt.connectome_pulse(str(payload.get("focus") or "working_memory"))
        else:
            out = {"ok": False, "error": "connectome unbound"}
        return NodeResult(
            ok=bool(out.get("ok", True)),
            engine="mesie.connectome",
            action="connectome",
            output=out if isinstance(out, dict) else {"result": str(out)[:500]},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref="mesie.connectome.44regions",
        )

    def _match_validate(self, payload: Dict[str, Any], t0: float, action: str) -> NodeResult:
        rt = self._runtime()
        if rt is None:
            return NodeResult(
                ok=False,
                engine="mesie",
                action=action,
                output={"error": "runtime unbound"},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                evidence_ref="mesie:unbound",
            )
        if action == "validate" and "record" in payload:
            out = rt.validate_spectral(payload["record"])
        elif action == "match" and "reference" in payload and "candidate" in payload:
            out = rt.match_spectral(payload["reference"], payload["candidate"])
        else:
            # spectral self-similarity via embeddings
            a = str(payload.get("a") or payload.get("text") or "ref")
            b = str(payload.get("b") or payload.get("text2") or a)
            va = np.asarray(rt.embed_text(a, dim=64))
            vb = np.asarray(rt.embed_text(b, dim=64))
            sim = float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12))
            out = {"ok": True, "cosine": sim, "method": "embed_cosine"}
        return NodeResult(
            ok=bool(out.get("ok", True)),
            engine="mesie.match_validate",
            action=action,
            output=out if isinstance(out, dict) else {"result": str(out)[:500]},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref=f"mesie.{action}",
        )

    def _health(self, t0: float) -> NodeResult:
        rt = self._runtime()
        out = rt.health() if rt is not None else {"installed": False}
        return NodeResult(
            ok=True,
            engine="mesie_ghost_node",
            action="health",
            output=out if isinstance(out, dict) else {"raw": str(out)[:500]},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            evidence_ref="mesie_runtime.health",
        )

    def bus_dispatch(self, engine: str, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal bus entry used by mesie GhostAgent."""
        res = self.run(action or engine, payload)
        return {"ok": res.ok, "engine": engine, "action": action, **res.to_dict()}
