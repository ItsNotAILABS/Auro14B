"""SOLUS Pattern Forge — local X-ray math caretaker (your Pattern Forge engine)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.sdk.solus.constants import LOCAL_ENGINE, PHI, SOLUS_BRAND
from mesie.sdk.solus.mini_brain import MiniBrain
from mesie.sdk.solus.mini_heart import MiniHeart


@dataclass
class PatternForgeReport:
    action: str
    ok: bool
    data: Dict[str, Any]
    heart: Dict[str, Any]
    brain: Dict[str, Any]
    sovereign: bool = True
    engine: str = LOCAL_ENGINE


class SolusPatternForge:
    """X-ray mathematics caretaker — spectral decomposition, z-depth, phi-harmonics. Fully local."""

    name = "Pattern Forge"
    caretaker_id = "solus-pattern-forge"

    def __init__(self) -> None:
        self.heart = MiniHeart(self.caretaker_id)
        self.brain = MiniBrain("pattern_xray")
        self.analysis_count = 0

    @staticmethod
    def _mean(arr: Sequence[float]) -> float:
        a = np.asarray(arr, dtype=np.float64)
        return float(np.mean(a)) if a.size else 0.0

    @staticmethod
    def _stddev(arr: Sequence[float]) -> float:
        a = np.asarray(arr, dtype=np.float64)
        return float(np.std(a, ddof=1)) if a.size > 1 else 0.0

    def xray_depth(self, values: Sequence[float]) -> Dict[str, Any]:
        self.analysis_count += 1
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 2:
            return {"error": "Need at least 2 values"}
        n = arr.size
        sqrt_n = math.sqrt(n)
        norm = np.linalg.norm(arr)
        normalized = arr / norm if norm > 1e-12 else arr
        mean = float(np.mean(normalized))
        sd = float(np.std(normalized, ddof=1)) if n > 1 else 0.0
        se = sd / sqrt_n if sqrt_n else sd
        depths = []
        signals = 0
        for i, v in enumerate(normalized):
            z = (v - mean) / se if se > 1e-12 else 0.0
            is_signal = abs(z) > PHI
            if is_signal:
                signals += 1
            depths.append({
                "index": i, "raw": round(float(arr[i]), 4), "normalized": round(float(v), 4),
                "z_score": round(z, 4), "phi_depth": round(abs(z) / PHI, 4),
                "is_signal": is_signal,
                "classification": "extreme" if abs(z) > 3 else "strong" if abs(z) > PHI * PHI else "signal" if is_signal else "noise",
            })
        return {
            "n": n, "sqrt_n": round(sqrt_n, 4), "signal_count": signals,
            "signal_ratio": round(signals / n, 4), "depths": depths, "engine": LOCAL_ENGINE,
        }

    def spectral_decompose(self, values: Sequence[float]) -> Dict[str, Any]:
        self.analysis_count += 1
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 4:
            return {"error": "Need at least 4 values"}
        centered = arr - np.mean(arr)
        n = arr.size
        half = n // 2
        mags = []
        for k in range(half + 1):
            t = np.arange(n)
            angle = 2 * np.pi * k * t / n
            re = float(np.sum(centered * np.cos(angle)))
            im = float(-np.sum(centered * np.sin(angle)))
            mag = math.sqrt(re * re + im * im) / n
            period = n / k if k > 0 else None
            mags.append({
                "frequency": k, "magnitude": round(mag, 4),
                "period": round(period, 2) if period else None,
                "phi_harmonic": round(period / PHI, 2) if period else None,
            })
        dominant = sorted(mags, key=lambda x: x["magnitude"], reverse=True)[:5]
        return {"n": n, "fundamental_frequencies": dominant, "engine": LOCAL_ENGINE}

    def cross_correlation(self, systems: Dict[str, Sequence[float]]) -> Dict[str, Any]:
        self.analysis_count += 1
        names = list(systems.keys())
        if len(names) < 2:
            return {"error": "Need at least 2 systems"}
        matrix: Dict[str, Dict[str, float]] = {}
        opportunities = []
        for i, a in enumerate(names):
            matrix[a] = {}
            for j, b in enumerate(names):
                if i == j:
                    matrix[a][b] = 1.0
                    continue
                x, y = np.asarray(systems[a]), np.asarray(systems[b])
                n = min(x.size, y.size)
                if n < 3:
                    r = 0.0
                else:
                    r = float(np.corrcoef(x[:n], y[:n])[0, 1])
                matrix[a][b] = round(r, 4)
                if i < j:
                    opportunities.append({"pair": [a, b], "correlation": round(r, 4), "phi_score": round(abs(1 - abs(r) / PHI), 4)})
        return {"matrix": matrix, "opportunities": opportunities, "engine": LOCAL_ENGINE}

    def detect_anomalies(self, values: Sequence[float]) -> Dict[str, Any]:
        self.analysis_count += 1
        arr = np.asarray(values, dtype=np.float64)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - PHI * iqr, q3 + PHI * iqr
        anomalies = [
            {"index": i, "value": float(v), "type": "low-outlier" if v < lower else "high-outlier"}
            for i, v in enumerate(arr) if v < lower or v > upper
        ]
        return {"q1": round(float(q1), 4), "q3": round(float(q3), 4), "anomaly_count": len(anomalies), "anomalies": anomalies, "engine": LOCAL_ENGINE}

    def caretaker_run(self, action: str, **kwargs: Any) -> PatternForgeReport:
        handlers = {
            "xray": lambda: self.xray_depth(kwargs.get("values", [])),
            "spectral": lambda: self.spectral_decompose(kwargs.get("values", [])),
            "correlate": lambda: self.cross_correlation(kwargs.get("systems", {})),
            "anomalies": lambda: self.detect_anomalies(kwargs.get("values", [])),
        }
        fn = handlers.get(action)
        if not fn:
            data = {"error": f"unknown action: {action}"}
            ok = False
        else:
            data = fn()
            ok = "error" not in data
        signal_ratio = data.get("signal_ratio", 0.5)
        vitals = self.heart.pulse(signal_ratio)
        thought = self.brain.reason({"score": signal_ratio, "metric": data.get("anomaly_count", 0)})
        return PatternForgeReport(
            action=action, ok=ok, data=data,
            heart={"bpm": vitals.bpm, "coherence": vitals.coherence, "sdk_health": vitals.sdk_health, **self.heart.to_dict()},
            brain={"conclusion": thought.conclusion, "confidence": thought.confidence, "evidence": thought.evidence},
        )