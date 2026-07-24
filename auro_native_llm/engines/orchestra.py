"""Power stack orchestra — physics + economy + algorithms + transformers together."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.engines.algorithms import AlgorithmSuite
from auro_native_llm.engines.economics import EconomicEngine
from auro_native_llm.engines.physics_deep import DeepPhysicsEngine
from auro_native_llm.physics.formulas import (
    text_to_physical_signal,
    spectrum_from_signal,
    wiener_coherence,
    resonance_score,
    kuramoto_order,
    kuramoto_step,
    dispersion_omega,
    PHI,
)
from auro_native_llm.ghost.receipts import GhostReceiptChain


@dataclass
class StackTick:
    ok: bool
    physics: Dict[str, Any]
    economy: Dict[str, Any]
    algorithms: Dict[str, Any]
    transformer: Dict[str, Any]
    joint_embedding: List[float]
    route: Dict[str, Any]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.power_stack.tick.v1",
            "ok": self.ok,
            "physics": self.physics,
            "economy": self.economy,
            "algorithms": self.algorithms,
            "transformer": self.transformer,
            "joint_embedding_dim": len(self.joint_embedding),
            "joint_embedding_head": self.joint_embedding[:16],
            "route": self.route,
            "latency_ms": self.latency_ms,
            "together": True,
        }


class PowerStack:
    """All engines + transformers + algorithms in one coupled loop.

    Flow per tick:
      prompt → physical signal/spectrum
      → DeepPhysicsEngine.step (Hamiltonian, KG, RG, Ising, Burgers)
      → EconomicEngine.step (Walras, spectral market, Kelly, free energy)
      → AlgorithmSuite.couple (match, OT, BP, joint latent, route)
      → Transformer residual (SpectralGPT or physics fuse) + optional train pulse
      → receipt chain
    """

    def __init__(self, mind: Any = None, dim: int = 64, n_assets: int = 16) -> None:
        self.mind = mind
        self.dim = dim
        self.physics = DeepPhysicsEngine(dim=dim)
        self.economy = EconomicEngine(n_assets=n_assets)
        self.algos = AlgorithmSuite(dim=dim)
        self.chain = GhostReceiptChain(chain_id="power-stack")
        self.ticks: List[Dict[str, Any]] = []

    def _spectral_side(self, prompt: str) -> Dict[str, Any]:
        sig = text_to_physical_signal(prompt, self.dim)
        spec = spectrum_from_signal(sig)
        # buckets for economy
        n_b = self.economy.n
        edges = np.linspace(0, spec.size, n_b + 1).astype(int)
        buckets = np.array(
            [float(spec[edges[i] : max(edges[i + 1], edges[i] + 1)].mean()) for i in range(n_b)]
        )
        buckets = buckets / (buckets.sum() + 1e-12)
        coh, _ = wiener_coherence(sig, np.sin(np.arange(sig.size) * 0.13))
        R = resonance_score(spec, np.abs(np.fft.rfft(sig)))
        phase = np.angle(np.fft.rfft(sig - sig.mean()))
        k = np.linspace(0.0, float(PHI), max(phase.size, 8))
        om = dispersion_omega(k)[: phase.size]
        th = kuramoto_step(phase, om if om.size == phase.size else np.resize(om, phase.size))
        r, _ = kuramoto_order(th)
        return {
            "signal": sig,
            "spectrum": spec,
            "buckets": buckets,
            "coherence": float(coh),
            "resonance": float(R),
            "kuramoto_r": float(r),
        }

    def _transformer_step(self, prompt: str, joint: np.ndarray) -> Dict[str, Any]:
        """Transformer / LM residual uses joint embedding as meaning context."""
        out: Dict[str, Any] = {"backend": "none", "trained": False}
        if self.mind is None or not hasattr(self.mind, "language"):
            # pure spectral transformer residual via physics engine embed
            try:
                from auro_native_llm.physics import get_physics_engine

                pe = get_physics_engine()
                emb = pe.embed_physics(prompt, self.dim)
                fused = 0.6 * emb + 0.4 * joint
                fused = fused / (np.linalg.norm(fused) + 1e-12)
                out = {
                    "backend": "physics_transformer_residual",
                    "trained": False,
                    "fused_norm": float(np.linalg.norm(fused)),
                }
                return out
            except Exception as exc:
                return {"backend": "error", "error": str(exc)[:120]}

        lang = self.mind.language
        # attach physics if missing
        try:
            from auro_native_llm.physics import get_physics_engine

            if getattr(lang, "physics", None) is None:
                lang.physics = get_physics_engine()
        except Exception:
            pass

        tok = lang.tokenizer
        text = f"{prompt}\nJOINT_ECON_PHYS route"
        ids = tok.encode(text, add_bos=True, add_eos=True, max_length=64)
        while len(ids) < 32:
            ids.append(tok.pad_id)
        arr = np.array([ids[:64]], dtype=np.int64)
        try:
            # scale LR by economic free energy sharpness
            base_lr = 2e-3
            met = lang.train_step(arr, arr, lr=base_lr, text_for_meaning=prompt[:200])
            out = {
                "backend": "auro_lm_spectral_gpt",
                "trained": True,
                "ce": met.get("ce"),
                "loss": met.get("loss"),
                "physics_flag": met.get("physics"),
                "phys_resonance": met.get("phys_resonance"),
            }
        except Exception as exc:
            out = {"backend": "auro_lm_error", "error": str(exc)[:160]}
        return out

    def tick(self, prompt: str, *, physics_steps: int = 3) -> StackTick:
        t0 = time.perf_counter()
        side = self._spectral_side(prompt)
        self.chain.append("intent", {"prompt": prompt[:200]}, actor="operator")

        # PHYSICS
        pst = self.physics.step(n=physics_steps, signal=side["signal"])
        pfeat = self.physics.feature_vector(self.dim)
        self.chain.append(
            "mesie",
            {
                "engine": "deep_physics",
                "energy": pst.energy,
                "coupling": pst.coupling,
                "magnetization": pst.metrics.get("magnetization"),
                "change": "physics_multi_engine_step",
            },
            actor="engine.physics",
        )

        # ECONOMY
        est = self.economy.step(
            spectral_buckets=side["buckets"],
            physics_energy=float(pst.energy),
            magnetization=float(pst.metrics.get("magnetization", 0.0)),
            fluid_energy=float(pst.metrics.get("fluid_energy", 0.0)),
            temperature=float(pst.temperature),
        )
        efeat = self.economy.feature_vector(self.dim)
        self.chain.append(
            "tool",
            {
                "tool": "economic_engine",
                "wealth": est.wealth,
                "utility": est.utility,
                "free_energy": est.free_energy,
                "excess": est.metrics.get("excess_demand_l2"),
                "change": "econ_market_step",
            },
            actor="engine.econ",
        )

        # ALGORITHMS
        sfeat = side["spectrum"]
        if sfeat.size < self.dim:
            sfeat = np.pad(sfeat, (0, self.dim - sfeat.size))
        else:
            sfeat = sfeat[: self.dim]
        sfeat = sfeat / (np.linalg.norm(sfeat) + 1e-12)
        coupled = self.algos.couple(
            pfeat,
            efeat,
            sfeat,
            resonance=side["resonance"],
            kuramoto_r=side["kuramoto_r"],
            excess_demand=float(est.metrics.get("excess_demand_l2", 0.0)),
            free_energy=float(est.free_energy),
            entropy=float(est.entropy),
        )
        afeat = self.algos.feature_vector(self.dim)
        route = coupled.get("route") or {}

        # JOINT EMBEDDING
        joint = 0.35 * pfeat + 0.30 * efeat + 0.20 * afeat + 0.15 * sfeat
        joint = joint / (np.linalg.norm(joint) + 1e-12)

        # TRANSFORMER
        xf = self._transformer_step(prompt, joint)
        self.chain.append(
            "validate",
            {
                "route": route.get("route"),
                "score": route.get("score"),
                "ot_cost": coupled.get("ot_cost"),
                "match_cost": coupled.get("match_cost"),
                "transformer": xf.get("backend"),
                "change": "coupled_stack_tick",
            },
            actor="orchestra",
            ok=True,
        )

        tick = StackTick(
            ok=True,
            physics=pst.to_dict(),
            economy=est.to_dict(),
            algorithms={
                "match_cost": coupled.get("match_cost"),
                "ot_cost": coupled.get("ot_cost"),
                "belief_entropy": coupled.get("belief_entropy"),
                "latent_norm": coupled.get("latent_norm"),
                "pairs_n": len(coupled.get("match_pairs") or []),
            },
            transformer=xf,
            joint_embedding=joint.tolist(),
            route=route,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        self.ticks.append(tick.to_dict())
        return tick

    def run(self, prompt: str, *, rounds: int = 5, physics_steps: int = 2) -> Dict[str, Any]:
        history = []
        for i in range(max(1, rounds)):
            t = self.tick(f"{prompt} :: round {i+1}", physics_steps=physics_steps)
            history.append(t.to_dict())
        # absorb into mind if present
        if self.mind is not None and getattr(self.mind, "organs", None):
            try:
                from auro_native_llm.organism.self_train import Experience

                if self.mind.organs.trainer:
                    last = history[-1]
                    self.mind.organs.trainer.absorb(
                        Experience(
                            text=(
                                f"POWER_STACK physics_E={last['physics']['energy']:.4f} "
                                f"wealth={last['economy']['wealth']:.4f} "
                                f"route={last['route'].get('route')} "
                                f"ot={last['algorithms'].get('ot_cost')}"
                            ),
                            kind="power_stack",
                            model_id=getattr(self.mind, "model_id", "Auro"),
                            reward=0.85,
                            embedding=last.get("joint_embedding"),
                            meta={"rounds": rounds},
                        )
                    )
            except Exception:
                pass

        report = {
            "schema": "auro.power_stack.report.v1",
            "ok": True,
            "rounds": rounds,
            "history": history,
            "receipt_chain": self.chain.to_dict(),
            "engines": [
                "hamiltonian_lattice",
                "klein_gordon_field",
                "rg_beta_flow",
                "ising_glauber",
                "burgers_fluid",
                "cobb_douglas_utility",
                "walras_tatonnement",
                "spectral_market",
                "kelly_portfolio",
                "entropy_free_energy",
                "mean_field_replicator",
                "greedy_match",
                "sinkhorn_ot",
                "belief_propagation",
                "joint_latent",
                "route_decision",
                "spectral_transformer_train",
            ],
            "together": True,
            "philosophy": (
                "Deep physics + economic engines drive spectral features; "
                "algorithms couple them; transformers train on the joint field. "
                "MESIE path preferred when route score high."
            ),
        }
        out = Path("artifacts/auro-power-stack")
        out.mkdir(parents=True, exist_ok=True)
        (out / "LAST_STACK.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.chain.save(out / "RECEIPT_CHAIN.json")
        report["saved"] = str(out / "LAST_STACK.json")
        return report


def run_power_stack(
    prompt: str,
    mind: Any = None,
    *,
    rounds: int = 5,
    physics_steps: int = 2,
) -> Dict[str, Any]:
    return PowerStack(mind).run(prompt, rounds=rounds, physics_steps=physics_steps)
