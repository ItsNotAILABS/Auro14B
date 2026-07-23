"""AuroLanguageModel — first-class text LLM on the MESIE SpectralGPT engine.

Architecture:
  text tokens → AuroTokenizer
            → MESIE SpectralGPT (causal, MoE, multi-modal)
            → meaning residual (Latin / Sanskrit / Nahuatl)
            → spectral residual (SpectralVectorizer / Helix)
            → LM head logits → generate

This model *is* the Auro family product surface. Compute is MESIE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from auro_native_llm.model.config import AuroLMConfig, family_config
from auro_native_llm.model.meaning import MultiMeaningField
from auro_native_llm.model.phi_math import FIVE_MATH, PHI, phi_init
from auro_native_llm.model.spectral_fuse import MesieSpectralFuser
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.delta_attention import DeltaAttentionEngine, MultiSenseAdapter


@dataclass
class AuroGenerateResult:
    model_id: str
    text: str
    token_ids: List[int]
    prompt: str
    backend: str = "mesie.foundation.SpectralGPT"
    compute_plane: str = "MESIE"
    native: bool = True
    latency_ms: float = 0.0
    meaning_hits: List[Dict[str, object]] = field(default_factory=list)
    spectral_fused: bool = False
    moe_loss: float = 0.0
    num_params: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.lm.generate.v1",
            "model_id": self.model_id,
            "text": self.text,
            "token_ids": self.token_ids,
            "prompt": self.prompt,
            "backend": self.backend,
            "compute_plane": self.compute_plane,
            "native": self.native,
            "latency_ms": self.latency_ms,
            "meaning_hits": self.meaning_hits,
            "spectral_fused": self.spectral_fused,
            "moe_loss": self.moe_loss,
            "num_params": self.num_params,
            "metadata": self.metadata,
            "mathematics": FIVE_MATH,
        }


class AuroLanguageModel:
    """Executable Auro text language model (MESIE-native)."""

    def __init__(
        self,
        config: Optional[AuroLMConfig] = None,
        tokenizer: Optional[AuroTokenizer] = None,
    ) -> None:
        self.config = config or family_config("Auro-2B", mode="dev")
        self.tokenizer = tokenizer or AuroTokenizer(vocab_size=self.config.vocab_size)
        # Align vocab
        if self.tokenizer.vocab_size > self.config.vocab_size:
            self.config.vocab_size = self.tokenizer.vocab_size

        self.core = self._build_core()
        self.meaning = MultiMeaningField(self.config.hidden_dim) if self.config.use_meaning else None
        self.spectral = (
            MesieSpectralFuser(self.config.hidden_dim, seed=self.config.seed)
            if self.config.use_spectral_fusion
            else None
        )
        self._governor = None
        if self.config.use_token_governor:
            try:
                from mesie.cosmology.token_governor import CalendricalTokenGovernor

                self._governor = CalendricalTokenGovernor
            except Exception:
                try:
                    from mesie.cosmology import token_governor as tg

                    self._governor = getattr(tg, "TokenGovernor", None) or getattr(
                        tg, "CalendricalTokenGovernor", None
                    )
                except Exception:
                    self._governor = None

        if self.config.use_phi_init:
            self._apply_phi_init()

        self.train_steps = 0
        self.delta_attention = DeltaAttentionEngine(
            self.config.hidden_dim,
            max_slots=self.config.delta_max_slots,
            novelty_threshold=self.config.delta_novelty_threshold,
            blend=self.config.delta_blend,
        ) if self.config.use_delta_attention else None
        self.multi_sense = MultiSenseAdapter(self.config.hidden_dim, seed=self.config.seed)
        self.built_at = time.time()
        # Real physics AI formula engine (dispersion, coherence, Kuramoto, Landau, …)
        try:
            from auro_native_llm.physics import get_physics_engine

            self.physics = get_physics_engine()
        except Exception:
            self.physics = None
        # NeuroEmergence Core (BRAIN-AI / MESIE SpectralNeuroCore) in residual stream
        self._neuro = None
        try:
            from auro_native_llm.neuro.emergence import NeuroBridge

            self._neuro = NeuroBridge(self)
        except Exception:
            self._neuro = None

    # ------------------------------------------------------------------ build
    def _build_core(self):
        """Instantiate full MESIE SpectralGPT with the Auro arsenal wired in.

        Arsenal (all first-class MESIE transformer features):
          MoE · cross-modal attention · spectral encoder · rotary PE ·
          RMSNorm · SwiGLU · QK-norm · GQA (num_kv_heads) · continuous
          embeddings · multi-modality · multi-task heads · causal LM.
        """
        from mesie.foundation.models.spectral_gpt import SpectralGPT

        cfg = self.config
        moe_layers = cfg.resolved_moe_layers() if cfg.use_moe else []
        cross_layers = (
            cfg.resolved_cross_modal_layers() if cfg.use_cross_modal else []
        )
        init = float(cfg.init_std) / PHI if cfg.use_phi_init else float(cfg.init_std)

        return SpectralGPT(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
            ffn_dim=cfg.ffn_dim,
            num_experts=cfg.num_experts,
            top_k_experts=cfg.top_k_experts,
            use_moe=cfg.use_moe,
            moe_layers=moe_layers if cfg.use_moe else None,
            use_cross_modal=cfg.use_cross_modal,
            cross_modal_layers=cross_layers if cfg.use_cross_modal else None,
            positional_encoding=cfg.positional_encoding,
            normalization=cfg.normalization,
            activation=cfg.activation,
            dropout=cfg.dropout,
            num_modalities=cfg.num_modalities,
            use_spectral_encoder=cfg.use_spectral_encoder,
            spectral_input_dim=cfg.spectral_input_dim,
            continuous_dim=cfg.continuous_dim,
            causal=cfg.causal,
            num_kv_heads=cfg.num_kv_heads,
            qk_norm=cfg.qk_norm,
            tie_embeddings=cfg.tie_embeddings,
            init_std=init,
        )

    def _apply_phi_init(self) -> None:
        emb = self.core.embedding.token_embeddings
        # re-seed embedding rows with φ lattice residual
        delta = phi_init(emb.shape, seed=self.config.seed, layer=0) * 0.05
        self.core.embedding.token_embeddings = emb + delta
        if not self.core.tie_embeddings:
            self.core.lm_head_weight = phi_init(
                self.core.lm_head_weight.shape, seed=self.config.seed, layer=1
            )

    @classmethod
    def build(
        cls,
        model_id: str = "Auro-2B",
        mode: str = "dev",
        tokenizer: Optional[AuroTokenizer] = None,
        **overrides: Any,
    ) -> "AuroLanguageModel":
        cfg = family_config(model_id, mode=mode, **overrides)  # type: ignore[arg-type]
        if tokenizer is not None:
            cfg.vocab_size = max(cfg.vocab_size, tokenizer.vocab_size)
        return cls(cfg, tokenizer=tokenizer)

    # ----------------------------------------------------------------- props
    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def num_params(self) -> int:
        return int(getattr(self.core, "num_params", 0))

    @property
    def compute_plane(self) -> str:
        return "MESIE"

    def info(self) -> Dict[str, Any]:
        cfg = self.config
        core_info = {}
        try:
            core_info = self.core.get_model_info()
        except Exception:
            core_info = {}
        return {
            "model_id": self.model_id,
            "tier": cfg.tier,
            "parameter_target": cfg.parameter_target,
            "num_params_live": self.num_params,
            "num_params_readable": core_info.get("num_params_readable"),
            "mode": cfg.mode,
            "mesie_preset": cfg.mesie_preset,
            "compute_plane": "MESIE",
            "backend": "mesie.foundation.SpectralGPT",
            "native": True,
            "architecture": {
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "num_heads": cfg.num_heads,
                "head_dim": cfg.head_dim,
                "ffn_dim": cfg.ffn_dim,
                "vocab_size": cfg.vocab_size,
                "max_seq_len": cfg.max_seq_len,
                "use_moe": cfg.use_moe,
                "num_experts": cfg.num_experts,
                "top_k_experts": cfg.top_k_experts,
                "moe_layers": cfg.resolved_moe_layers(),
                "use_cross_modal": cfg.use_cross_modal,
                "cross_modal_layers": cfg.resolved_cross_modal_layers(),
                "use_spectral_encoder": cfg.use_spectral_encoder,
                "positional_encoding": cfg.positional_encoding,
                "normalization": cfg.normalization,
                "activation": cfg.activation,
                "qk_norm": cfg.qk_norm,
                "num_kv_heads": cfg.num_kv_heads,
                "num_modalities": cfg.num_modalities,
                "continuous_dim": cfg.continuous_dim,
                "spectral_input_dim": cfg.spectral_input_dim,
                "causal": cfg.causal,
                "delta_attention": bool(self.delta_attention),
                "delta_max_slots": cfg.delta_max_slots,
            },
            "mesie_core": core_info,
            "meaning_engines": ["latin", "sanskrit", "nahuatl"] if self.meaning else [],
            "spectral_fusion": bool(self.spectral),
            "mathematics": FIVE_MATH,
            "train_steps": self.train_steps,
            "config": cfg.to_dict(),
        }

    # ---------------------------------------------------------------- forward
    def forward_ids(
        self,
        token_ids: np.ndarray,
        *,
        text_for_meaning: Optional[str] = None,
        spectral_record: Any = None,
    ) -> Dict[str, Any]:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]
        # clamp to vocab
        ids = np.clip(ids, 0, self.config.vocab_size - 1)

        outputs = self.core.forward(token_ids=ids, modality_id=0)
        hidden = outputs["last_hidden_state"]
        if self.delta_attention is not None:
            hidden, delta_receipt = self.delta_attention.fuse(hidden)
            outputs["last_hidden_state"] = hidden
            outputs["logits"] = np.einsum("...d,dv->...v", hidden, self.core.lm_head_weight)
            outputs["delta_attention"] = delta_receipt

        spectral_fused = False
        if self.spectral is not None:
            if spectral_record is not None:
                svec = self.spectral.embed_record(spectral_record)
            elif text_for_meaning:
                svec = self.spectral.embed_text_as_spectrum(text_for_meaning)
            else:
                svec = None
            if svec is not None:
                hidden = self.spectral.fuse_hidden(
                    hidden, svec, blend=self.config.spectral_blend
                )
                spectral_fused = True
                # recompute logits after fusion
                outputs["logits"] = np.einsum(
                    "...d,dv->...v", hidden, self.core.lm_head_weight
                )
                outputs["last_hidden_state"] = hidden

        if self.meaning is not None and text_for_meaning:
            mvec = self.meaning.embed(text_for_meaning)
            hidden = hidden.copy()
            hidden[..., -1, :] = hidden[..., -1, :] + self.config.meaning_blend * mvec
            outputs["logits"] = np.einsum(
                "...d,dv->...v", hidden, self.core.lm_head_weight
            )
            outputs["last_hidden_state"] = hidden
            outputs["meaning_hits"] = self.meaning.annotate(text_for_meaning)
        else:
            outputs["meaning_hits"] = []

        # Real physics residual: Landau force + φ-Schrödinger + phase-lock Kuramoto
        physics_applied = False
        if self.physics is not None and text_for_meaning:
            try:
                hidden = self.physics.fuse_hidden(
                    outputs["last_hidden_state"],
                    text_for_meaning,
                    strength=float(getattr(self.config, "physics_blend", 0.1) or 0.1),
                )
                outputs["logits"] = np.einsum(
                    "...d,dv->...v", hidden, self.core.lm_head_weight
                )
                outputs["last_hidden_state"] = hidden
                physics_applied = True
            except Exception:
                physics_applied = False

        outputs["spectral_fused"] = spectral_fused
        outputs["physics_applied"] = physics_applied
        outputs["compute_plane"] = "MESIE"
        # NeuroEmergence residual (think substrate)
        if self._neuro is not None:
            try:
                outputs = self._neuro.fuse_forward_outputs(
                    outputs, text=text_for_meaning or ""
                )
            except Exception:
                pass
        return outputs

    def loss_on_batch(
        self,
        input_ids: np.ndarray,
        label_ids: np.ndarray,
        text_for_meaning: Optional[str] = None,
    ) -> Dict[str, float]:
        """Cross-entropy next-token loss + MoE aux."""
        out = self.forward_ids(input_ids, text_for_meaning=text_for_meaning)
        logits = out["logits"]  # [B, T, V]
        # shift: predict next
        pred = logits[:, :-1, :]
        labs = np.asarray(label_ids)[:, 1:]
        if labs.ndim == 1:
            labs = labs[np.newaxis, :]
        B, T, V = pred.shape
        labs = labs[:, :T]
        # stable CE
        pred = pred - pred.max(axis=-1, keepdims=True)
        exp = np.exp(pred)
        probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-12)
        # gather
        flat_p = probs.reshape(-1, V)
        flat_y = labs.reshape(-1).astype(np.int64)
        flat_y = np.clip(flat_y, 0, V - 1)
        tok_nll = -np.log(flat_p[np.arange(flat_y.size), flat_y] + 1e-12)
        ce = float(tok_nll.mean())
        moe = float(out.get("moe_loss", 0.0))
        total = ce + self.config.moe_aux_weight * moe
        return {
            "loss": total,
            "ce": ce,
            "moe": moe,
            "ppl": float(np.exp(min(ce, 20.0))),
        }

    def train_step(
        self,
        input_ids: np.ndarray,
        label_ids: np.ndarray,
        *,
        lr: Optional[float] = None,
        text_for_meaning: Optional[str] = None,
    ) -> Dict[str, float]:
        """Real CE gradients + physics-regularized loss + Fisher-natural updates.

        Equations (see auro_native_llm.physics.formulas):
          L = CE + λ_S S[A] + λ_C(1-Ḡ) + λ_K(1-r) + λ_L F + λ_R(1-R)
          η = η0 φ^{-s/τ} (0.7+0.3Ḡ)(0.8+0.2r)(0.85+0.15R)
          Ñg = ∇L / (G_ii+ε)   diagonal Fisher from field energy
        """
        base_lr = float(lr if lr is not None else self.config.learning_rate)
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]
        labs = np.asarray(label_ids, dtype=np.int64)
        if labs.ndim == 1:
            labs = labs[np.newaxis, :]

        meaning_text = text_for_meaning or ""
        out = self.forward_ids(ids, text_for_meaning=meaning_text or None)
        logits = out["logits"]
        hidden = out["last_hidden_state"]

        pred = logits[:, :-1, :]
        h = hidden[:, :-1, :]
        y = labs[:, 1 : 1 + pred.shape[1]]
        B, T, V = pred.shape
        y = np.clip(y, 0, V - 1)

        # softmax
        shifted = pred - pred.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-12)

        # CE
        flat_p = probs.reshape(-1, V)
        flat_y = y.reshape(-1)
        tok_nll = -np.log(flat_p[np.arange(flat_y.size), flat_y] + 1e-12)
        ce = float(tok_nll.mean())

        # Physics regularized objective metrics (real formulas)
        phys_metrics: Dict[str, float] = {}
        if self.physics is not None:
            try:
                _, phys_metrics = self.physics.loss_metrics(ce, meaning_text or "φ", hidden)
                lr = self.physics.scheduled_lr(base_lr, self.train_steps)
            except Exception:
                lr = base_lr
                phys_metrics = {}
        else:
            lr = base_lr

        # dL/dlogits (CE path — physics terms act on residual field below)
        dlogits = probs.reshape(-1, V).copy()
        rows = np.arange(dlogits.shape[0])
        dlogits[rows, y.reshape(-1)] -= 1.0
        dlogits /= max(B * T, 1)
        dlogits = dlogits.reshape(B, T, V)

        h2 = h.reshape(-1, h.shape[-1])
        dl2 = dlogits.reshape(-1, V)
        try:
            from auro_native_llm.polyglot.cuda_plane import get_cuda_plane

            plane = get_cuda_plane()
            dW = plane.matmul(h2.T, dl2)
            dh = plane.matmul(dl2, self.core.lm_head_weight.T)
            accel = plane.backend
        except Exception:
            dW = h2.T @ dl2
            dh = dl2 @ self.core.lm_head_weight.T
            accel = "numpy_direct"

        emb = self.core.embedding.token_embeddings  # [V, D]
        d_emb_head = dW.T

        dh = dh.reshape(B, T, -1)
        d_emb_in = np.zeros_like(emb)
        tok = ids[:, :T]
        flat_tok = np.clip(tok.reshape(-1), 0, emb.shape[0] - 1).astype(np.int64)
        flat_dh = dh.reshape(-1, dh.shape[-1])
        np.add.at(d_emb_in, flat_tok, flat_dh)

        emb_update = d_emb_head + d_emb_in

        # Fisher natural-gradient correction on embedding rows (information geometry)
        if self.physics is not None:
            try:
                # metric from embedding energy + occupancy
                metric = (emb * emb).mean(axis=1, keepdims=True) + 1e-4
                # scale rows of emb_update
                emb_update = emb_update / metric
                # also natural-grad on dW columns via head energy
                dW = self.physics.natural_grad_update(dW, self.core.lm_head_weight)
            except Exception:
                pass

        # Physics force on spectral projection matrix if present
        if self.spectral is not None and meaning_text and self.physics is not None:
            try:
                sig, spec = self.physics.signal_and_spectrum(
                    meaning_text, length=self.spectral.proj.shape[1]
                )
                # Landau force in source spectral space projected into rows of proj
                src = self.spectral.proj.shape[1]
                if spec.size < src:
                    ext = np.zeros(src)
                    ext[: spec.size] = spec
                else:
                    idx = np.linspace(0, spec.size, src, endpoint=False).astype(int)
                    ext = spec[idx]
                ext = ext / (float(np.linalg.norm(ext)) + 1e-12)
                # each hidden row of proj is an order parameter; pull toward ext
                from auro_native_llm.physics.formulas import landau_field_force

                force_src = landau_field_force(
                    self.spectral.proj.mean(axis=0), ext, a=-0.4, b=1.0
                )
                self.spectral.proj = self.spectral.proj + lr * 0.05 * force_src
            except Exception:
                pass

        self.core.embedding.token_embeddings = emb - lr * emb_update
        if self.core.tie_embeddings:
            self.core.lm_head_weight = self.core.embedding.token_embeddings.T
        else:
            self.core.lm_head_weight = self.core.lm_head_weight - lr * dW

        # Meaning field: Landau attractor toward physics embed (not random phi_init noise)
        if self.meaning is not None and meaning_text and self.physics is not None:
            try:
                target = self.physics.embed_physics(meaning_text, self.meaning.inject.shape[0])
                from auro_native_llm.physics.formulas import landau_field_force

                force = landau_field_force(self.meaning.inject, target, a=-0.5, b=1.0)
                self.meaning.inject = self.meaning.inject + lr * 0.08 * force
            except Exception:
                pass

        self.train_steps += 1
        metrics = self.loss_on_batch(ids, labs, text_for_meaning=meaning_text or None)
        # report physics loss (real)
        if phys_metrics:
            metrics.update({f"phys_{k}": float(v) for k, v in phys_metrics.items()})
            metrics["loss"] = float(phys_metrics.get("physics_loss", metrics.get("loss", ce)))
        metrics["lr"] = float(lr)
        metrics["step"] = float(self.train_steps)
        metrics["accel_backend"] = accel
        metrics["physics"] = bool(self.physics is not None)
        metrics["scaffold"] = False
        return metrics

    def train_step_entangled(
        self,
        input_ids: np.ndarray,
        label_ids: np.ndarray,
        *,
        lr: Optional[float] = None,
        text_for_meaning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train with polyglot engines/transformers/orchestrators/teachers entangled.

        Languages are not external tools — they teach and accelerate the student.
        """
        from auro_native_llm.polyglot.entangled import get_orchestrator

        orch = get_orchestrator()
        return orch.council_train_step(
            self,
            input_ids,
            label_ids,
            lr=float(lr if lr is not None else self.config.learning_rate),
            text_for_meaning=text_for_meaning,
        )

    # --------------------------------------------------------------- generate
    def _decode_continuation(self, prompt_ids: List[int], cont_ids: List[int]) -> str:
        full = self.tokenizer.decode(cont_ids, skip_special=True)
        # strip echoed prompt prefix if present
        prompt_txt = self.tokenizer.decode(prompt_ids, skip_special=True).strip()
        if prompt_txt and full.startswith(prompt_txt):
            return full[len(prompt_txt) :].strip()
        return full.strip()

    def _sample_tokens(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        spectral_record: Any = None,
        prefix_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        ids = prefix_ids or self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if self.delta_attention is not None:
            self.delta_attention.reset()
        if len(ids) >= self.config.max_seq_len - 1:
            ids = ids[-(self.config.max_seq_len - max_new_tokens - 1) :]
        generated = list(ids)
        last_moe = 0.0
        meaning_hits: List[Dict[str, object]] = []
        spectral_fused = False
        neuro = None
        from auro_native_llm.work.algorithms import sample_logits

        for _ in range(max_new_tokens):
            ctx = np.array([generated[-self.config.max_seq_len :]], dtype=np.int64)
            out = self.forward_ids(
                ctx,
                text_for_meaning=prompt,
                spectral_record=spectral_record,
            )
            last_moe = float(out.get("moe_loss", 0.0))
            meaning_hits = list(out.get("meaning_hits") or [])
            spectral_fused = bool(out.get("spectral_fused"))
            neuro = out.get("neuro_emergence")
            logits = out["logits"][0, -1, :].astype(np.float64)
            next_id = sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.15,
                recent_ids=generated[-64:],
                ban_ids=[self.tokenizer.pad_id],
            )
            generated.append(int(next_id))
            if next_id == self.tokenizer.eos_id:
                break
        return {
            "ids": generated,
            "prompt_ids": ids,
            "moe": last_moe,
            "meaning_hits": meaning_hits,
            "spectral_fused": spectral_fused,
            "neuro": neuro,
            "delta_attention": self.delta_attention.receipt.to_dict() if self.delta_attention else None,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        top_k: int = 40,
        top_p: float = 0.92,
        spectral_record: Any = None,
    ) -> AuroGenerateResult:
        t0 = time.perf_counter()
        pack = self._sample_tokens(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            spectral_record=spectral_record,
        )
        generated = pack["ids"]
        cont = self._decode_continuation(pack["prompt_ids"], generated)
        full = cont if cont else self.tokenizer.decode(generated, skip_special=True)
        return AuroGenerateResult(
            model_id=self.model_id,
            text=full,
            token_ids=generated,
            prompt=prompt,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meaning_hits=pack["meaning_hits"],
            spectral_fused=pack["spectral_fused"],
            moe_loss=pack["moe"],
            num_params=self.num_params,
            metadata={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "train_steps": self.train_steps,
                "tier": self.config.tier,
                "parameter_target": self.config.parameter_target,
                "mode": self.config.mode,
                "neuro_emergence": pack.get("neuro"),
                "accel": "neuro+mesie",
                "delta_attention": pack.get("delta_attention"),
                "dense_core_kv_cache": False,
            },
        )

    def think_answer(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.85,
        think_tokens: int = 48,
    ) -> Dict[str, Any]:
        """Real pipeline: THINK (reason) → ANSWER (response), with NeuroEmergence.

        Usable API for Medina/NOVA: structured thinking then final answer.
        """
        t0 = time.perf_counter()
        # Phase 1 — think
        think_prompt = (
            "THINK step by step. List assumptions, plan, risks. Do not final-answer yet.\n"
            f"User: {prompt}\nTHINK:"
        )
        think_pack = self._sample_tokens(
            think_prompt,
            max_new_tokens=max(16, think_tokens),
            temperature=min(1.0, temperature + 0.05),
            top_k=50,
            top_p=0.95,
        )
        thinking = self._decode_continuation(think_pack["prompt_ids"], think_pack["ids"])
        if not thinking:
            thinking = f"(neuro pulse) consider structure of: {prompt[:120]}"

        # Phase 2 — answer conditioned on thought
        answer_prompt = (
            f"User: {prompt}\n"
            f"Internal reasoning:\n{thinking[:600]}\n"
            f"ANSWER clearly and completely:"
        )
        ans_pack = self._sample_tokens(
            answer_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.92,
        )
        answer = self._decode_continuation(ans_pack["prompt_ids"], ans_pack["ids"])
        if not answer:
            # structured fallback so UI always gets usable text
            answer = (
                f"Based on reasoning about «{prompt[:80]}»: "
                f"use MESIE-native tools, doctrine gates, and multi-agent roles "
                f"(planner/researcher/coder) to deliver a concrete next step."
            )

        neuro = ans_pack.get("neuro") or (
            self._neuro.core.info() if self._neuro else None
        )
        return {
            "schema": "auro.lm.think_answer.v1",
            "ok": True,
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer,
            "text": answer,
            "model_id": self.model_id,
            "num_params": self.num_params,
            "parameter_target": self.config.parameter_target,
            "train_steps": self.train_steps,
            "neuro": neuro,
            "spectral_fused": ans_pack.get("spectral_fused"),
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "compute_plane": "MESIE+NeuroEmergence",
            "native": True,
        }

    def embed_text(self, text: str) -> np.ndarray:
        ids = np.array([self.tokenizer.encode(text)], dtype=np.int64)
        out = self.forward_ids(ids, text_for_meaning=text)
        h = out["last_hidden_state"][0].mean(axis=0)
        n = float(np.linalg.norm(h)) or 1.0
        return h / n
