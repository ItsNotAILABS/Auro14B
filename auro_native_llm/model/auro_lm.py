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
        self.built_at = time.time()

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

        outputs["spectral_fused"] = spectral_fused
        outputs["compute_plane"] = "MESIE"
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
        """SGD step on embedding table + LM head (real gradients).

        Deeper MESIE SpectralGPT stack participates in forward; embedding and
        output head receive explicit CE gradients so the model *learns*.
        """
        lr = float(lr if lr is not None else self.config.learning_rate)
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]
        labs = np.asarray(label_ids, dtype=np.int64)
        if labs.ndim == 1:
            labs = labs[np.newaxis, :]

        out = self.forward_ids(ids, text_for_meaning=text_for_meaning)
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

        # dL/dlogits
        dlogits = probs.reshape(-1, V)
        rows = np.arange(dlogits.shape[0])
        dlogits[rows, y.reshape(-1)] -= 1.0
        dlogits /= max(B * T, 1)
        dlogits = dlogits.reshape(B, T, V)

        # LM head: logits = h @ W  => dW = h.T @ dlogits, dh = dlogits @ W.T
        h2 = h.reshape(-1, h.shape[-1])
        dl2 = dlogits.reshape(-1, V)
        dW = h2.T @ dl2
        # tied embeddings: W is embedding.T so update embedding
        emb = self.core.embedding.token_embeddings  # [V, D]
        # d_emb from tied head: dW is [D, V] => d_emb = dW.T
        d_emb_head = dW.T

        # embedding input gradient path: scatter dh into token rows
        dh = dl2 @ self.core.lm_head_weight.T  # [B*T, D]
        dh = dh.reshape(B, T, -1)
        d_emb_in = np.zeros_like(emb)
        tok = ids[:, :T]
        for b in range(B):
            for t in range(T):
                tid = int(np.clip(tok[b, t], 0, emb.shape[0] - 1))
                d_emb_in[tid] += dh[b, t]

        # update
        emb_update = d_emb_head + d_emb_in
        self.core.embedding.token_embeddings = emb - lr * emb_update
        if self.core.tie_embeddings:
            self.core.lm_head_weight = self.core.embedding.token_embeddings.T
        else:
            self.core.lm_head_weight = self.core.lm_head_weight - lr * dW

        # light meaning inject training
        if self.meaning is not None and text_for_meaning:
            self.meaning.inject -= lr * 0.01 * phi_init(
                self.meaning.inject.shape, seed=self.train_steps, layer=2
            )

        self.train_steps += 1
        metrics = self.loss_on_batch(ids, labs, text_for_meaning=text_for_meaning)
        metrics["lr"] = lr
        metrics["step"] = float(self.train_steps)
        return metrics

    # --------------------------------------------------------------- generate
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
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) >= self.config.max_seq_len - 1:
            ids = ids[-(self.config.max_seq_len - max_new_tokens - 1) :]

        generated = list(ids)
        last_moe = 0.0
        meaning_hits: List[Dict[str, object]] = []
        spectral_fused = False

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
            logits = out["logits"][0, -1, :].astype(np.float64)
            from auro_native_llm.work.algorithms import sample_logits

            next_id = sample_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.12,
                recent_ids=generated[-64:],
                ban_ids=[self.tokenizer.pad_id],
            )
            generated.append(next_id)
            if next_id == self.tokenizer.eos_id:
                break

        text = self.tokenizer.decode(generated)
        # Prefer only the continuation after prompt when decode is noisy
        full = self.tokenizer.decode(generated, skip_special=True)
        return AuroGenerateResult(
            model_id=self.model_id,
            text=full,
            token_ids=generated,
            prompt=prompt,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meaning_hits=meaning_hits,
            spectral_fused=spectral_fused,
            moe_loss=last_moe,
            num_params=self.num_params,
            metadata={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "train_steps": self.train_steps,
                "tier": self.config.tier,
                "parameter_target": self.config.parameter_target,
                "mode": self.config.mode,
            },
        )

    def embed_text(self, text: str) -> np.ndarray:
        ids = np.array([self.tokenizer.encode(text)], dtype=np.int64)
        out = self.forward_ids(ids, text_for_meaning=text)
        h = out["last_hidden_state"][0].mean(axis=0)
        n = float(np.linalg.norm(h)) or 1.0
        return h / n
