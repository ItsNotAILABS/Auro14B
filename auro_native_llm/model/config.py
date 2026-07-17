"""Auro language-model family configuration.

Executable architectures are real MESIE SpectralGPT causal MoE stacks.
``parameter_target`` records the family claim (2B…100B); ``num_params`` is
the live countable parameter mass of the built model (scales with mode).

Modes:
  - ``dev``  — MESIE spectral_gpt ladder (tiny→base), arsenal ON, laptop-executable
  - ``full`` — MESIE spectral_gpt small→xl scale (architecture targets; heavy)

Presets map 1:1 onto ``mesie.foundation`` SpectralGPT factories + pretraining
ModelConfig (rotary, SwiGLU, RMSNorm, MoE, cross-modal, spectral encoder, GQA).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

from auro_native_llm.model.phi_math import PHI

Mode = Literal["dev", "full"]

# MESIE SpectralGPT preset names (foundation factories / pretraining_config)
MesiePreset = Literal[
    "spectral_gpt_tiny",
    "spectral_gpt_small",
    "spectral_gpt_base",
    "spectral_gpt_large",
    "spectral_gpt_xl",
]


@dataclass
class AuroLMConfig:
    model_id: str = "Auro-2B"
    tier: str = "edge"
    parameter_target: int = 2_000_000_000
    mode: Mode = "dev"
    mesie_preset: str = "spectral_gpt_tiny"

    # ---- Transformer (MESIE SpectralGPT core) ----
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 1024
    vocab_size: int = 4096
    max_seq_len: int = 512

    # ---- MoE (Mixture of Experts) ----
    use_moe: bool = True
    num_experts: int = 8
    top_k_experts: int = 2
    moe_layers: Optional[List[int]] = None  # None → every 2nd layer
    moe_every: int = 2  # used when moe_layers is None

    # ---- Cross-modal + spectral encoder (MESIE arsenal) ----
    use_cross_modal: bool = True
    cross_modal_layers: Optional[List[int]] = None  # None → quartile defaults
    use_spectral_encoder: bool = True
    spectral_input_dim: int = 1024
    continuous_dim: int = 256
    num_modalities: int = 8  # text + 7 spectral domains

    # ---- Block geometry (MESIE TransformerBlock / Attention defaults) ----
    positional_encoding: str = "rotary"
    normalization: str = "rms_norm"
    activation: str = "swiglu"
    dropout: float = 0.0
    num_kv_heads: int = 0  # 0 = full MHA; >0 enables GQA
    qk_norm: bool = True
    tie_embeddings: bool = True
    causal: bool = True
    init_std: float = 0.02

    # ---- Multi-everything residual planes (Auro surface) ----
    use_meaning: bool = True
    use_spectral_fusion: bool = True
    use_helix: bool = True
    use_token_governor: bool = True
    multi_task: bool = True

    # ---- Math / init ----
    use_phi_init: bool = True
    seed: int = 42

    # ---- Training defaults ----
    learning_rate: float = 3e-3
    moe_aux_weight: float = 0.01
    meaning_blend: float = 0.15
    spectral_blend: float = 0.10

    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_moe_layers(self) -> List[int]:
        if self.moe_layers is not None:
            return list(self.moe_layers)
        step = max(1, int(self.moe_every))
        return list(range(1, self.num_layers, step))

    def resolved_cross_modal_layers(self) -> List[int]:
        if self.cross_modal_layers is not None:
            return list(self.cross_modal_layers)
        n = self.num_layers
        if n <= 1:
            return [0]
        pts = sorted(
            {
                max(0, n // 4),
                max(0, n // 2),
                max(0, (3 * n) // 4),
                max(0, n - 1),
            }
        )
        return pts

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["moe_layers_resolved"] = self.resolved_moe_layers()
        d["cross_modal_layers_resolved"] = self.resolved_cross_modal_layers()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuroLMConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# MESIE SpectralGPT factory dimensions (source of truth)
# Matches create_spectral_gpt_* + pretraining_config.spectral_gpt_*
# ---------------------------------------------------------------------------

_MESIE_PRESETS: Dict[str, Dict[str, Any]] = {
    "spectral_gpt_tiny": dict(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        head_dim=64,
        ffn_dim=1024,
        vocab_size=4096,
        max_seq_len=512,
        num_experts=8,
        top_k_experts=2,
        use_moe=True,  # arsenal ON (factory tiny disables MoE; Auro enables it)
        continuous_dim=64,
        spectral_input_dim=256,
    ),
    "spectral_gpt_small": dict(
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        head_dim=64,
        ffn_dim=2048,
        vocab_size=16384,
        max_seq_len=2048,
        num_experts=8,
        top_k_experts=2,
        use_moe=True,
        continuous_dim=128,
        spectral_input_dim=512,
    ),
    "spectral_gpt_base": dict(
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        head_dim=64,
        ffn_dim=4096,
        vocab_size=32768,
        max_seq_len=8192,
        num_experts=8,
        top_k_experts=2,
        use_moe=True,
        continuous_dim=256,
        spectral_input_dim=1024,
    ),
    "spectral_gpt_large": dict(
        hidden_dim=2048,
        num_layers=36,
        num_heads=32,
        head_dim=64,
        ffn_dim=8192,
        vocab_size=65536,
        max_seq_len=16384,
        num_experts=16,
        top_k_experts=4,
        use_moe=True,
        continuous_dim=512,
        spectral_input_dim=2048,
        num_kv_heads=8,  # GQA for large
    ),
    "spectral_gpt_xl": dict(
        hidden_dim=4096,
        num_layers=48,
        num_heads=64,
        head_dim=64,
        ffn_dim=16384,
        vocab_size=131072,
        max_seq_len=32768,
        num_experts=32,
        top_k_experts=4,
        use_moe=True,
        continuous_dim=1024,
        spectral_input_dim=4096,
        num_kv_heads=8,  # GQA for xl
    ),
}

# Mid-scale between small and base (executable "general" tier on laptop/dev)
_MESIE_MID: Dict[str, Any] = dict(
    hidden_dim=768,
    num_layers=16,
    num_heads=12,
    head_dim=64,
    ffn_dim=3072,
    vocab_size=24576,
    max_seq_len=4096,
    num_experts=12,
    top_k_experts=2,
    use_moe=True,
    continuous_dim=192,
    spectral_input_dim=768,
)


def mesie_preset_dims(preset: str) -> Dict[str, Any]:
    """Return MESIE SpectralGPT dimensions for a named preset."""
    if preset not in _MESIE_PRESETS:
        raise ValueError(f"unknown MESIE preset {preset}; choose from {list(_MESIE_PRESETS)}")
    return dict(_MESIE_PRESETS[preset])


def list_mesie_presets() -> List[str]:
    return list(_MESIE_PRESETS.keys())


def _try_import_mesie_model_config(preset: str) -> Optional[Dict[str, Any]]:
    """Live-sync dims from MESIE pretraining_config when importable."""
    try:
        from mesie.foundation.config import pretraining_config as pc

        factory = {
            "spectral_gpt_tiny": pc.spectral_gpt_tiny,
            "spectral_gpt_small": pc.spectral_gpt_small,
            "spectral_gpt_base": pc.spectral_gpt_base,
            "spectral_gpt_large": pc.spectral_gpt_large,
            "spectral_gpt_xl": pc.spectral_gpt_xl,
        }.get(preset)
        if factory is None:
            return None
        m = factory().model
        ffn = int(getattr(m.transformer_block.ffn, "hidden_dim", 0) or 0)
        if ffn <= 0:
            ffn = int(m.hidden_dim * 4)
        return {
            "hidden_dim": int(m.hidden_dim),
            "num_layers": int(m.num_layers),
            "num_heads": int(m.num_heads),
            "head_dim": int(m.head_dim),
            "ffn_dim": ffn,
            "vocab_size": int(m.vocab_size),
            "max_seq_len": int(m.max_seq_len),
            "num_experts": int(m.num_experts),
            "top_k_experts": int(m.top_k_experts),
            "use_moe": bool(m.use_mixture_of_experts),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Family ↔ MESIE mapping
# dev  = MESIE tiny / small / mid / base-lite  (executable, arsenal full)
# full = MESIE small / base / large / xl       (architecture targets)
# ---------------------------------------------------------------------------

_FAMILY_META: Dict[str, Dict[str, Any]] = {
    "Auro-2B": dict(tier="edge", parameter_target=2_000_000_000),
    "Auro-4B": dict(tier="specialist", parameter_target=4_000_000_000),
    "Auro-8B": dict(tier="general", parameter_target=8_000_000_000),
    "Auro-14B": dict(tier="orchestrator", parameter_target=14_000_000_000),
    "Auro-100B": dict(tier="frontier", parameter_target=100_000_000_000),
}

# (mesie_preset_name, optional dim overrides on top of preset)
_FAMILY_DEV: Dict[str, Dict[str, Any]] = {
    # Edge: MESIE tiny + MoE arsenal
    "Auro-2B": dict(mesie_preset="spectral_gpt_tiny", **_MESIE_PRESETS["spectral_gpt_tiny"]),
    # Specialist: MESIE small
    "Auro-4B": dict(mesie_preset="spectral_gpt_small", **_MESIE_PRESETS["spectral_gpt_small"]),
    # General: mid (between small and base)
    "Auro-8B": dict(mesie_preset="spectral_gpt_small", **_MESIE_MID),
    # Orchestrator: base-lite (half layers of base for local dev)
    "Auro-14B": dict(
        mesie_preset="spectral_gpt_base",
        hidden_dim=1024,
        num_layers=16,
        num_heads=16,
        head_dim=64,
        ffn_dim=4096,
        vocab_size=32768,
        max_seq_len=4096,
        num_experts=12,
        top_k_experts=2,
        use_moe=True,
        continuous_dim=256,
        spectral_input_dim=1024,
        num_kv_heads=4,
    ),
    # Frontier dev: full base geometry (MESIE spectral_gpt_base)
    "Auro-100B": dict(mesie_preset="spectral_gpt_base", **_MESIE_PRESETS["spectral_gpt_base"]),
}

_FAMILY_FULL: Dict[str, Dict[str, Any]] = {
    # Full ladder maps onto MESIE factory stack
    "Auro-2B": dict(mesie_preset="spectral_gpt_small", **_MESIE_PRESETS["spectral_gpt_small"]),
    "Auro-4B": dict(mesie_preset="spectral_gpt_base", **_MESIE_PRESETS["spectral_gpt_base"]),
    "Auro-8B": dict(
        mesie_preset="spectral_gpt_large",
        # large but slightly tempered seq for local instantiation
        **{
            **_MESIE_PRESETS["spectral_gpt_large"],
            "max_seq_len": 8192,
        },
    ),
    "Auro-14B": dict(mesie_preset="spectral_gpt_large", **_MESIE_PRESETS["spectral_gpt_large"]),
    "Auro-100B": dict(mesie_preset="spectral_gpt_xl", **_MESIE_PRESETS["spectral_gpt_xl"]),
}

# Shared arsenal defaults applied to every family member
_ARSENAL_DEFAULTS: Dict[str, Any] = dict(
    use_cross_modal=True,
    use_spectral_encoder=True,
    use_moe=True,
    positional_encoding="rotary",
    normalization="rms_norm",
    activation="swiglu",
    dropout=0.0,
    qk_norm=True,
    tie_embeddings=True,
    causal=True,
    num_modalities=8,
    multi_task=True,
    use_meaning=True,
    use_spectral_fusion=True,
    use_helix=True,
    use_token_governor=True,
    use_phi_init=True,
    moe_every=2,
)


def family_config(
    model_id: str = "Auro-2B",
    mode: Mode = "dev",
    *,
    sync_mesie: bool = False,
    **overrides: Any,
) -> AuroLMConfig:
    """Build AuroLMConfig for a family member from MESIE SpectralGPT scale.

    Parameters
    ----------
    model_id:
        One of Auro-2B / 4B / 8B / 14B / 100B.
    mode:
        ``dev`` (MESIE tiny→base ladder) or ``full`` (MESIE small→xl).
    sync_mesie:
        If True, overwrite dims from live ``pretraining_config`` factories.
    **overrides:
        Any AuroLMConfig field (hidden_dim, use_moe, …).
    """
    table = _FAMILY_DEV if mode == "dev" else _FAMILY_FULL
    if model_id not in table:
        raise ValueError(f"unknown model_id {model_id}; choose from {list(table)}")

    meta = dict(_FAMILY_META[model_id])
    base = {**_ARSENAL_DEFAULTS, **dict(table[model_id]), **meta}

    if sync_mesie and base.get("mesie_preset"):
        live = _try_import_mesie_model_config(str(base["mesie_preset"]))
        if live:
            # keep Auro arsenal flags; take geometry from MESIE when present
            for k, v in live.items():
                if k not in overrides:
                    base[k] = v
            # Auro always prefers MoE + cross-modal + spectral encoder on
            base["use_moe"] = True

    # Preserve MESIE geometry exactly (no φ-rounding that shrinks presets).
    # Only ensure ffn_dim is set if missing.
    if not base.get("ffn_dim"):
        base["ffn_dim"] = int(base["hidden_dim"] * 4)

    cfg = AuroLMConfig(model_id=model_id, mode=mode, **{
        k: v for k, v in base.items() if k in AuroLMConfig.__dataclass_fields__
    })
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            cfg.extra[k] = v
    # optional φ note in extra for docs only
    cfg.extra.setdefault("phi", PHI)
    cfg.extra.setdefault("mesie_geometry_locked", True)
    return cfg


def family_config_from_mesie(
    model_id: str,
    preset: MesiePreset,
    mode: Mode = "dev",
    **overrides: Any,
) -> AuroLMConfig:
    """Force a family member onto an explicit MESIE SpectralGPT preset."""
    dims = mesie_preset_dims(preset)
    meta = _FAMILY_META.get(model_id, dict(tier="custom", parameter_target=0))
    return family_config(
        model_id,
        mode=mode,
        mesie_preset=preset,
        **{**dims, **meta, **overrides},
    )


def all_family_ids() -> list[str]:
    return list(_FAMILY_DEV.keys())


def family_scale_table() -> Dict[str, Dict[str, Any]]:
    """Human-readable map: family → MESIE preset + key dims (dev + full)."""
    out: Dict[str, Dict[str, Any]] = {}
    for mid in all_family_ids():
        d = _FAMILY_DEV[mid]
        f = _FAMILY_FULL[mid]
        out[mid] = {
            "parameter_target": _FAMILY_META[mid]["parameter_target"],
            "tier": _FAMILY_META[mid]["tier"],
            "dev": {
                "mesie_preset": d.get("mesie_preset"),
                "hidden_dim": d["hidden_dim"],
                "num_layers": d["num_layers"],
                "num_experts": d["num_experts"],
                "ffn_dim": d["ffn_dim"],
            },
            "full": {
                "mesie_preset": f.get("mesie_preset"),
                "hidden_dim": f["hidden_dim"],
                "num_layers": f["num_layers"],
                "num_experts": f["num_experts"],
                "ffn_dim": f["ffn_dim"],
            },
        }
    return out
