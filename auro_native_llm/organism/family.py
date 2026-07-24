"""Family factory — every Auro size is a full AuroMind with all organs."""

from __future__ import annotations

from typing import Any, Dict, List

from auro_native_llm.organism.mind import AuroMind

FAMILY_IDS: List[str] = ["Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"]


def build_mind(
    model_id: str = "Auro-2B",
    *,
    mode: str = "dev",
    lite: bool = True,
    chrome_mock: bool = True,
    **overrides: Any,
) -> AuroMind:
    if model_id not in FAMILY_IDS:
        raise ValueError(f"unknown model_id {model_id}; choose from {FAMILY_IDS}")
    return AuroMind.build(
        model_id, mode=mode, lite=lite, chrome_mock=chrome_mock, **overrides
    )


def build_family(
    *,
    mode: str = "dev",
    lite: bool = True,
    chrome_mock: bool = True,
    ids: List[str] | None = None,
) -> Dict[str, AuroMind]:
    """Instantiate every family member as a complete embedded mind.

    Each mind gets: language · doctrine · constitutional · memory ·
    continuous self-train · work · chrome · spectral/meaning.
    """
    out: Dict[str, AuroMind] = {}
    for mid in ids or FAMILY_IDS:
        out[mid] = build_mind(mid, mode=mode, lite=lite, chrome_mock=chrome_mock)
    return out


def family_manifest(family: Dict[str, AuroMind]) -> Dict[str, Any]:
    return {
        "schema": "auro.organism.family_manifest.v1",
        "count": len(family),
        "models": {mid: m.info() for mid, m in family.items()},
        "every_model_has_full_organs": all(
            all(m.organs.manifest().values())
            or all(
                m.organs.manifest().get(k)
                for k in (
                    "language",
                    "canon",
                    "constitutional",
                    "memory",
                    "trainer",
                    "rules",
                    "chrome",
                )
            )
            for m in family.values()
        ),
        "always_training": True,
        "compute_plane": "MESIE",
    }
