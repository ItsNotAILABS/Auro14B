"""Auro organism — each model is a complete embedded mind, not a feature stack.

Every family member (2B–100B) embeds:
  language core · doctrine · constitutional critique · memory ·
  continuous self-training · work/tools · chrome DOM · spectral/meaning

Nothing is an optional product feature. Subsystems are organs of the mind.
Every act is absorbed into training — the mind always trains itself.
"""

from auro_native_llm.organism.mind import AuroMind, MindResult
from auro_native_llm.organism.family import build_family, build_mind, FAMILY_IDS
from auro_native_llm.organism.self_train import ContinuousMindTrainer, Experience
from auro_native_llm.organism.checkpoint import save_mind, load_mind

# value_train is heavy; import via auro_native_llm.organism.value_train or CLI
# (avoid circular import warning when running python -m ...value_train)

__all__ = [
    "AuroMind",
    "MindResult",
    "ContinuousMindTrainer",
    "Experience",
    "FAMILY_IDS",
    "build_family",
    "build_mind",
    "load_mind",
    "save_mind",
]
