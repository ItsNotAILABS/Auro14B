"""MESIE power plane — multi-embedding, compression, powered training.

Uses everything available in MESIE to densify the autocycle:

  SENSE   → multi-embed + ANN/LSH retrieve over GitHub DB
  ABSORB  → compressed experience vectors (SVD / top-k / φ)
  TRAIN   → powered SpectralGPT core + multi-view CE
  GOVERN  → doctrine unchanged; receipts on compress ops
"""

from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder
from auro_native_llm.mesie_power.compress import MesieCompressor, CompressedBank
from auro_native_llm.mesie_power.power_train import (
    PowerTrainConfig,
    PowerProfile,
    power_family_overrides,
    run_power_train,
)
from auro_native_llm.mesie_power.stack import MesiePowerStack, get_power_stack

__all__ = [
    "CompressedBank",
    "MesieCompressor",
    "MesiePowerStack",
    "MultiMesieEmbedder",
    "PowerProfile",
    "PowerTrainConfig",
    "get_power_stack",
    "power_family_overrides",
    "run_power_train",
]
