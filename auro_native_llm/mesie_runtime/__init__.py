"""Installed MESIE package → Auro 14B family runtime bridge.

Uses the user's pip-installed ``mesie`` (spectral, transformers, helix,
intelligence, connectome, pretraining, miniverse nesting) as the live
capability plane for every Auro model lane.
"""

from auro_native_llm.mesie_runtime.stack import (
    MesieRuntimeStack,
    attach_mesie_runtime,
    get_mesie_runtime,
    probe_mesie_install,
)

__all__ = [
    "MesieRuntimeStack",
    "attach_mesie_runtime",
    "get_mesie_runtime",
    "probe_mesie_install",
]
