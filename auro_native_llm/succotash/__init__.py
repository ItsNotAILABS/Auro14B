"""FreddyCreates/potential-succotash — primary engines, models, agents & training corpus.

Source of truth:
  https://github.com/FreddyCreates/potential-succotash

This package:
  - Locates / materializes the monorepo
  - Loads AI model families, multimodal families, engines, agents, protocols,
    extensions, marketplace, architectural laws, frontier register
  - Exposes engines + models the main Auro LLM can route to
  - Harvests multi-area training corpus (docs, words, agents, SDKs, research…)
"""

from auro_native_llm.succotash.paths import (
    SUCCOTASH_REPO,
    SUCCOTASH_URL,
    ensure_succotash,
    succotash_root,
)
from auro_native_llm.succotash.registry import (
    SuccotashRegistry,
    load_registry,
)
from auro_native_llm.succotash.corpus import (
    TRAINING_AREAS,
    harvest_succotash_corpus,
    collect_area_texts,
    collect_all_area_texts,
)
from auro_native_llm.succotash.router import (
    EngineModelRouter,
    route_task,
)

__all__ = [
    "SUCCOTASH_REPO",
    "SUCCOTASH_URL",
    "TRAINING_AREAS",
    "EngineModelRouter",
    "SuccotashRegistry",
    "collect_all_area_texts",
    "collect_area_texts",
    "ensure_succotash",
    "harvest_succotash_corpus",
    "load_registry",
    "route_task",
    "succotash_root",
]
