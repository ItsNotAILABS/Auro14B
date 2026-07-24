"""Multi-repo MESIE/Auro corpus — harvest all your GitHub + local monorepos."""

from auro_native_llm.corpus.harvest import (
    CorpusDocument,
    CorpusIndex,
    default_roots,
    harvest_all,
    harvest_paths,
    list_github_org_repos,
    materialize_github_org,
)
from auro_native_llm.corpus.bridge import collect_corpus_texts, collect_work_corpus
from auro_native_llm.corpus.embeddings import MaxEmbedder
from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

try:
    from auro_native_llm.succotash.corpus import (
        TRAINING_AREAS,
        collect_all_area_texts,
        harvest_succotash_corpus,
    )
except Exception:  # pragma: no cover
    TRAINING_AREAS = {}
    collect_all_area_texts = None  # type: ignore
    harvest_succotash_corpus = None  # type: ignore

# Lazy: continual imported by callers to avoid runpy double-load warning
def run_continual_training(*args, **kwargs):
    from auro_native_llm.corpus.continual import run_continual_training as _run

    return _run(*args, **kwargs)


def ContinualConfig(*args, **kwargs):
    from auro_native_llm.corpus.continual import ContinualConfig as _Cfg

    return _Cfg(*args, **kwargs)


__all__ = [
    "ContinualConfig",
    "CorpusDocument",
    "CorpusIndex",
    "GitHubKnowledgeDB",
    "MaxEmbedder",
    "TRAINING_AREAS",
    "collect_all_area_texts",
    "collect_corpus_texts",
    "collect_work_corpus",
    "default_roots",
    "harvest_all",
    "harvest_paths",
    "harvest_succotash_corpus",
    "list_github_org_repos",
    "materialize_github_org",
    "run_continual_training",
]
