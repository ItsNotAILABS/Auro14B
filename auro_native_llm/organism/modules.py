"""Embedded organs of an Auro mind — always present, never optional features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EmbeddedOrgans:
    """All subsystems live here as organs of one model instance.

    Construction failures degrade gracefully but organs remain addressable.
    """

    language: Any = None  # AuroLanguageModel core
    canon: Any = None
    constitutional: Any = None
    memory: Any = None
    trainer: Any = None  # ContinuousMindTrainer
    rules: Any = None
    process: Any = None
    governance: Any = None
    hooks: Any = None
    work: Any = None  # lazy WorkAgent binding to same language
    chrome: Any = None  # ChromeToolbelt
    meaning: Any = None
    spectral: Any = None
    monaco: Any = None
    jupyter: Any = None
    search: Any = None
    mcp: Any = None
    curriculum: Any = None
    # FreddyCreates/potential-succotash — engines + models the LLM can use
    succotash: Any = None  # EngineModelRouter
    engines: Any = None  # alias list/dict of engines
    model_catalogue: Any = None  # models for routing
    python: Any = None  # PythonOrgan — embedded compute for LLM
    mesie_power: Any = None  # multi-embed + compress stack
    polyglot: Any = None  # Python + Julia + Haskell + CUDA plane
    brains: Any = None  # MiniBrainCluster code/research/math + heart
    portal: Any = None  # Interior MCP portal
    fleet: Any = None  # MultiSiteFleet concurrent browsers
    agent_manager: Any = None  # Internal multi-agent manager
    neuro: Any = None  # NeuroEmergence bridge ref

    def manifest(self) -> Dict[str, bool]:
        return {
            "language": self.language is not None,
            "canon": self.canon is not None,
            "constitutional": self.constitutional is not None,
            "memory": self.memory is not None,
            "trainer": self.trainer is not None,
            "rules": self.rules is not None,
            "process": self.process is not None,
            "governance": self.governance is not None,
            "hooks": self.hooks is not None,
            "work": self.work is not None,
            "chrome": self.chrome is not None,
            "meaning": self.meaning is not None,
            "spectral": self.spectral is not None,
            "monaco": self.monaco is not None,
            "jupyter": self.jupyter is not None,
            "search": self.search is not None,
            "mcp": self.mcp is not None,
            "curriculum": self.curriculum is not None,
            "succotash": self.succotash is not None,
            "engines": self.engines is not None,
            "model_catalogue": self.model_catalogue is not None,
            "python": self.python is not None,
            "mesie_power": self.mesie_power is not None,
            "polyglot": self.polyglot is not None,
            "brains": self.brains is not None,
            "portal": self.portal is not None,
            "fleet": self.fleet is not None,
            "agent_manager": self.agent_manager is not None,
            "neuro": self.neuro is not None,
        }


def build_organs(
    language: Any,
    *,
    chrome_mock: bool = True,
    lite_tools: bool = True,
) -> EmbeddedOrgans:
    """Assemble full organ set for one model — every mind gets everything."""
    organs = EmbeddedOrgans(language=language)
    model_id = getattr(language, "model_id", "Auro-2B")

    # Doctrine
    try:
        from auro_native_llm.scripture.canon import load_canon
        from auro_native_llm.scripture.constitutional import ConstitutionalEngine
        from auro_native_llm.scripture.memory import ScripturalMemory
        from auro_native_llm.scripture.rules_engine import RulesEngine
        from auro_native_llm.scripture.process_model import ProcessModel
        from auro_native_llm.scripture.governance import InnerGovernance
        from auro_native_llm.scripture.hooks import EnforcementHooks
        from auro_native_llm.scripture.gates import GateMachine

        organs.canon = load_canon()
        organs.constitutional = ConstitutionalEngine(organs.canon)
        mem_cfg = organs.canon.memory or {}
        organs.memory = ScripturalMemory(
            capacity=int(mem_cfg.get("capacity", 2048)),
            embed_dim=int(mem_cfg.get("embed_dim", 256)),
            decay=float(mem_cfg.get("decay", 0.98)),
        )
        organs.rules = RulesEngine.from_canon(organs.canon)
        organs.process = ProcessModel.from_canon(organs.canon)
        organs.governance = InnerGovernance(organs.canon)
        organs.hooks = EnforcementHooks(
            organs.rules,
            organs.process,
            organs.governance,
            GateMachine(organs.canon.gates),
            canon_id=organs.canon.canon_id,
        )
    except Exception:
        pass

    # Continuous self-training organ
    from auro_native_llm.organism.self_train import ContinuousMindTrainer

    organs.trainer = ContinuousMindTrainer(
        capacity=4096,
        batch_size=2,
        seq_len=min(64, getattr(language.config, "max_seq_len", 64)),
        lr=float(getattr(language.config, "learning_rate", 2e-3)),
        auto_steps_per_absorb=1,
        messy_mix=0.4,
    )
    if organs.canon is not None:
        seeds = [organs.canon.principle, organs.canon.claim_boundary]
        seeds.extend(a.text for a in organs.canon.articles[:12])
        seeds.extend(p.get("text", "") for p in (organs.canon.principles or []) if p.get("text"))
        organs.trainer.seed_doctrine([s for s in seeds if s])

    # Meaning / spectral already on language; mirror references
    organs.meaning = getattr(language, "meaning", None)
    organs.spectral = getattr(language, "spectral", None)

    # Chrome organ (always present; mock by default)
    try:
        from auro_native_llm.chrome.tools import ChromeToolbelt

        organs.chrome = ChromeToolbelt(mock=chrome_mock, auto_start=False)
    except Exception:
        organs.chrome = None

    # Work organ shares this mind's identity (lazy full WorkAgent optional)
    organs.work = {
        "model_id": model_id,
        "lite": lite_tools,
        "chrome_mock": chrome_mock,
    }

    # Monaco / Jupyter / Search / MCP / curriculum — always embedded
    try:
        from auro_native_llm.embedded.monaco import MonacoOrgan
        from auro_native_llm.embedded.jupyter import JupyterOrgan
        from auro_native_llm.embedded.search import SearchOrgan
        from auro_native_llm.embedded.mcp_hub import MCPOrgan
        from auro_native_llm.embedded.teach import ToolCurriculum

        organs.monaco = MonacoOrgan()
        organs.jupyter = JupyterOrgan()
        organs.search = SearchOrgan(offline=False)
        organs.mcp = MCPOrgan()
        organs.mcp.wire_from_mind_organs(
            monaco=organs.monaco,
            jupyter=organs.jupyter,
            search=organs.search,
            chrome=organs.chrome,
            mind_info=lambda: {
                "model_id": model_id,
                "organs": organs.manifest(),
                "compute_plane": "MESIE",
            },
        )
        organs.curriculum = ToolCurriculum()
        # Seed trainer with tool curriculum at birth (mind already knows the tools)
        if organs.trainer is not None and organs.curriculum is not None:
            for lesson in organs.curriculum.lessons():
                from auro_native_llm.organism.self_train import Experience

                organs.trainer.absorb(
                    Experience(
                        text=f"BIRTH_TEACH: {lesson}",
                        kind="teach",
                        model_id=model_id,
                        reward=0.88,
                        meta={"birth": True},
                    )
                )
    except Exception:
        pass

    # potential-succotash engines + models (main LLM catalogue + router)
    try:
        from auro_native_llm.succotash.router import EngineModelRouter

        router = EngineModelRouter.load(clone=True)
        organs.succotash = router
        organs.engines = router.list_engines()
        organs.model_catalogue = router.list_models()
        if organs.trainer is not None:
            from auro_native_llm.organism.self_train import Experience

            organs.trainer.absorb(
                Experience(
                    text=router.catalogue_prompt(),
                    kind="succotash_catalogue",
                    model_id=model_id,
                    reward=0.95,
                    meta={
                        "source": "FreddyCreates/potential-succotash",
                        "url": "https://github.com/FreddyCreates/potential-succotash",
                    },
                )
            )
    except Exception:
        pass

    # Python organ — LLM-usable sandboxed compute + doctrine
    try:
        from auro_native_llm.embedded.python_organ import PythonOrgan

        organs.python = PythonOrgan(enable_github=True)
        if organs.trainer is not None:
            from auro_native_llm.organism.self_train import Experience

            organs.trainer.absorb(
                Experience(
                    text=organs.python.doctrine_prompt(),
                    kind="python_doctrine",
                    model_id=model_id,
                    reward=0.95,
                    meta={"doctrine_id": organs.python.doctrine.get("doctrine_id")},
                )
            )
    except Exception:
        pass

    # Polyglot organ — Julia + Haskell + Python + CUDA plane
    try:
        from auro_native_llm.polyglot.organ import PolyglotOrgan

        organs.polyglot = PolyglotOrgan()
        if organs.trainer is not None:
            from auro_native_llm.organism.self_train import Experience

            st = organs.polyglot.info()
            organs.trainer.absorb(
                Experience(
                    text=(
                        "POLYGLOT compute: python+julia+haskell+cuda_plane. "
                        f"status={st}"
                    )[:3000],
                    kind="polyglot_info",
                    model_id=model_id,
                    reward=0.92,
                    meta={"cuda_backend": st.get("cuda", {}).get("backend")},
                )
            )
    except Exception:
        pass

    # Mini brains + heart (BRAIN-AI + SOLUS) — code / research / math teachers
    try:
        from auro_native_llm.brain.organs import build_brain_cluster

        organs.brains = build_brain_cluster()
        if organs.trainer is not None:
            from auro_native_llm.organism.self_train import Experience

            for lesson in organs.brains.teacher.lesson_batch(3):
                organs.trainer.absorb(
                    Experience(
                        text=lesson,
                        kind="mini_brain_lesson",
                        model_id=model_id,
                        reward=0.9,
                        meta={"lineage": organs.brains.lineage},
                    )
                )
    except Exception:
        pass

    # Multi-repo SDK injection (all Medina / ItsNotAILABS / FreddyCreates trees)
    try:
        from auro_native_llm.sdk_runtime.injector import inject_repo_sdks

        # delayed: need language identity; call after organs attached via mind
        organs._sdk_inject_pending = True  # type: ignore[attr-defined]
    except Exception:
        pass

    return organs
