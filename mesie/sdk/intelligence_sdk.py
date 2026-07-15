"""Unified SDK entrypoint for the MESIE Spectral Intelligence Engine — V0.3.

Provides a single high-level interface for ALL MESIE capabilities:
corpus loading, matching, generation, embeddings, validation,
**agentic task execution**, **multi-network dispatch**, **workflow embedding**,
and **reasoning with real physics/math/geometry training data**.

V0.3 additions:
- Ghost agents: spawn autonomous computational workers
- Multi-network: parallel agent topologies (star, mesh, pipeline, swarm)
- Core engine: inner nucleus tying all engines together
- Workflow & dataset embedding: native vector representation of workflows
- Reasoning datasets: real physics, math, geometry training data

Example:
    >>> from mesie.sdk import SpectralIntelligenceSDK
    >>> engine = SpectralIntelligenceSDK()
    >>> corpus = engine.load_corpus("/path/to/spectral/library")
    >>> result = engine.match(reference, candidate)
    >>> # V0.3: spawn a ghost agent to do work
    >>> ghost_result = engine.spawn_task("analyze spectrum", [
    ...     {"engine": "validation", "action": "validate"},
    ...     {"engine": "intelligence", "action": "reason"},
    ... ], record=my_record)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from mesie.agentic.ghost import GhostAgent, GhostConfig, GhostResult, TaskSpec
from mesie.agentic.network import AgentNetwork, NetworkResult, NetworkTopology
from mesie.agentic.spawner import AgentSpawner
from mesie.core.config import GenerationConfig
from mesie.core.records import MultiElementRecord
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.generation.fas import generate_fas
from mesie.generation.psd import generate_psd
from mesie.generation.rotdnn import generate_rotdnn
from mesie.io.corpus import SpectralCorpus
from mesie.io.loaders import RecordInput, load_record
from mesie.matching.matcher import MatchResult, SpectralMatcher, match_records
from mesie.processing.normalize import normalize_record
from mesie.validation.validators import ValidationReport, validate_record


class SpectralIntelligenceSDK:
    """Unified entrypoint for MESIE — the Multi-Element Spectral Intelligence Engine.

    V0.3: Now with agentic ghost computation, multi-network speed,
    workflow embedding, and real physics/math training datasets.

    Wraps all engine capabilities behind a single, discoverable interface.
    Designed for interactive use, scripting, and integration into larger
    systems.

    Args:
        phase_aware: Enable phase-aware matching by default.
        n_bands: Number of frequency bands for embedding vectorization.
        enable_core: Activate the inner core engine (agents, networks).
        max_agents: Maximum ghost agents in the pool.
        network_width: Default number of agents in network tasks.

    Example:
        >>> engine = SpectralIntelligenceSDK()
        >>> corpus = engine.load_corpus("./spectral_data")
        >>> record = engine.load("signal.json")
        >>> results = engine.rank(record, top_k=5)
        >>> # Spawn a ghost to do work
        >>> result = engine.spawn_task("find anomalies", [...])
    """

    def __init__(
        self,
        *,
        phase_aware: bool = False,
        n_bands: int = 8,
        enable_core: bool = True,
        max_agents: int = 128,
        network_width: int = 4,
    ) -> None:
        self._matcher = SpectralMatcher(phase_aware=phase_aware)
        self._vectorizer = SpectralVectorizer(n_bands=n_bands)
        self._corpus: Optional[SpectralCorpus] = None

        # V0.3: Core engine with agentic capabilities (lazy import to avoid circular)
        self._core = None
        if enable_core:
            from mesie.engines.core_engine import CoreConfig, CoreEngine
            self._core = CoreEngine(
                config=CoreConfig(
                    max_agents=max_agents,
                    network_width=network_width,
                    embed_workflows=True,
                    enable_reasoning=True,
                )
            )

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def load_corpus(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = True,
        skip_errors: bool = False,
    ) -> SpectralCorpus:
        """Load a spectral library from a directory and fit the matcher.

        Args:
            path: Path to directory containing spectral files.
            recursive: Search subdirectories recursively.
            skip_errors: Skip unloadable files instead of raising.

        Returns:
            The loaded SpectralCorpus.
        """
        self._corpus = SpectralCorpus.from_directory(
            path, recursive=recursive, skip_errors=skip_errors
        )
        self._matcher.fit(list(self._corpus))
        return self._corpus

    @property
    def corpus(self) -> Optional[SpectralCorpus]:
        """The currently loaded corpus, if any."""
        return self._corpus

    # ------------------------------------------------------------------
    # Record loading
    # ------------------------------------------------------------------

    def load(self, source: RecordInput, record_id: Optional[str] = None) -> MultiElementRecord:
        """Load a single spectral record from any supported format.

        Args:
            source: File path, dict, array, or existing record.
            record_id: Optional record identifier override.

        Returns:
            A MultiElementRecord instance.
        """
        return load_record(source, record_id=record_id)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, reference: RecordInput, candidate: RecordInput) -> MatchResult:
        """Match two spectral records and return a composite score.

        Args:
            reference: Reference record.
            candidate: Candidate record.

        Returns:
            MatchResult with score and metric breakdown.
        """
        return self._matcher.score(reference, candidate)

    def rank(self, candidate: RecordInput, top_k: int = 10) -> List[MatchResult]:
        """Rank corpus records against a candidate.

        Requires a corpus to be loaded first via load_corpus().

        Args:
            candidate: The candidate record to rank against.
            top_k: Number of top results to return.

        Returns:
            Sorted list of MatchResults.

        Raises:
            RuntimeError: If no corpus has been loaded.
        """
        if not self._corpus or len(self._corpus) == 0:
            raise RuntimeError(
                "No corpus loaded. Call load_corpus() first."
            )
        return self._matcher.rank_matches(candidate, top_k=top_k)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_psd(self, config: Optional[GenerationConfig] = None, **kwargs) -> MultiElementRecord:
        """Generate a Power Spectral Density record.

        Args:
            config: Optional generation configuration (uses default if None).
            **kwargs: Additional parameters passed to generate_psd.

        Returns:
            Generated MultiElementRecord.
        """
        cfg = config or GenerationConfig()
        return generate_psd(config=cfg, **kwargs)

    def generate_fas(self, config: Optional[GenerationConfig] = None, **kwargs) -> MultiElementRecord:
        """Generate a Fourier Amplitude Spectrum record.

        Args:
            config: Optional generation configuration (uses default if None).
            **kwargs: Additional parameters passed to generate_fas.

        Returns:
            Generated MultiElementRecord.
        """
        cfg = config or GenerationConfig()
        return generate_fas(config=cfg, **kwargs)

    def generate_rotdnn(self, config: Optional[GenerationConfig] = None, **kwargs) -> MultiElementRecord:
        """Generate a RotDnn spectrum record.

        Args:
            config: Optional generation configuration (uses default if None).
            **kwargs: Additional parameters passed to generate_rotdnn.

        Returns:
            Generated MultiElementRecord.
        """
        cfg = config or GenerationConfig()
        return generate_rotdnn(config=cfg, **kwargs)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, records: Union[RecordInput, Sequence[RecordInput]]) -> np.ndarray:
        """Compute spectral embeddings for one or more records.

        Args:
            records: A single record or sequence of records.

        Returns:
            Embedding array of shape (n_records, embedding_dim).
        """
        if isinstance(records, (list, tuple)):
            loaded = [load_record(r) for r in records]
        else:
            loaded = [load_record(records)]
        vectors = [self._vectorizer.transform(r) for r in loaded]
        return np.vstack(vectors)

    # ------------------------------------------------------------------
    # Validation & Normalization
    # ------------------------------------------------------------------

    def validate(self, record: RecordInput) -> ValidationReport:
        """Validate a spectral record.

        Args:
            record: Record to validate.

        Returns:
            ValidationReport with any issues found.
        """
        return validate_record(load_record(record))

    def normalize(self, record: RecordInput) -> MultiElementRecord:
        """Normalize a spectral record.

        Args:
            record: Record to normalize.

        Returns:
            Normalized MultiElementRecord.
        """
        return normalize_record(load_record(record))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Return the MESIE version string."""
        from mesie import __version__
        return __version__

    @property
    def core(self) -> Optional["CoreEngine"]:
        """Access the inner core engine directly."""
        return self._core

    def __repr__(self) -> str:
        corpus_info = f", corpus={len(self._corpus)} records" if self._corpus else ""
        core_info = ", core=active" if self._core else ""
        return f"SpectralIntelligenceSDK(v{self.version}{corpus_info}{core_info})"

    # ==================================================================
    # V0.3: AGENTIC GHOST COMPUTATION
    # ==================================================================

    def spawn_task(
        self,
        intent: str,
        actions: List[Dict[str, Any]],
        *,
        record: Optional[RecordInput] = None,
        chain: bool = True,
        timeout_s: float = 30.0,
    ) -> GhostResult:
        """Spawn a ghost agent to perform a computational task.

        This is the primary agentic interface — it activates a ghost
        in the machine to do actual work across multiple engines.

        Args:
            intent: Human-readable description of what the ghost should do.
            actions: List of engine/action/payload dicts defining the work.
            record: Optional record to inject into action payloads.
            chain: Chain results between steps.
            timeout_s: Maximum execution time.

        Returns:
            GhostResult with step-by-step outcomes and optional embedding.

        Raises:
            RuntimeError: If core engine not enabled.

        Example:
            >>> result = engine.spawn_task("full analysis", [
            ...     {"engine": "validation", "action": "validate"},
            ...     {"engine": "embedding", "action": "transform"},
            ...     {"engine": "intelligence", "action": "reason"},
            ... ], record=my_record)
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled. Pass enable_core=True.")

        # Inject record into payloads if provided
        if record is not None:
            rec = load_record(record)
            for act in actions:
                act.setdefault("payload", {})["record"] = rec

        task = TaskSpec(
            intent=intent,
            actions=actions,
            chain=chain,
            timeout_s=timeout_s,
        )
        return self._core.spawn_ghost(task)

    def spawn_many(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[GhostResult]:
        """Spawn multiple ghost agents in parallel.

        Args:
            tasks: List of dicts with 'intent' and 'actions' keys.

        Returns:
            List of GhostResults.
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")

        specs = [
            TaskSpec(intent=t.get("intent", "task"), actions=t.get("actions", []))
            for t in tasks
        ]
        return self._core.spawn_many(specs)

    # ==================================================================
    # V0.3: MULTI-NETWORK SPEED
    # ==================================================================

    def run_network(
        self,
        tasks: List[Dict[str, Any]],
        *,
        topology: str = "star",
        n_agents: Optional[int] = None,
    ) -> NetworkResult:
        """Execute tasks across a multi-agent network.

        Creates a network of ghost agents and distributes work across them
        for multi-network speed processing.

        Args:
            tasks: List of task dicts with 'intent' and 'actions'.
            topology: Network topology: "star", "mesh", "pipeline", "swarm".
            n_agents: Number of agents in the network.

        Returns:
            Aggregated NetworkResult.

        Example:
            >>> result = engine.run_network([
            ...     {"intent": "analyze A", "actions": [...]},
            ...     {"intent": "analyze B", "actions": [...]},
            ... ], topology="star", n_agents=4)
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")

        specs = [
            TaskSpec(intent=t.get("intent", "net_task"), actions=t.get("actions", []))
            for t in tasks
        ]
        topo = NetworkTopology(topology)
        return self._core.run_network(specs, topology=topo, n_agents=n_agents)

    # ==================================================================
    # V0.3: WORKFLOW & DATASET EMBEDDING
    # ==================================================================

    def embed_workflow(self, steps: List[Dict[str, Any]]) -> np.ndarray:
        """Embed a workflow definition into vector space.

        Converts a sequence of engine/action steps into a fixed-dimension
        vector for similarity search, recommendation, and clustering.

        Args:
            steps: Workflow step dicts with 'engine' and 'action' keys.

        Returns:
            Normalized embedding vector.

        Example:
            >>> emb = engine.embed_workflow([
            ...     {"engine": "validation", "action": "validate"},
            ...     {"engine": "matching", "action": "match"},
            ... ])
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        return self._core.embed_workflow(steps)

    def embed_dataset(
        self,
        dataset_id: str,
        records: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        """Embed a dataset into a single vector representation.

        Creates a centroid embedding from dataset records for
        dataset-level similarity search and retrieval.

        Args:
            dataset_id: Identifier for the dataset.
            records: Sequence of record dicts with 'amplitude'/'frequency'.

        Returns:
            Dataset centroid embedding vector.
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        return self._core.embed_dataset(dataset_id, records)

    # ==================================================================
    # V0.3: REASONING WITH REAL DATA
    # ==================================================================

    def reason(
        self,
        query: str,
        *,
        record: Optional[RecordInput] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a reasoning chain through the intelligence engine.

        Uses the core engine to reason about spectral data with
        physics-grounded inference.

        Args:
            query: What to reason about.
            record: Optional spectral record for context.
            context: Additional context dict.

        Returns:
            Dict with conclusion, confidence, and evidence.

        Example:
            >>> result = engine.reason("What is the dominant frequency?",
            ...                        record=my_record)
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        ctx = dict(context or {})
        if record is not None:
            ctx["record"] = load_record(record)
        return self._core.reason_chain(query, ctx)

    def load_training_datasets(self) -> Dict[str, Any]:
        """Load all built-in reasoning training datasets.

        Returns manifest with real physics, math, and geometry
        training examples for embedding calibration and reasoning.

        Returns:
            Dataset manifest summary dict.
        """
        from mesie.pretraining.reasoning_datasets import build_reasoning_datasets
        manifest = build_reasoning_datasets()
        return manifest.summary()

    # ==================================================================
    # V0.3: ENGINE DISPATCH (direct bus access)
    # ==================================================================

    def dispatch(
        self,
        engine: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch an action to any engine on the internal bus.

        Gives direct access to the full engine registry for advanced use.

        Args:
            engine: Target engine name.
            action: Action to perform.
            payload: Action payload.

        Returns:
            Response dict with 'ok', 'data', and optional 'error'.
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        resp = self._core.dispatch(engine, action, payload or {})
        return {"ok": resp.ok, "data": resp.data, "error": resp.error}

    def list_engines(self) -> List[str]:
        """List all available engines in the registry."""
        if self._core is None:
            return []
        return self._core.registry.names()

    def status(self) -> Dict[str, Any]:
        """Get full system status including agents and embeddings."""
        if self._core is None:
            return {"core": "disabled", "version": self.version}
        return {
            "version": self.version,
            "core": "active",
            "engines": self._core.registry.names(),
            "active_agents": self._core._spawner.active_count,
            "total_tasks": len(self._core.task_log),
            "workflow_embeddings": self._core.workflow_embedding_count,
            "dataset_embeddings": self._core.dataset_embedding_count,
        }

    # ==================================================================
    # V0.3.1: MULTI-MODEL PYTHON↔JULIA BRIDGE
    # ==================================================================

    @property
    def julia(self) -> "JuliaBridge":
        """Access the Julia bridge for direct Python↔Julia interop.

        Returns:
            JuliaBridge instance (lazy-initialized).

        Example:
            >>> result = engine.julia.validate(record_dict)
            >>> embedding = engine.julia.embed(record_dict)
        """
        if not hasattr(self, "_julia_bridge") or self._julia_bridge is None:
            from mesie.polyglot.julia_bridge import JuliaBridge
            self._julia_bridge = JuliaBridge(backend="auto")
        return self._julia_bridge

    def multimodel_validate(self, record: RecordInput) -> Dict[str, Any]:
        """Validate a record using both Python and Julia (multi-model consensus).

        Runs validation in both runtimes and returns a consensus result
        with cross-runtime confidence scoring.

        Args:
            record: Record to validate.

        Returns:
            Dict with consensus validation result, per-runtime details.

        Example:
            >>> result = engine.multimodel_validate(my_record)
            >>> print(result["confidence"])  # 1.0 if both agree
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        resp = self._core.dispatch(
            "multimodel_validation", "validate_consensus",
            {"record": record},
        )
        return {"ok": resp.ok, "data": resp.data, "error": resp.error}

    def multimodel_match(
        self,
        reference: RecordInput,
        candidate: RecordInput,
    ) -> Dict[str, Any]:
        """Match two records using fused Python+Julia scoring (multi-model).

        Combines Python's flexible metrics with Julia's numerical precision
        for a weighted fused score.

        Args:
            reference: Reference record.
            candidate: Candidate record.

        Returns:
            Dict with fused score and per-runtime breakdowns.

        Example:
            >>> result = engine.multimodel_match(ref, cand)
            >>> print(result["data"]["fused_score"])
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        resp = self._core.dispatch(
            "multimodel_matching", "match_fused",
            {"record_a": reference, "record_b": candidate},
        )
        return {"ok": resp.ok, "data": resp.data, "error": resp.error}

    def multimodel_embed(
        self,
        record: RecordInput,
        *,
        fusion_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Embed a record using fused Python+Julia embeddings (multi-model).

        Computes embeddings from both runtimes and fuses them via
        concatenation (richer) or averaging (compact).

        Args:
            record: Record to embed.
            fusion_mode: "concatenate" or "average" (default: engine setting).

        Returns:
            Fused embedding vector as numpy array.

        Example:
            >>> emb = engine.multimodel_embed(my_record)
            >>> print(emb.shape)  # larger dim from dual runtimes
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        payload: Dict[str, Any] = {"record": record}
        if fusion_mode:
            payload["fusion_mode"] = fusion_mode
        resp = self._core.dispatch("multimodel_embedding", "embed_fused", payload)
        if resp.ok and resp.data:
            return np.array(resp.data.get("embedding", []), dtype=np.float64)
        return np.array([], dtype=np.float64)

    def multimodel_fingerprint(self, record: RecordInput, resolution: int = 16) -> np.ndarray:
        """Compute a binary spectral fingerprint using Julia (multi-model).

        Uses Julia's numerical backend for fast binary fingerprint
        computation suitable for Hamming-distance ANN search.

        Args:
            record: Record to fingerprint.
            resolution: Fingerprint bit resolution (default 16).

        Returns:
            Binary fingerprint as uint8 numpy array.

        Example:
            >>> fp = engine.multimodel_fingerprint(my_record)
            >>> print(fp)  # array([1, 0, 1, 1, ...])
        """
        if self._core is None:
            raise RuntimeError("Core engine not enabled.")
        resp = self._core.dispatch(
            "multimodel_fingerprint", "fingerprint",
            {"record": record, "resolution": resolution},
        )
        if resp.ok and resp.data:
            return np.array(resp.data.get("fingerprint", []), dtype=np.uint8)
        return np.array([], dtype=np.uint8)

