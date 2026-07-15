"""Pipeline orchestrator — coordinates multi-stage AI processing pipelines.

Manages the execution flow of spectral data through connected AI subsystems,
providing stage-by-stage processing with intermediate state management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.features.electro_spectral import ElectroSpectralLayer
from mesie.ai.intelligence_protocols import IntelligenceProtocol, IntelligenceConfig, ReasoningResult


@dataclass
class OrchestratorConfig:
    """Configuration for the pipeline orchestrator.

    Args:
        stages: Ordered list of processing stage names.
        intelligence_config: Config for intelligence protocol integration.
        enable_caching: Whether to cache intermediate results.
        max_pipeline_depth: Maximum number of stages per pipeline run.
    """

    stages: List[str] = field(default_factory=lambda: [
        "embed", "extract_features", "reason", "route"
    ])
    intelligence_config: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    enable_caching: bool = True
    max_pipeline_depth: int = 10


class PipelineOrchestrator:
    """Orchestrates multi-stage AI processing pipelines.

    Coordinates data flow through embedding, feature extraction,
    reasoning, and routing stages with full state tracking.

    Args:
        config: Orchestrator configuration.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self._vectorizer = SpectralVectorizer()
        self._electro = ElectroSpectralLayer()
        self._protocol = IntelligenceProtocol(config=self.config.intelligence_config)
        self._stage_handlers: Dict[str, Callable[..., Any]] = {
            "embed": self._stage_embed,
            "extract_features": self._stage_extract_features,
            "reason": self._stage_reason,
            "route": self._stage_route,
        }
        self._pipeline_results: List[Dict[str, Any]] = []
        self._cache: Dict[str, Dict[str, Any]] = {}

    def register_stage(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a custom pipeline stage handler.

        Args:
            name: Stage name.
            handler: Function that accepts and returns a pipeline context dict.
        """
        self._stage_handlers[name] = handler

    def run(self, record: RecordInput) -> Dict[str, Any]:
        """Run the full pipeline on a record.

        Args:
            record: Input spectral record.

        Returns:
            Final pipeline context with all stage results.
        """
        rec = load_record(record)
        context: Dict[str, Any] = {
            "record_id": rec.record_id,
            "record": rec,
            "stages_completed": [],
            "stage_results": {},
        }

        # Check cache
        if self.config.enable_caching and rec.record_id in self._cache:
            return self._cache[rec.record_id]

        for stage_name in self.config.stages:
            if stage_name not in self._stage_handlers:
                continue
            if len(context["stages_completed"]) >= self.config.max_pipeline_depth:
                break

            handler = self._stage_handlers[stage_name]
            stage_result = handler(context)
            context["stage_results"][stage_name] = stage_result
            context["stages_completed"].append(stage_name)

        # Cache result
        if self.config.enable_caching:
            self._cache[rec.record_id] = context

        self._pipeline_results.append(context)
        return context

    def run_batch(self, records: Sequence[RecordInput]) -> List[Dict[str, Any]]:
        """Run the pipeline on multiple records.

        Args:
            records: Input records.

        Returns:
            List of pipeline context dictionaries.
        """
        return [self.run(r) for r in records]

    def _stage_embed(self, context: Dict[str, Any]) -> np.ndarray:
        """Embedding stage."""
        rec = context["record"]
        embedding = self._vectorizer.transform(rec)
        context["embedding"] = embedding
        return embedding

    def _stage_extract_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Feature extraction stage."""
        rec = context["record"]
        sig = self._electro.compute_signature(rec)
        features = {
            "centroid": sig.spectral_centroid,
            "spread": sig.spectral_spread,
            "resonance": sig.frequency_resonance,
            "coherence": sig.coherence_signature,
            "harmonic_alignment": sig.harmonic_alignment,
        }
        context["features"] = features
        return features

    def _stage_reason(self, context: Dict[str, Any]) -> ReasoningResult:
        """Reasoning stage."""
        embedding = context.get("embedding")
        if embedding is None:
            rec = context["record"]
            embedding = self._vectorizer.transform(rec)
        result = self._protocol.reason(embedding)
        context["reasoning"] = result
        return result

    def _stage_route(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Routing stage — determines downstream processing path."""
        reasoning = context.get("reasoning")
        features = context.get("features", {})

        route = "standard"
        priority = 0.5

        if reasoning and reasoning.conclusion == "anomaly_detected":
            route = "anomaly_investigation"
            priority = 1.0
        elif reasoning and reasoning.conclusion == "low_signal":
            route = "signal_enhancement"
            priority = 0.3
        elif features.get("coherence", 0) > 0.8:
            route = "high_coherence_analysis"
            priority = 0.7

        routing = {"route": route, "priority": priority}
        context["routing"] = routing
        return routing

    @property
    def pipeline_history(self) -> List[Dict[str, Any]]:
        """History of all pipeline runs."""
        return self._pipeline_results

    def clear_cache(self) -> None:
        """Clear the pipeline result cache."""
        self._cache.clear()
