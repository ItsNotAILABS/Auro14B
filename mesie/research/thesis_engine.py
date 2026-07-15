"""Thesis Engine — structures research into hypothesis → experiment → analysis → conclusion.

Provides a pipeline-based approach to scientific inquiry, where each stage
feeds into the next with validation gates between them.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ThesisStage(str, Enum):
    """Stages of the thesis research pipeline."""

    HYPOTHESIS = "hypothesis"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    DATA_COLLECTION = "data_collection"
    EXPERIMENT = "experiment"
    ANALYSIS = "analysis"
    INTERPRETATION = "interpretation"
    CONCLUSION = "conclusion"
    PEER_REVIEW = "peer_review"


@dataclass
class StageResult:
    """Output from a single thesis stage."""

    stage: ThesisStage
    status: str  # "passed", "failed", "needs_revision"
    data: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0


@dataclass
class ThesisResult:
    """Final output from a complete thesis pipeline run."""

    thesis_id: str
    title: str
    hypothesis: str
    conclusion: str
    stages: List[StageResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thesis_id": self.thesis_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "conclusion": self.conclusion,
            "stages": [
                {"stage": s.stage.value, "status": s.status, "notes": s.notes}
                for s in self.stages
            ],
            "passed": self.passed,
            "metadata": self.metadata,
        }


# Type alias for stage handlers
StageHandler = Callable[[ThesisStage, Dict[str, Any]], StageResult]


@dataclass
class ThesisPipeline:
    """Configurable pipeline of thesis stages.

    Allows customization of which stages to include and what handlers
    to use for each stage.

    Attributes:
        stages: Ordered list of stages to execute.
        handlers: Mapping of stage to handler function.
        gate_threshold: Minimum pass rate to proceed between stages.
    """

    stages: List[ThesisStage] = field(default_factory=lambda: [
        ThesisStage.HYPOTHESIS,
        ThesisStage.LITERATURE_REVIEW,
        ThesisStage.METHODOLOGY,
        ThesisStage.DATA_COLLECTION,
        ThesisStage.EXPERIMENT,
        ThesisStage.ANALYSIS,
        ThesisStage.INTERPRETATION,
        ThesisStage.CONCLUSION,
    ])
    handlers: Dict[ThesisStage, StageHandler] = field(default_factory=dict)
    gate_threshold: float = 0.7

    def set_handler(self, stage: ThesisStage, handler: StageHandler) -> None:
        """Register a custom handler for a stage."""
        self.handlers[stage] = handler


class ThesisEngine:
    """Orchestrates thesis-driven research workflows.

    Executes a ThesisPipeline, moving through stages sequentially with
    validation gates. Each stage can leverage tools from the research
    platform (data sources, labs, analysis engines).

    Args:
        pipeline: The thesis pipeline configuration. Defaults to standard stages.
        max_retries: Maximum retries per stage on failure.
        strict_gates: If True, fail the entire thesis on any stage failure.

    Example:
        >>> engine = ThesisEngine()
        >>> result = engine.run(
        ...     title="Thermal Conductivity of Novel Alloys",
        ...     hypothesis="Alloy X has higher conductivity than pure copper",
        ... )
    """

    def __init__(
        self,
        pipeline: Optional[ThesisPipeline] = None,
        max_retries: int = 2,
        strict_gates: bool = False,
    ) -> None:
        self._pipeline = pipeline or ThesisPipeline()
        self._max_retries = max_retries
        self._strict_gates = strict_gates

    @property
    def pipeline(self) -> ThesisPipeline:
        return self._pipeline

    def run(
        self,
        title: str,
        hypothesis: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> ThesisResult:
        """Execute the full thesis pipeline.

        Args:
            title: Title of the research thesis.
            hypothesis: The hypothesis to test.
            context: Additional context passed to all stages.
            data: Initial data payload for the pipeline.

        Returns:
            A ThesisResult summarizing all stage outcomes.
        """
        thesis_id = str(uuid.uuid4())[:12]
        stage_results: List[StageResult] = []
        current_data = {
            "title": title,
            "hypothesis": hypothesis,
            **(context or {}),
            **(data or {}),
        }
        all_passed = True

        for stage in self._pipeline.stages:
            start = time.time()
            result = self._execute_stage(stage, current_data)
            result.duration_seconds = time.time() - start
            stage_results.append(result)

            if result.status == "passed":
                current_data.update(result.data)
            elif result.status == "failed":
                all_passed = False
                if self._strict_gates:
                    break
            # "needs_revision" continues but flags

        conclusion = self._derive_conclusion(hypothesis, stage_results)

        return ThesisResult(
            thesis_id=thesis_id,
            title=title,
            hypothesis=hypothesis,
            conclusion=conclusion,
            stages=stage_results,
            passed=all_passed,
            metadata={
                "total_stages": len(self._pipeline.stages),
                "completed_stages": len(stage_results),
                "strict_gates": self._strict_gates,
            },
        )

    def _execute_stage(
        self, stage: ThesisStage, data: Dict[str, Any]
    ) -> StageResult:
        """Execute a single pipeline stage with retry logic."""
        handler = self._pipeline.handlers.get(stage, self._default_handler)

        for attempt in range(self._max_retries + 1):
            try:
                result = handler(stage, data)
                return result
            except Exception as exc:
                if attempt == self._max_retries:
                    return StageResult(
                        stage=stage,
                        status="failed",
                        notes=f"Failed after {self._max_retries + 1} attempts: {exc}",
                    )
        # Unreachable, but satisfies type checker
        return StageResult(stage=stage, status="failed", notes="Exhausted retries")

    def _default_handler(
        self, stage: ThesisStage, data: Dict[str, Any]
    ) -> StageResult:
        """Default no-op handler that passes all stages."""
        return StageResult(
            stage=stage,
            status="passed",
            data={"stage_executed": stage.value, "input_keys": list(data.keys())},
            notes=f"Default handler executed for {stage.value}",
        )

    def _derive_conclusion(
        self, hypothesis: str, results: List[StageResult]
    ) -> str:
        """Derive conclusion from stage results."""
        passed_count = sum(1 for r in results if r.status == "passed")
        total = len(results)
        ratio = passed_count / total if total > 0 else 0.0

        if ratio >= 0.8:
            return (
                f"Evidence supports the hypothesis: '{hypothesis}'. "
                f"{passed_count}/{total} stages passed successfully."
            )
        elif ratio >= 0.5:
            return (
                f"Partial support for hypothesis: '{hypothesis}'. "
                f"{passed_count}/{total} stages passed. Further research needed."
            )
        else:
            return (
                f"Insufficient evidence for hypothesis: '{hypothesis}'. "
                f"Only {passed_count}/{total} stages passed."
            )
