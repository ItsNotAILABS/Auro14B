"""Spectral Distributed Processing and Pipeline Orchestration.

Provides distributed computing patterns, pipeline orchestration,
and workflow management for large-scale spectral processing.

Key Components:
    - SpectralPipelineStage: Individual processing stage
    - SpectralProcessingPipeline: Multi-stage pipeline
    - BatchProcessor: Efficient batch spectral processing
    - StreamProcessor: Real-time streaming pipeline
    - WorkflowOrchestrator: Complex workflow management
    - ResultAggregator: Combine results from distributed processing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class StageStatus(Enum):
    """Pipeline stage status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineMode(Enum):
    """Pipeline execution mode."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    STREAMING = "streaming"


class AggregationMethod(Enum):
    """Result aggregation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED = "weighted"
    VOTING = "voting"
    STACKING = "stacking"
    CONCATENATION = "concatenation"


class WorkflowState(Enum):
    """Workflow execution state."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class StageResult:
    """Result from a pipeline stage.

    Args:
        stage_name: Name of the stage.
        output: Stage output data.
        duration: Execution time in seconds.
        status: Stage status.
        metrics: Performance metrics.
        error: Error message if failed.
    """
    stage_name: str
    output: Any = None
    duration: float = 0.0
    status: StageStatus = StageStatus.COMPLETED
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BatchItem:
    """An item in a batch processing queue.

    Args:
        item_id: Unique identifier.
        data: Input data.
        priority: Processing priority (higher = first).
        metadata: Additional metadata.
    """
    item_id: str
    data: np.ndarray
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamFrame:
    """A frame in a streaming pipeline.

    Args:
        frame_id: Sequential frame ID.
        data: Frame data.
        timestamp: Frame timestamp.
        is_keyframe: Whether this is a keyframe.
    """
    frame_id: int
    data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    is_keyframe: bool = False


@dataclass
class WorkflowStep:
    """A step in a workflow.

    Args:
        step_id: Step identifier.
        name: Step name.
        function: Processing function.
        dependencies: Steps that must complete first.
        condition: Optional condition for execution.
    """
    step_id: str
    name: str
    function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None


# =============================================================================
# Pipeline Stage
# =============================================================================


class SpectralPipelineStage:
    """A single stage in a spectral processing pipeline.

    Wraps a processing function with input/output validation,
    timing, and error handling.

    Args:
        name: Stage name.
        process_fn: Processing function (takes array, returns array).
        validate_input: Optional input validation function.
        validate_output: Optional output validation function.
    """

    def __init__(
        self,
        name: str,
        process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        validate_input: Optional[Callable[[np.ndarray], bool]] = None,
        validate_output: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> None:
        self.name = name
        self._process_fn = process_fn or (lambda x: x)
        self._validate_input = validate_input
        self._validate_output = validate_output
        self._execution_count: int = 0
        self._total_time: float = 0.0
        self._status = StageStatus.IDLE

    def execute(self, data: np.ndarray) -> StageResult:
        """Execute this pipeline stage.

        Args:
            data: Input data.

        Returns:
            StageResult with output and metrics.
        """
        self._status = StageStatus.RUNNING
        start_time = time.time()

        try:
            # Validate input
            if self._validate_input and not self._validate_input(data):
                self._status = StageStatus.FAILED
                return StageResult(
                    stage_name=self.name,
                    status=StageStatus.FAILED,
                    error="Input validation failed",
                )

            # Process
            output = self._process_fn(data)

            # Validate output
            if self._validate_output and not self._validate_output(output):
                self._status = StageStatus.FAILED
                return StageResult(
                    stage_name=self.name,
                    status=StageStatus.FAILED,
                    error="Output validation failed",
                )

            duration = time.time() - start_time
            self._execution_count += 1
            self._total_time += duration
            self._status = StageStatus.COMPLETED

            return StageResult(
                stage_name=self.name,
                output=output,
                duration=duration,
                status=StageStatus.COMPLETED,
                metrics={
                    "execution_count": self._execution_count,
                    "avg_time": self._total_time / self._execution_count,
                },
            )

        except Exception as e:
            self._status = StageStatus.FAILED
            return StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )

    @property
    def status(self) -> StageStatus:
        """Current stage status."""
        return self._status

    @property
    def avg_execution_time(self) -> float:
        """Average execution time."""
        if self._execution_count == 0:
            return 0.0
        return self._total_time / self._execution_count


# =============================================================================
# Spectral Processing Pipeline
# =============================================================================


class SpectralProcessingPipeline:
    """Multi-stage spectral processing pipeline.

    Chains multiple processing stages together with
    data flow management and error handling.

    Args:
        name: Pipeline name.
        mode: Execution mode.
    """

    def __init__(
        self,
        name: str = "spectral_pipeline",
        mode: PipelineMode = PipelineMode.SEQUENTIAL,
    ) -> None:
        self.name = name
        self.mode = mode
        self._stages: List[SpectralPipelineStage] = []
        self._results: List[StageResult] = []
        self._execution_count: int = 0

    def add_stage(self, stage: SpectralPipelineStage) -> "SpectralProcessingPipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Processing stage.

        Returns:
            Self for chaining.
        """
        self._stages.append(stage)
        return self

    def add_function(self, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> "SpectralProcessingPipeline":
        """Add a function as a pipeline stage.

        Args:
            name: Stage name.
            fn: Processing function.

        Returns:
            Self for chaining.
        """
        stage = SpectralPipelineStage(name=name, process_fn=fn)
        self._stages.append(stage)
        return self

    def execute(self, data: np.ndarray) -> Dict[str, Any]:
        """Execute the full pipeline.

        Args:
            data: Input data.

        Returns:
            Dictionary with final output and all stage results.
        """
        self._results = []
        current_data = np.atleast_1d(data).copy()
        start_time = time.time()

        for stage in self._stages:
            result = stage.execute(current_data)
            self._results.append(result)

            if result.status == StageStatus.FAILED:
                return {
                    "output": None,
                    "status": "failed",
                    "failed_stage": stage.name,
                    "error": result.error,
                    "results": self._results,
                    "duration": time.time() - start_time,
                }

            if result.output is not None:
                current_data = result.output

        self._execution_count += 1
        total_duration = time.time() - start_time

        return {
            "output": current_data,
            "status": "completed",
            "results": self._results,
            "duration": total_duration,
            "n_stages": len(self._stages),
        }

    @property
    def n_stages(self) -> int:
        """Number of stages."""
        return len(self._stages)

    @property
    def execution_count(self) -> int:
        """Total pipeline executions."""
        return self._execution_count


# =============================================================================
# Batch Processor
# =============================================================================


class BatchProcessor:
    """Efficient batch processing of spectral data.

    Processes multiple spectra in batches with priority
    scheduling, progress tracking, and result collection.

    Args:
        pipeline: Processing pipeline to use.
        batch_size: Maximum batch size.
        max_queue_size: Maximum queue depth.
    """

    def __init__(
        self,
        pipeline: Optional[SpectralProcessingPipeline] = None,
        batch_size: int = 32,
        max_queue_size: int = 1000,
    ) -> None:
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self._pipeline = pipeline or SpectralProcessingPipeline()
        self._queue: List[BatchItem] = []
        self._results: Dict[str, Any] = {}
        self._processed_count: int = 0

    def enqueue(self, item: BatchItem) -> bool:
        """Add item to processing queue.

        Args:
            item: Item to process.

        Returns:
            True if added, False if queue full.
        """
        if len(self._queue) >= self.max_queue_size:
            return False
        self._queue.append(item)
        # Sort by priority
        self._queue.sort(key=lambda x: x.priority, reverse=True)
        return True

    def enqueue_batch(self, items: List[BatchItem]) -> int:
        """Add multiple items to queue.

        Args:
            items: Items to add.

        Returns:
            Number of items successfully added.
        """
        added = 0
        for item in items:
            if self.enqueue(item):
                added += 1
        return added

    def process_batch(self) -> List[Dict[str, Any]]:
        """Process one batch from the queue.

        Returns:
            List of results for processed items.
        """
        # Take batch from queue
        batch = self._queue[:self.batch_size]
        self._queue = self._queue[self.batch_size:]

        results = []
        for item in batch:
            result = self._pipeline.execute(item.data)
            result["item_id"] = item.item_id
            result["metadata"] = item.metadata
            results.append(result)
            self._results[item.item_id] = result
            self._processed_count += 1

        return results

    def process_all(self) -> List[Dict[str, Any]]:
        """Process entire queue.

        Returns:
            All results.
        """
        all_results = []
        while self._queue:
            batch_results = self.process_batch()
            all_results.extend(batch_results)
        return all_results

    def get_result(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific item."""
        return self._results.get(item_id)

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def processed_count(self) -> int:
        """Total items processed."""
        return self._processed_count


# =============================================================================
# Stream Processor
# =============================================================================


class StreamProcessor:
    """Real-time streaming spectral processor.

    Processes spectral data as a continuous stream with
    windowing, overlap, and continuous output.

    Args:
        window_size: Processing window size.
        hop_size: Hop between windows.
        pipeline: Processing pipeline.
    """

    def __init__(
        self,
        window_size: int = 1024,
        hop_size: int = 512,
        pipeline: Optional[SpectralProcessingPipeline] = None,
    ) -> None:
        self.window_size = window_size
        self.hop_size = hop_size
        self._pipeline = pipeline or SpectralProcessingPipeline()

        self._buffer = np.zeros(0)
        self._frame_count: int = 0
        self._output_buffer: List[np.ndarray] = []
        self._is_running: bool = False

    def start(self) -> None:
        """Start the stream processor."""
        self._is_running = True
        self._buffer = np.zeros(0)
        self._frame_count = 0

    def stop(self) -> None:
        """Stop the stream processor."""
        self._is_running = False

    def feed(self, data: np.ndarray) -> List[np.ndarray]:
        """Feed new data to the stream.

        Args:
            data: New incoming data.

        Returns:
            List of processed frames (may be empty).
        """
        if not self._is_running:
            return []

        data = np.atleast_1d(data).flatten()
        self._buffer = np.concatenate([self._buffer, data])

        outputs = []
        while len(self._buffer) >= self.window_size:
            # Extract window
            window = self._buffer[:self.window_size]
            self._buffer = self._buffer[self.hop_size:]

            # Process through pipeline
            result = self._pipeline.execute(window)
            if result["status"] == "completed" and result["output"] is not None:
                outputs.append(result["output"])
                self._output_buffer.append(result["output"])
                self._frame_count += 1

        return outputs

    def get_output(self, n_frames: Optional[int] = None) -> np.ndarray:
        """Get accumulated output frames.

        Args:
            n_frames: Number of frames (None = all).

        Returns:
            Concatenated output.
        """
        if not self._output_buffer:
            return np.array([])

        frames = self._output_buffer[-n_frames:] if n_frames else self._output_buffer
        return np.concatenate(frames)

    @property
    def frame_count(self) -> int:
        """Total frames processed."""
        return self._frame_count

    @property
    def is_running(self) -> bool:
        """Whether stream is active."""
        return self._is_running

    @property
    def buffer_level(self) -> int:
        """Current buffer level."""
        return len(self._buffer)


# =============================================================================
# Workflow Orchestrator
# =============================================================================


class WorkflowOrchestrator:
    """Manage complex spectral processing workflows.

    Orchestrates multi-step workflows with dependencies,
    conditions, and parallel execution paths.

    Args:
        name: Workflow name.
        max_retries: Maximum retries per step.
    """

    def __init__(
        self,
        name: str = "spectral_workflow",
        max_retries: int = 3,
    ) -> None:
        self.name = name
        self.max_retries = max_retries
        self._steps: Dict[str, WorkflowStep] = {}
        self._step_results: Dict[str, Any] = {}
        self._state = WorkflowState.PENDING
        self._execution_log: List[Dict[str, Any]] = []

    def add_step(self, step: WorkflowStep) -> "WorkflowOrchestrator":
        """Add a step to the workflow.

        Args:
            step: Workflow step.

        Returns:
            Self for chaining.
        """
        self._steps[step.step_id] = step
        return self

    def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the workflow.

        Args:
            initial_data: Initial data to pass to steps.

        Returns:
            Workflow results.
        """
        self._state = WorkflowState.ACTIVE
        self._step_results = {}
        context = initial_data or {}
        start_time = time.time()

        # Topological sort
        execution_order = self._topological_sort()

        for step_id in execution_order:
            step = self._steps[step_id]

            # Check condition
            if step.condition and not step.condition(context):
                self._log_step(step_id, "skipped", "Condition not met")
                continue

            # Check dependencies
            deps_met = all(
                dep in self._step_results
                for dep in step.dependencies
            )
            if not deps_met:
                self._log_step(step_id, "skipped", "Dependencies not met")
                continue

            # Execute with retries
            success = False
            for attempt in range(self.max_retries):
                try:
                    if step.function:
                        result = step.function(context)
                        self._step_results[step_id] = result
                        context[step_id] = result
                    success = True
                    self._log_step(step_id, "completed", f"Attempt {attempt + 1}")
                    break
                except Exception as e:
                    self._log_step(step_id, "retry", str(e))

            if not success:
                self._state = WorkflowState.FAILED
                return {
                    "status": "failed",
                    "failed_step": step_id,
                    "results": self._step_results,
                    "duration": time.time() - start_time,
                }

        self._state = WorkflowState.COMPLETED
        return {
            "status": "completed",
            "results": self._step_results,
            "duration": time.time() - start_time,
            "steps_executed": len(self._step_results),
        }

    def _topological_sort(self) -> List[str]:
        """Sort steps by dependencies."""
        visited = set()
        result = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            step = self._steps.get(step_id)
            if step:
                for dep in step.dependencies:
                    visit(dep)
            result.append(step_id)

        for step_id in self._steps:
            visit(step_id)

        return result

    def _log_step(self, step_id: str, status: str, message: str) -> None:
        """Log step execution."""
        self._execution_log.append({
            "step_id": step_id,
            "status": status,
            "message": message,
            "timestamp": time.time(),
        })

    @property
    def state(self) -> WorkflowState:
        """Current workflow state."""
        return self._state

    @property
    def n_steps(self) -> int:
        """Number of workflow steps."""
        return len(self._steps)


# =============================================================================
# Result Aggregator
# =============================================================================


class ResultAggregator:
    """Aggregate results from distributed spectral processing.

    Combines outputs from parallel processing with various
    aggregation strategies.

    Args:
        method: Aggregation method.
        weights: Optional weights for weighted aggregation.
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.MEAN,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.method = method
        self.weights = weights
        self._results: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    def add_result(self, result: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a result for aggregation.

        Args:
            result: Processing result.
            metadata: Result metadata.
        """
        self._results.append(np.atleast_1d(result).flatten())
        self._metadata.append(metadata or {})

    def aggregate(self) -> np.ndarray:
        """Aggregate all collected results.

        Returns:
            Aggregated result.
        """
        if not self._results:
            return np.array([])

        # Align lengths
        max_len = max(len(r) for r in self._results)
        aligned = np.zeros((len(self._results), max_len))
        for i, r in enumerate(self._results):
            aligned[i, :len(r)] = r

        if self.method == AggregationMethod.MEAN:
            return np.mean(aligned, axis=0)
        elif self.method == AggregationMethod.MEDIAN:
            return np.median(aligned, axis=0)
        elif self.method == AggregationMethod.WEIGHTED:
            weights = self.weights
            if weights is None:
                weights = np.ones(len(self._results)) / len(self._results)
            weights = weights[:len(self._results)]
            weights = weights / (np.sum(weights) + 1e-12)
            return np.average(aligned, axis=0, weights=weights)
        elif self.method == AggregationMethod.CONCATENATION:
            return np.concatenate(self._results)
        elif self.method == AggregationMethod.VOTING:
            # Majority voting (for classification results)
            return np.array([
                float(np.bincount(aligned[:, i].astype(int)).argmax())
                for i in range(max_len)
            ])
        else:
            return np.mean(aligned, axis=0)

    def clear(self) -> None:
        """Clear all collected results."""
        self._results = []
        self._metadata = []

    @property
    def n_results(self) -> int:
        """Number of results collected."""
        return len(self._results)

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about collected results."""
        if not self._results:
            return {}

        max_len = max(len(r) for r in self._results)
        aligned = np.zeros((len(self._results), max_len))
        for i, r in enumerate(self._results):
            aligned[i, :len(r)] = r

        return {
            "n_results": len(self._results),
            "mean_length": float(np.mean([len(r) for r in self._results])),
            "inter_result_std": float(np.mean(np.std(aligned, axis=0))),
            "agreement": float(1.0 - np.mean(np.std(aligned, axis=0)) / (np.mean(np.abs(aligned)) + 1e-12)),
        }
