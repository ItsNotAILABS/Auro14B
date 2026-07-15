"""Task decomposition and routing for research workflows.

Breaks a research question into subtasks and routes them to appropriate
tools/engines on the internal bus.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class TaskStatus(str, Enum):
    """Lifecycle state of a research task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ResearchTask:
    """Single unit of work within a research plan.

    Attributes:
        task_id: Unique identifier.
        name: Human-readable task name.
        description: What this task accomplishes.
        tool: Target tool or engine name to execute the task.
        action: The action/command within that tool.
        params: Parameters to pass to the tool.
        depends_on: List of task_ids that must complete first.
        status: Current lifecycle state.
        result: Output data once completed.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    tool: str = ""
    action: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING

    def mark_completed(self, result: Dict[str, Any]) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error


@dataclass
class TaskGraph:
    """Directed acyclic graph of research tasks.

    Manages dependencies and execution ordering.
    """

    tasks: List[ResearchTask] = field(default_factory=list)

    def add_task(self, task: ResearchTask) -> None:
        self.tasks.append(task)

    def get_ready_tasks(self) -> List[ResearchTask]:
        """Return tasks whose dependencies are all completed."""
        completed_ids = {
            t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED)
            for t in self.tasks
        )

    @property
    def completed_tasks(self) -> List[ResearchTask]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    @property
    def failed_tasks(self) -> List[ResearchTask]:
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "pending": len(
                [t for t in self.tasks if t.status == TaskStatus.PENDING]
            ),
        }


class ResearchPlanner:
    """Decomposes a research question into a TaskGraph.

    The planner analyzes the research question, identifies required tools
    and data sources, and constructs an execution plan as a DAG of tasks.

    Args:
        available_tools: List of tool/engine names available for routing.
        max_depth: Maximum decomposition depth for nested sub-questions.
    """

    def __init__(
        self,
        available_tools: Optional[Sequence[str]] = None,
        max_depth: int = 3,
    ) -> None:
        self._available_tools = list(available_tools or [])
        self._max_depth = max_depth
        self._templates: Dict[str, List[Dict[str, Any]]] = {
            "literature_review": [
                {"name": "Search Literature", "tool": "data_sources", "action": "search_papers"},
                {"name": "Extract Key Findings", "tool": "research_agent", "action": "extract_findings"},
                {"name": "Synthesize Review", "tool": "research_agent", "action": "synthesize"},
            ],
            "hypothesis_test": [
                {"name": "Formulate Hypothesis", "tool": "research_agent", "action": "formulate"},
                {"name": "Acquire Data", "tool": "data_sources", "action": "fetch"},
                {"name": "Run Analysis", "tool": "statistics", "action": "test_hypothesis"},
                {"name": "Interpret Results", "tool": "research_agent", "action": "interpret"},
            ],
            "data_analysis": [
                {"name": "Acquire Data", "tool": "data_sources", "action": "fetch"},
                {"name": "Preprocess", "tool": "signal_processing", "action": "preprocess"},
                {"name": "Analyze", "tool": "statistics", "action": "analyze"},
                {"name": "Visualize", "tool": "research_agent", "action": "visualize"},
            ],
            "spectral_research": [
                {"name": "Load Spectral Data", "tool": "mesie-data", "action": "load"},
                {"name": "Fingerprint", "tool": "fingerprint", "action": "compute"},
                {"name": "Match & Rank", "tool": "matching", "action": "rank"},
                {"name": "Report", "tool": "research_agent", "action": "report"},
            ],
        }

    @property
    def templates(self) -> List[str]:
        """Available research plan templates."""
        return list(self._templates.keys())

    def plan(
        self,
        question: str,
        *,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskGraph:
        """Create a task graph from a research question.

        Args:
            question: The research question to decompose.
            template: Optional named template to apply.
            context: Additional context for planning.

        Returns:
            A TaskGraph with ordered, dependency-linked tasks.
        """
        graph = TaskGraph()
        ctx = context or {}

        if template and template in self._templates:
            steps = self._templates[template]
        else:
            steps = self._infer_steps(question, ctx)

        prev_id: Optional[str] = None
        for step_def in steps:
            task = ResearchTask(
                name=step_def.get("name", "Unnamed Step"),
                description=step_def.get("description", ""),
                tool=step_def.get("tool", ""),
                action=step_def.get("action", ""),
                params={**step_def.get("params", {}), "question": question, **ctx},
                depends_on=[prev_id] if prev_id else [],
            )
            graph.add_task(task)
            prev_id = task.task_id

        return graph

    def _infer_steps(
        self, question: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Heuristic step inference based on question keywords."""
        q_lower = question.lower()

        if any(kw in q_lower for kw in ("review", "survey", "literature")):
            return self._templates["literature_review"]
        elif any(kw in q_lower for kw in ("hypothesis", "test", "significant")):
            return self._templates["hypothesis_test"]
        elif any(kw in q_lower for kw in ("spectral", "spectrum", "frequency")):
            return self._templates["spectral_research"]
        else:
            return self._templates["data_analysis"]
