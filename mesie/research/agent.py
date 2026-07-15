"""Research Agent — orchestrates multi-step scientific workflows.

The ResearchAgent plans, executes, and reports on research tasks using
the MESIE tool ecosystem, data sources, and labs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mesie.research.planner import ResearchPlanner, ResearchTask, TaskGraph, TaskStatus
from mesie.research.reporter import ReportFormat, ResearchReport, ResearchReporter
from mesie.research.thesis_engine import ThesisEngine, ThesisResult


@dataclass
class ResearchConfig:
    """Configuration for the Research Agent.

    Attributes:
        max_concurrent_tasks: Maximum parallel tasks.
        auto_report: Automatically generate report on completion.
        report_format: Default format for auto-reports.
        strict_mode: Fail entire research if any task fails.
        timeout_seconds: Maximum time for entire research run.
    """

    max_concurrent_tasks: int = 4
    auto_report: bool = True
    report_format: ReportFormat = ReportFormat.MARKDOWN
    strict_mode: bool = False
    timeout_seconds: float = 3600.0


@dataclass
class ResearchResult:
    """Outcome of a research agent run.

    Attributes:
        research_id: Unique identifier for this research session.
        question: The original research question.
        status: Final status (completed/failed/timeout).
        task_graph: The execution plan with results.
        report: Generated research report.
        thesis_result: If thesis mode was used, the thesis outcome.
        duration_seconds: Total wall-clock time.
    """

    research_id: str
    question: str
    status: str  # "completed", "failed", "timeout"
    task_graph: TaskGraph
    report: Optional[ResearchReport] = None
    thesis_result: Optional[ThesisResult] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_id": self.research_id,
            "question": self.question,
            "status": self.status,
            "graph_summary": self.task_graph.summary(),
            "duration_seconds": self.duration_seconds,
            "has_report": self.report is not None,
            "has_thesis": self.thesis_result is not None,
            "metadata": self.metadata,
        }


class ResearchAgent:
    """Autonomous agent that plans and executes multi-step research.

    The ResearchAgent accepts a research question or hypothesis, decomposes
    it into tasks using the ResearchPlanner, executes those tasks by
    dispatching to tools/engines, and generates a structured report.

    It can operate in two modes:
    - **Question mode**: Open-ended research (literature review, analysis)
    - **Thesis mode**: Hypothesis-driven pipeline (hypothesis → conclusion)

    Args:
        config: Agent configuration.
        planner: Optional custom planner instance.
        reporter: Optional custom reporter instance.
        thesis_engine: Optional custom thesis engine.
        tool_dispatch: Optional callback to dispatch tasks to tools.
            Signature: (tool: str, action: str, params: dict) -> dict

    Example:
        >>> agent = ResearchAgent()
        >>> result = agent.research("What is the spectral signature of steel?")
        >>> print(result.report.title)
    """

    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        planner: Optional[ResearchPlanner] = None,
        reporter: Optional[ResearchReporter] = None,
        thesis_engine: Optional[ThesisEngine] = None,
        tool_dispatch: Optional[Any] = None,
    ) -> None:
        self._config = config or ResearchConfig()
        self._planner = planner or ResearchPlanner()
        self._reporter = reporter or ResearchReporter(
            default_format=self._config.report_format
        )
        self._thesis_engine = thesis_engine or ThesisEngine()
        self._tool_dispatch = tool_dispatch
        self._history: List[ResearchResult] = []

    @property
    def config(self) -> ResearchConfig:
        return self._config

    @property
    def history(self) -> List[ResearchResult]:
        """Previous research results from this agent session."""
        return list(self._history)

    def research(
        self,
        question: str,
        *,
        template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        """Execute a research workflow for the given question.

        Args:
            question: The research question to investigate.
            template: Optional planner template name.
            context: Additional context for planning and execution.

        Returns:
            A ResearchResult with task outcomes and report.
        """
        research_id = str(uuid.uuid4())[:12]
        start_time = time.time()

        # Plan
        graph = self._planner.plan(question, template=template, context=context)

        # Execute
        self._execute_graph(graph)

        # Report
        report: Optional[ResearchReport] = None
        if self._config.auto_report:
            results_data = [
                {
                    "task_name": t.name,
                    "tool": t.tool,
                    "action": t.action,
                    "status": t.status.value,
                    "data": t.result,
                }
                for t in graph.tasks
            ]
            report = self._reporter.generate(
                title=f"Research: {question[:80]}",
                question=question,
                results=results_data,
            )

        duration = time.time() - start_time
        status = "completed" if not graph.failed_tasks else "failed"

        result = ResearchResult(
            research_id=research_id,
            question=question,
            status=status,
            task_graph=graph,
            report=report,
            duration_seconds=duration,
        )
        self._history.append(result)
        return result

    def thesis(
        self,
        title: str,
        hypothesis: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        """Run thesis-mode research (hypothesis → conclusion pipeline).

        Args:
            title: Thesis title.
            hypothesis: The hypothesis to test.
            context: Additional context.
            data: Initial data for the pipeline.

        Returns:
            A ResearchResult with thesis outcome.
        """
        research_id = str(uuid.uuid4())[:12]
        start_time = time.time()

        thesis_result = self._thesis_engine.run(
            title=title, hypothesis=hypothesis, context=context, data=data
        )

        # Also generate a report from thesis stages
        results_data = [
            {
                "task_name": f"Stage: {s.stage.value}",
                "tool": "thesis_engine",
                "action": s.stage.value,
                "status": s.status,
                "data": s.data,
            }
            for s in thesis_result.stages
        ]
        report = self._reporter.generate(
            title=title,
            question=hypothesis,
            results=results_data,
        )

        duration = time.time() - start_time
        graph = TaskGraph()  # Empty graph for thesis mode

        result = ResearchResult(
            research_id=research_id,
            question=hypothesis,
            status="completed" if thesis_result.passed else "failed",
            task_graph=graph,
            report=report,
            thesis_result=thesis_result,
            duration_seconds=duration,
        )
        self._history.append(result)
        return result

    def _execute_graph(self, graph: TaskGraph) -> None:
        """Execute all tasks in the graph respecting dependencies."""
        deadline = time.time() + self._config.timeout_seconds

        while not graph.is_complete():
            if time.time() > deadline:
                # Mark remaining as failed
                for t in graph.tasks:
                    if t.status == TaskStatus.PENDING:
                        t.mark_failed("Timeout exceeded")
                break

            ready = graph.get_ready_tasks()
            if not ready:
                # Deadlock — mark remaining as failed
                for t in graph.tasks:
                    if t.status == TaskStatus.PENDING:
                        t.mark_failed("Deadlock: dependencies cannot be resolved")
                break

            for task in ready[: self._config.max_concurrent_tasks]:
                self._execute_task(task)

    def _execute_task(self, task: ResearchTask) -> None:
        """Execute a single task, dispatching to tool if available."""
        task.mark_running()
        try:
            if self._tool_dispatch:
                result = self._tool_dispatch(task.tool, task.action, task.params)
            else:
                # Default: simulate execution
                result = {
                    "summary": f"Executed {task.name} via {task.tool}/{task.action}",
                    "simulated": True,
                }
            task.mark_completed(result)
        except Exception as exc:
            task.mark_failed(str(exc))
            if self._config.strict_mode:
                raise
