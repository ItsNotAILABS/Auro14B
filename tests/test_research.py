"""Tests for the research module: agent, thesis engine, planner, reporter."""

import pytest

from mesie.research.agent import ResearchAgent, ResearchConfig, ResearchResult
from mesie.research.planner import ResearchPlanner, ResearchTask, TaskGraph, TaskStatus
from mesie.research.reporter import ReportFormat, ResearchReport, ResearchReporter
from mesie.research.thesis_engine import ThesisEngine, ThesisPipeline, ThesisResult, ThesisStage


class TestResearchPlanner:
    def test_plan_creates_task_graph(self):
        planner = ResearchPlanner()
        graph = planner.plan("What are the spectral properties of iron?")
        assert isinstance(graph, TaskGraph)
        assert len(graph.tasks) > 0

    def test_plan_with_template(self):
        planner = ResearchPlanner()
        graph = planner.plan("Test hypothesis", template="hypothesis_test")
        assert len(graph.tasks) == 4
        assert graph.tasks[0].name == "Formulate Hypothesis"

    def test_plan_infers_literature_template(self):
        planner = ResearchPlanner()
        graph = planner.plan("Survey of machine learning in spectroscopy")
        assert graph.tasks[0].name == "Search Literature"

    def test_task_graph_ready_tasks(self):
        graph = TaskGraph()
        t1 = ResearchTask(task_id="t1", name="First")
        t2 = ResearchTask(task_id="t2", name="Second", depends_on=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t1"

    def test_task_graph_completion(self):
        graph = TaskGraph()
        t1 = ResearchTask(task_id="t1", name="Only task")
        graph.add_task(t1)
        assert not graph.is_complete()
        t1.mark_completed({"done": True})
        assert graph.is_complete()


class TestThesisEngine:
    def test_run_default_pipeline(self):
        engine = ThesisEngine()
        result = engine.run(
            title="Test Thesis",
            hypothesis="X is greater than Y",
        )
        assert isinstance(result, ThesisResult)
        assert result.thesis_id
        assert result.passed is True
        assert len(result.stages) == 8

    def test_strict_gates_fail(self):
        def failing_handler(stage, data):
            from mesie.research.thesis_engine import StageResult
            return StageResult(stage=stage, status="failed", notes="Deliberate failure")

        pipeline = ThesisPipeline()
        pipeline.set_handler(ThesisStage.METHODOLOGY, failing_handler)
        engine = ThesisEngine(pipeline=pipeline, strict_gates=True)
        result = engine.run("Test", "H0")
        assert result.passed is False

    def test_conclusion_synthesis(self):
        engine = ThesisEngine()
        result = engine.run("T", "All stages pass")
        assert "supports" in result.conclusion.lower() or "evidence" in result.conclusion.lower()


class TestResearchReporter:
    def test_generate_markdown_report(self):
        reporter = ResearchReporter(default_format=ReportFormat.MARKDOWN)
        results = [
            {"task_name": "Step 1", "tool": "stats", "action": "analyze",
             "status": "completed", "data": {"summary": "Found patterns"}},
        ]
        report = reporter.generate("Test Report", "What is X?", results)
        assert isinstance(report, ResearchReport)
        assert report.title == "Test Report"
        assert len(report.sections) == 4

    def test_render_json(self):
        reporter = ResearchReporter()
        report = ResearchReport(title="T", abstract="A", sections=[])
        output = reporter.render(report, format=ReportFormat.JSON)
        assert '"title": "T"' in output

    def test_render_latex(self):
        reporter = ResearchReporter()
        report = ResearchReport(title="T", abstract="A", sections=[])
        output = reporter.render(report, format=ReportFormat.LATEX)
        assert r"\documentclass" in output


class TestResearchAgent:
    def test_research_workflow(self):
        agent = ResearchAgent()
        result = agent.research("What is the hardness of diamond?")
        assert isinstance(result, ResearchResult)
        assert result.status in ("completed", "failed")
        assert result.report is not None

    def test_thesis_mode(self):
        agent = ResearchAgent()
        result = agent.thesis(
            title="Diamond Hardness",
            hypothesis="Diamond is the hardest natural material",
        )
        assert result.thesis_result is not None
        assert result.thesis_result.passed is True

    def test_history_tracked(self):
        agent = ResearchAgent()
        agent.research("Q1")
        agent.research("Q2")
        assert len(agent.history) == 2

    def test_custom_dispatch(self):
        def my_dispatch(tool, action, params):
            return {"custom": True, "tool": tool}

        agent = ResearchAgent(tool_dispatch=my_dispatch)
        result = agent.research("Test with custom dispatch")
        completed = result.task_graph.completed_tasks
        assert len(completed) > 0
        assert completed[0].result.get("custom") is True
