"""Structured research report generation.

Auto-generates reports in Markdown, JSON, or LaTeX from research results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReportFormat(str, Enum):
    """Output format for research reports."""

    MARKDOWN = "markdown"
    JSON = "json"
    LATEX = "latex"


@dataclass
class ReportSection:
    """A single section within a research report."""

    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None


@dataclass
class ResearchReport:
    """Complete research report with metadata and structured content."""

    title: str
    abstract: str
    sections: List[ReportSection] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "sections": [
                {"title": s.title, "content": s.content, "data": s.data}
                for s in self.sections
            ],
            "citations": self.citations,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class ResearchReporter:
    """Generates structured research reports from task results.

    Args:
        default_format: Default output format for reports.
        include_raw_data: Whether to include raw data in report sections.
    """

    def __init__(
        self,
        default_format: ReportFormat = ReportFormat.MARKDOWN,
        include_raw_data: bool = False,
    ) -> None:
        self._default_format = default_format
        self._include_raw_data = include_raw_data

    def generate(
        self,
        title: str,
        question: str,
        results: List[Dict[str, Any]],
        *,
        format: Optional[ReportFormat] = None,
        citations: Optional[List[Dict[str, str]]] = None,
    ) -> ResearchReport:
        """Generate a research report from collected results.

        Args:
            title: Report title.
            question: The original research question.
            results: List of task result dictionaries.
            format: Output format override.
            citations: Optional citation list.

        Returns:
            A structured ResearchReport.
        """
        fmt = format or self._default_format
        sections: List[ReportSection] = []

        # Introduction
        sections.append(ReportSection(
            title="Introduction",
            content=f"Research question: {question}",
        ))

        # Methods / Steps
        method_content = "\n".join(
            f"- Step {i + 1}: {r.get('task_name', 'unnamed')} "
            f"(tool: {r.get('tool', 'N/A')})"
            for i, r in enumerate(results)
        )
        sections.append(ReportSection(
            title="Methods",
            content=method_content or "No methods recorded.",
        ))

        # Results
        findings: List[str] = []
        for r in results:
            if r.get("status") == "completed" and r.get("data"):
                summary = r["data"].get("summary", str(r["data"]))
                findings.append(f"- {r.get('task_name', 'Step')}: {summary}")

        sections.append(ReportSection(
            title="Results",
            content="\n".join(findings) if findings else "No results collected.",
            data=results if self._include_raw_data else None,
        ))

        # Conclusion
        sections.append(ReportSection(
            title="Conclusion",
            content=self._synthesize_conclusion(question, findings),
        ))

        report = ResearchReport(
            title=title,
            abstract=f"Automated research report for: {question}",
            sections=sections,
            citations=citations or [],
            metadata={
                "format": fmt.value,
                "num_steps": len(results),
                "question": question,
            },
        )
        return report

    def render(self, report: ResearchReport, format: Optional[ReportFormat] = None) -> str:
        """Render a report to string in the specified format.

        Args:
            report: The report to render.
            format: Output format (defaults to reporter's default).

        Returns:
            Formatted string output.
        """
        fmt = format or self._default_format
        if fmt == ReportFormat.JSON:
            return json.dumps(report.to_dict(), indent=2)
        elif fmt == ReportFormat.LATEX:
            return self._render_latex(report)
        else:
            return self._render_markdown(report)

    def _render_markdown(self, report: ResearchReport) -> str:
        lines = [f"# {report.title}", "", f"**Abstract:** {report.abstract}", ""]
        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        if report.citations:
            lines.append("## References")
            lines.append("")
            for i, cite in enumerate(report.citations, 1):
                lines.append(
                    f"{i}. {cite.get('authors', 'Unknown')}. "
                    f"*{cite.get('title', 'Untitled')}*. "
                    f"{cite.get('source', '')}, {cite.get('year', '')}."
                )
            lines.append("")
        return "\n".join(lines)

    def _render_latex(self, report: ResearchReport) -> str:
        lines = [
            r"\documentclass{article}",
            r"\begin{document}",
            f"\\title{{{report.title}}}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            report.abstract,
            r"\end{abstract}",
            "",
        ]
        for section in report.sections:
            lines.append(f"\\section{{{section.title}}}")
            lines.append(section.content)
            lines.append("")
        lines.append(r"\end{document}")
        return "\n".join(lines)

    def _synthesize_conclusion(self, question: str, findings: List[str]) -> str:
        if not findings:
            return f"Further investigation is needed to address: {question}"
        return (
            f"Based on {len(findings)} result(s), initial findings suggest "
            f"progress toward answering: {question}. "
            "Additional analysis may refine these conclusions."
        )
