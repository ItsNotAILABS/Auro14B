"""Validation reporting utilities."""

from __future__ import annotations

from typing import List

from mesie.validation.validators import ValidationReport


def format_report(report: ValidationReport) -> str:
    """Format a validation report as a human-readable string.

    Args:
        report: ValidationReport to format.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append(f"Validation: {'PASSED' if report.is_valid else 'FAILED'}")
    lines.append(f"Level: {report.level}/6")

    if report.errors:
        lines.append("\nErrors:")
        for e in report.errors:
            lines.append(f"  - {e}")

    if report.warnings:
        lines.append("\nWarnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def batch_validate_summary(reports: List[ValidationReport]) -> str:
    """Summarize a batch of validation reports.

    Args:
        reports: List of ValidationReports.

    Returns:
        Summary string.
    """
    total = len(reports)
    valid = sum(1 for r in reports if r.is_valid)
    return f"Validated {total} records: {valid} passed, {total - valid} failed"
