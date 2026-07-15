"""Validation engine — multi-level spectral validation (6 levels).

Provides record validation, batch validation, level-specific checks,
and validation summaries for production spectral intelligence workflows.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record
from mesie.validation.validators import ValidationReport, validate_record


class ValidationEngine(Engine):
    """Multi-level spectral validation engine.

    Capabilities:
        validate: Full 6-level validation of a single record.
        batch_validate: Validate multiple records and return summary.
        check_level: Validate to a specific level only.
        summary: Return validation statistics from session.
        reset: Clear session statistics.
    """

    name = "validation"
    capabilities = ["validate", "batch_validate", "check_level", "summary", "reset"]

    def __init__(self) -> None:
        self._stats: Dict[str, int] = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "by_level": {str(i): 0 for i in range(7)},
        }

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "validate":
                return self._handle_validate(message.payload)
            elif action == "batch_validate":
                return self._handle_batch_validate(message.payload)
            elif action == "check_level":
                return self._handle_check_level(message.payload)
            elif action == "summary":
                return self._handle_summary()
            elif action == "reset":
                return self._handle_reset()
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _handle_validate(self, payload: Dict) -> EngineResponse:
        """Full 6-level validation of a single record."""
        rec = load_record(payload["record"])
        report = validate_record(rec)
        self._update_stats(report)
        return EngineResponse(
            True,
            self.name,
            "validate",
            {
                "is_valid": report.is_valid,
                "level": report.level,
                "max_level": 6,
                "errors": report.errors[:10],
                "warnings": report.warnings[:10],
            },
        )

    def _handle_batch_validate(self, payload: Dict) -> EngineResponse:
        """Validate multiple records and return aggregate results."""
        records = payload.get("records", [])
        if not records:
            return EngineResponse(False, self.name, "batch_validate", error="No records provided")

        results: List[Dict] = []
        passed = 0
        failed = 0
        levels: List[int] = []

        for i, raw_record in enumerate(records):
            try:
                rec = load_record(raw_record)
                report = validate_record(rec)
                self._update_stats(report)
                results.append({
                    "index": i,
                    "is_valid": report.is_valid,
                    "level": report.level,
                    "error_count": len(report.errors),
                    "warning_count": len(report.warnings),
                })
                levels.append(report.level)
                if report.is_valid:
                    passed += 1
                else:
                    failed += 1
            except (KeyError, TypeError, ValueError) as exc:
                results.append({"index": i, "is_valid": False, "level": 0, "error": str(exc)})
                failed += 1
                levels.append(0)

        return EngineResponse(
            True,
            self.name,
            "batch_validate",
            {
                "total": len(records),
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / max(len(records), 1),
                "mean_level": sum(levels) / max(len(levels), 1),
                "min_level": min(levels) if levels else 0,
                "max_level_achieved": max(levels) if levels else 0,
                "results": results[:50],  # Cap detailed results
            },
        )

    def _handle_check_level(self, payload: Dict) -> EngineResponse:
        """Validate a record to a specific target level."""
        target_level = payload.get("level", 6)
        rec = load_record(payload["record"])
        report = validate_record(rec)
        self._update_stats(report)

        meets_level = report.level >= target_level
        return EngineResponse(
            True,
            self.name,
            "check_level",
            {
                "target_level": target_level,
                "achieved_level": report.level,
                "meets_target": meets_level,
                "is_valid": report.is_valid,
                "errors": report.errors[:5] if not meets_level else [],
                "warnings": report.warnings[:5],
            },
        )

    def _handle_summary(self) -> EngineResponse:
        """Return validation session statistics."""
        return EngineResponse(
            True,
            self.name,
            "summary",
            {
                "total_validated": self._stats["total"],
                "passed": self._stats["passed"],
                "failed": self._stats["failed"],
                "pass_rate": self._stats["passed"] / max(self._stats["total"], 1),
                "level_distribution": dict(self._stats["by_level"]),
            },
        )

    def _handle_reset(self) -> EngineResponse:
        """Clear session statistics."""
        self._stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "by_level": {str(i): 0 for i in range(7)},
        }
        return EngineResponse(True, self.name, "reset", {"cleared": True})

    def _update_stats(self, report: ValidationReport) -> None:
        """Update internal validation statistics."""
        self._stats["total"] += 1
        if report.is_valid:
            self._stats["passed"] += 1
        else:
            self._stats["failed"] += 1
        self._stats["by_level"][str(report.level)] += 1