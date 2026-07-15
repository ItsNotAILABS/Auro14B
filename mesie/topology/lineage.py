"""Record lineage tracking and provenance."""

from __future__ import annotations

from typing import Dict, List, Optional

from mesie.core.records import MultiElementRecord


class RecordLineageTracker:
    """Track lineage and provenance of spectral records.

    Maintains a history of operations applied to records for
    audit trails and reproducibility.
    """

    def __init__(self) -> None:
        self._history: Dict[str, List[str]] = {}

    def register(self, record: MultiElementRecord) -> None:
        """Register a record in the lineage tracker.

        Args:
            record: Record to register.
        """
        self._history[record.record_id] = list(record.lineage)

    def add_operation(self, record_id: str, operation: str) -> None:
        """Add an operation to a record's lineage.

        Args:
            record_id: Record identifier.
            operation: Operation description.
        """
        if record_id not in self._history:
            self._history[record_id] = []
        self._history[record_id].append(operation)

    def get_lineage(self, record_id: str) -> List[str]:
        """Get the full lineage for a record.

        Args:
            record_id: Record identifier.

        Returns:
            List of lineage operations.
        """
        return list(self._history.get(record_id, []))

    def get_all_records(self) -> List[str]:
        """Get all tracked record identifiers.

        Returns:
            List of record IDs.
        """
        return list(self._history.keys())
