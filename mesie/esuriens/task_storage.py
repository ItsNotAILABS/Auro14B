"""ESURIENS Task Storage — hybrid volatile/persistent task management.

Active tasks are held in volatile memory for fast access and mutation.
When task chains complete, they are archived as sovereign artifacts in
the Memory Temple for permanent persistence.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from mesie.esuriens.task_horizon import HorizonLevel, TaskHorizon
from mesie.esuriens.memory_temple import MemoryTemple
from mesie.esuriens.sovereign_artifact import SovereignArtifact, TaskChain


class EsuriensTaskStorage:
    """Hybrid task storage with volatile active tasks and persistent archives.

    Active tasks live in memory for low-latency access and mutation.
    When a task chain is fully completed, it is archived as a sovereign
    artifact in the Memory Temple and removed from volatile storage.

    On initialization, any previously persisted horizons are restored
    from the Memory Temple, enabling restart persistence.

    Args:
        temple: MemoryTemple instance for persistence.
        auto_persist: Whether to auto-save horizons on mutations.
    """

    def __init__(self, temple: MemoryTemple, auto_persist: bool = True) -> None:
        self.temple = temple
        self.auto_persist = auto_persist
        self._volatile: Dict[str, TaskHorizon] = {}
        self._restore_from_temple()

    def _restore_from_temple(self) -> None:
        """Restore active tasks from the Memory Temple."""
        tasks = self.temple.load_horizons()
        for task in tasks:
            if task.is_active:
                self._volatile[task.task_id] = task

    def _persist(self) -> None:
        """Persist current volatile tasks to the Memory Temple."""
        if self.auto_persist:
            self.temple.save_horizons(list(self._volatile.values()))

    def add_task(self, task: TaskHorizon) -> str:
        """Add a new task to volatile storage.

        Args:
            task: The task to add.

        Returns:
            The task ID.
        """
        self._volatile[task.task_id] = task
        self._persist()
        return task.task_id

    def get_task(self, task_id: str) -> Optional[TaskHorizon]:
        """Get a task by ID from volatile storage.

        Args:
            task_id: The task identifier.

        Returns:
            The task or None if not found.
        """
        return self._volatile.get(task_id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from volatile storage.

        Args:
            task_id: The task to remove.

        Returns:
            True if removed, False if not found.
        """
        if task_id in self._volatile:
            del self._volatile[task_id]
            self._persist()
            return True
        return False

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed.

        Args:
            task_id: The task to complete.

        Returns:
            True if completed successfully, False if not found.
        """
        task = self._volatile.get(task_id)
        if task is None:
            return False
        task.complete()
        self._persist()
        return True

    def fail_task(self, task_id: str) -> bool:
        """Mark a task as failed.

        Args:
            task_id: The task to fail.

        Returns:
            True if marked, False if not found.
        """
        task = self._volatile.get(task_id)
        if task is None:
            return False
        task.fail()
        self._persist()
        return True

    def activate_task(self, task_id: str) -> bool:
        """Activate a pending task.

        Args:
            task_id: The task to activate.

        Returns:
            True if activated, False if not found.
        """
        task = self._volatile.get(task_id)
        if task is None:
            return False
        task.activate()
        self._persist()
        return True

    def get_by_horizon(self, horizon: HorizonLevel) -> List[TaskHorizon]:
        """Get all volatile tasks at a given horizon level.

        Args:
            horizon: The horizon level to filter by.

        Returns:
            List of tasks matching the horizon.
        """
        return [t for t in self._volatile.values() if t.horizon == horizon]

    def get_immediate(self) -> List[TaskHorizon]:
        """Get all immediate-horizon tasks."""
        return self.get_by_horizon(HorizonLevel.IMMEDIATE)

    def get_short_term(self) -> List[TaskHorizon]:
        """Get all short-term-horizon tasks."""
        return self.get_by_horizon(HorizonLevel.SHORT_TERM)

    def get_long_term(self) -> List[TaskHorizon]:
        """Get all long-term-horizon tasks."""
        return self.get_by_horizon(HorizonLevel.LONG_TERM)

    def get_active_tasks(self) -> List[TaskHorizon]:
        """Get all tasks currently in active/pending state."""
        return [t for t in self._volatile.values() if t.is_active]

    def get_completed_tasks(self) -> List[TaskHorizon]:
        """Get all tasks in terminal state still in volatile storage."""
        return [t for t in self._volatile.values() if t.is_terminal]

    @property
    def task_count(self) -> int:
        """Total number of tasks in volatile storage."""
        return len(self._volatile)

    def archive_completed_chain(self, task_ids: List[str]) -> Optional[SovereignArtifact]:
        """Archive a completed task chain as a sovereign artifact.

        Moves a set of completed tasks from volatile storage into
        a permanent sovereign artifact in the Memory Temple.

        Args:
            task_ids: Ordered list of task IDs forming the chain.

        Returns:
            The created SovereignArtifact, or None if tasks aren't all completed.
        """
        tasks = []
        for tid in task_ids:
            task = self._volatile.get(tid)
            if task is None or not task.is_terminal:
                return None
            tasks.append(task)

        horizon_levels = list(set(t.horizon.value for t in tasks))
        root_task = tasks[0] if tasks else None

        chain = TaskChain(
            task_ids=task_ids,
            root_task_id=root_task.task_id if root_task else None,
            horizon_levels=horizon_levels,
            created_at=tasks[0].created_at if tasks else time.time(),
            completed_at=tasks[-1].completed_at if tasks else time.time(),
            total_duration=(
                (tasks[-1].completed_at or time.time()) - tasks[0].created_at
                if tasks
                else 0.0
            ),
        )

        artifact = SovereignArtifact(
            chain=chain,
            sovereignty_level=min(len(tasks) // 3, 2),
        )

        self.temple.archive_artifact(artifact)

        # Remove archived tasks from volatile storage
        for tid in task_ids:
            task = self._volatile.get(tid)
            if task:
                task.archive()
            self._volatile.pop(tid, None)
        self._persist()

        return artifact

    def flush_completed(self) -> List[SovereignArtifact]:
        """Archive all completed task chains and return artifacts.

        Finds all completed tasks, groups them by chain (parent_id),
        and archives each chain.

        Returns:
            List of created sovereign artifacts.
        """
        completed = self.get_completed_tasks()
        if not completed:
            return []

        # Group by parent chain
        chains: Dict[str, List[str]] = {}
        standalone: List[str] = []

        for task in completed:
            if task.parent_id and task.parent_id in self._volatile:
                chains.setdefault(task.parent_id, [task.parent_id]).append(task.task_id)
            elif task.parent_id:
                chains.setdefault(task.parent_id, []).append(task.task_id)
            else:
                standalone.append(task.task_id)

        artifacts = []

        for chain_tasks in chains.values():
            # Only archive if all tasks in chain are terminal
            all_terminal = all(
                self._volatile.get(tid) and self._volatile[tid].is_terminal
                for tid in chain_tasks
                if tid in self._volatile
            )
            if all_terminal and chain_tasks:
                valid_ids = [tid for tid in chain_tasks if tid in self._volatile]
                if valid_ids:
                    art = self.archive_completed_chain(valid_ids)
                    if art:
                        artifacts.append(art)

        # Archive standalone completed tasks individually
        for tid in standalone:
            if tid in self._volatile:
                art = self.archive_completed_chain([tid])
                if art:
                    artifacts.append(art)

        return artifacts

    def snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of the current volatile state.

        Returns:
            Dictionary with task counts and horizon breakdowns.
        """
        return {
            "total_tasks": self.task_count,
            "immediate": len(self.get_immediate()),
            "short_term": len(self.get_short_term()),
            "long_term": len(self.get_long_term()),
            "active": len(self.get_active_tasks()),
            "completed": len(self.get_completed_tasks()),
            "archived_artifacts": len(self.temple.list_artifacts()),
        }
