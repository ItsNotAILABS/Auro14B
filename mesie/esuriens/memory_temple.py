"""Memory Temple — persistent storage for task horizons and sovereign artifacts.

The Memory Temple provides JSON-based persistence for the ESURIENS task system.
It stores the TASK_HORIZON state (immediate, short-term, long-term goals) so
they persist across restarts, and archives completed task chains as sovereign
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mesie.esuriens.task_horizon import HorizonLevel, TaskHorizon
from mesie.esuriens.sovereign_artifact import SovereignArtifact, TaskChain


class MemoryTemple:
    """Persistent storage layer for ESURIENS tasks and artifacts.

    Uses a JSON file-based approach for cross-restart persistence.
    Active tasks are loaded into memory (volatile) for fast access,
    while completed task chains are archived as sovereign artifacts
    on disk.

    Args:
        temple_path: Directory path for the Memory Temple storage.
    """

    HORIZON_FILE = "task_horizons.json"
    ARTIFACTS_DIR = "sovereign_artifacts"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, temple_path: str | Path) -> None:
        self.temple_path = Path(temple_path)
        self.temple_path.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir = self.temple_path / self.ARTIFACTS_DIR
        self._artifacts_dir.mkdir(exist_ok=True)
        self._manifest_path = self.temple_path / self.MANIFEST_FILE
        self._horizon_path = self.temple_path / self.HORIZON_FILE

    def save_horizons(self, tasks: List[TaskHorizon]) -> None:
        """Persist all task horizons to disk.

        Args:
            tasks: List of tasks to persist.
        """
        data = {
            "version": 1,
            "saved_at": time.time(),
            "tasks": [t.to_dict() for t in tasks],
        }
        self._horizon_path.write_text(json.dumps(data, indent=2))

    def load_horizons(self) -> List[TaskHorizon]:
        """Load persisted task horizons from disk.

        Returns:
            List of deserialized TaskHorizon objects.
        """
        if not self._horizon_path.exists():
            return []
        data = json.loads(self._horizon_path.read_text())
        return [TaskHorizon.from_dict(t) for t in data.get("tasks", [])]

    def archive_artifact(self, artifact: SovereignArtifact) -> str:
        """Archive a sovereign artifact to persistent storage.

        Args:
            artifact: The artifact to archive.

        Returns:
            The artifact file path.
        """
        artifact_data = artifact.to_dict()
        content = json.dumps(artifact_data, indent=2)
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        artifact.digest = digest
        artifact_data["digest"] = digest

        filename = f"{artifact.artifact_id}_{digest}.json"
        filepath = self._artifacts_dir / filename
        filepath.write_text(json.dumps(artifact_data, indent=2))

        self._update_manifest(artifact.artifact_id, filename, digest)
        return str(filepath)

    def load_artifact(self, artifact_id: str) -> Optional[SovereignArtifact]:
        """Load a sovereign artifact by ID.

        Args:
            artifact_id: The artifact's unique identifier.

        Returns:
            The deserialized artifact or None if not found.
        """
        manifest = self._load_manifest()
        entry = manifest.get("artifacts", {}).get(artifact_id)
        if not entry:
            return None
        filepath = self._artifacts_dir / entry["filename"]
        if not filepath.exists():
            return None
        data = json.loads(filepath.read_text())
        return SovereignArtifact.from_dict(data)

    def list_artifacts(self) -> List[str]:
        """List all archived artifact IDs.

        Returns:
            List of artifact IDs.
        """
        manifest = self._load_manifest()
        return list(manifest.get("artifacts", {}).keys())

    def verify_artifact_integrity(self, artifact_id: str) -> bool:
        """Verify the integrity of a stored artifact.

        Args:
            artifact_id: The artifact's unique identifier.

        Returns:
            True if the artifact content matches its stored digest.
        """
        manifest = self._load_manifest()
        entry = manifest.get("artifacts", {}).get(artifact_id)
        if not entry:
            return False
        filepath = self._artifacts_dir / entry["filename"]
        if not filepath.exists():
            return False
        data = json.loads(filepath.read_text())
        stored_digest = data.get("digest", "")
        data["digest"] = ""
        content = json.dumps(data, indent=2)
        computed = hashlib.sha256(content.encode()).hexdigest()[:16]
        return computed == stored_digest

    def clear(self) -> None:
        """Clear all stored data (for testing)."""
        if self._horizon_path.exists():
            self._horizon_path.unlink()
        for f in self._artifacts_dir.iterdir():
            f.unlink()
        if self._manifest_path.exists():
            self._manifest_path.unlink()

    def _update_manifest(self, artifact_id: str, filename: str, digest: str) -> None:
        """Update the manifest with a new artifact entry."""
        manifest = self._load_manifest()
        manifest.setdefault("artifacts", {})[artifact_id] = {
            "filename": filename,
            "digest": digest,
            "archived_at": time.time(),
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2))

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest file."""
        if not self._manifest_path.exists():
            return {"artifacts": {}}
        return json.loads(self._manifest_path.read_text())
