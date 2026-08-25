"""Content-addressed artifact storage for AURO missions.

Mission workers may write only inside one mission directory. Every output is
hashed and represented by an immutable record. A bundle is evidence of files
that were produced; it is not proof that the files are correct.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping
import zipfile


_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _safe_id(value: str) -> str:
    normalized = _SAFE_SEGMENT.sub("-", str(value).strip()).strip("-.")
    if not normalized:
        raise ValueError("identifier cannot be empty")
    return normalized[:160]


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    mission_id: str
    task_id: str | None
    relative_path: str
    media_type: str
    bytes: int
    sha256: str
    created_at_unix: int
    label: str = ""
    schema: str = "auro.mission.artifact.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactStore:
    """Atomic, path-safe artifact store with deterministic manifests."""

    def __init__(
        self,
        root: str | Path = "state/mission-artifacts",
        *,
        max_artifact_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_artifact_bytes = max(1024, int(max_artifact_bytes))

    def mission_root(self, mission_id: str) -> Path:
        path = (self.root / _safe_id(mission_id)).resolve()
        path.relative_to(self.root)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, mission_id: str, relative_path: str) -> tuple[Path, str]:
        relative = Path(str(relative_path).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative and cannot traverse parents")
        clean_parts = [_safe_id(part) for part in relative.parts if part not in {"", "."}]
        if not clean_parts:
            raise ValueError("artifact path cannot be empty")
        normalized = Path(*clean_parts)
        root = self.mission_root(mission_id)
        path = (root / normalized).resolve()
        path.relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, normalized.as_posix()

    def write_bytes(
        self,
        mission_id: str,
        relative_path: str,
        data: bytes,
        *,
        task_id: str | None = None,
        media_type: str | None = None,
        label: str = "",
    ) -> ArtifactRecord:
        payload = bytes(data)
        if len(payload) > self.max_artifact_bytes:
            raise ValueError(
                f"artifact exceeds {self.max_artifact_bytes} byte limit"
            )
        path, normalized = self._resolve(mission_id, relative_path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        actual_media = media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        digest = _sha256_file(path)
        artifact_id = "artifact_" + hashlib.sha256(
            _canonical(
                {
                    "mission_id": mission_id,
                    "task_id": task_id,
                    "path": normalized,
                    "sha256": digest,
                }
            )
        ).hexdigest()[:24]
        return ArtifactRecord(
            artifact_id=artifact_id,
            mission_id=mission_id,
            task_id=task_id,
            relative_path=normalized,
            media_type=actual_media,
            bytes=len(payload),
            sha256=digest,
            created_at_unix=int(time.time()),
            label=str(label),
        )

    def write_text(
        self,
        mission_id: str,
        relative_path: str,
        text: str,
        *,
        task_id: str | None = None,
        media_type: str = "text/plain; charset=utf-8",
        label: str = "",
    ) -> ArtifactRecord:
        return self.write_bytes(
            mission_id,
            relative_path,
            str(text).encode("utf-8"),
            task_id=task_id,
            media_type=media_type,
            label=label,
        )

    def write_json(
        self,
        mission_id: str,
        relative_path: str,
        value: Mapping[str, Any] | list[Any],
        *,
        task_id: str | None = None,
        label: str = "",
    ) -> ArtifactRecord:
        encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        return self.write_text(
            mission_id,
            relative_path,
            encoded,
            task_id=task_id,
            media_type="application/json",
            label=label,
        )

    def list_files(self, mission_id: str) -> list[dict[str, Any]]:
        root = self.mission_root(mission_id)
        rows: list[dict[str, Any]] = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.name.endswith(".tmp"):
                continue
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "media_type": mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream",
                }
            )
        return rows

    def manifest(
        self,
        mission_id: str,
        records: Iterable[ArtifactRecord] | None = None,
    ) -> dict[str, Any]:
        supplied = list(records or [])
        files = [record.to_dict() for record in supplied] if supplied else self.list_files(mission_id)
        payload: dict[str, Any] = {
            "schema": "auro.mission.artifact-manifest.v1",
            "mission_id": mission_id,
            "artifact_count": len(files),
            "artifacts": files,
            "generated_at_unix": int(time.time()),
            "claim_boundary": (
                "hashes prove artifact identity and custody in this store; "
                "they do not prove semantic correctness"
            ),
        }
        payload["manifest_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        return payload

    def build_bundle(
        self,
        mission_id: str,
        *,
        records: Iterable[ArtifactRecord] | None = None,
        filename: str = "mission-artifacts.zip",
    ) -> ArtifactRecord:
        root = self.mission_root(mission_id)
        manifest = self.manifest(mission_id, records)
        manifest_path = root / "ARTIFACT_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bundle_path, normalized = self._resolve(mission_id, filename)
        temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                if path in {bundle_path, temporary} or path.name.endswith(".tmp"):
                    continue
                archive.write(path, path.relative_to(root).as_posix())
        temporary.replace(bundle_path)
        digest = _sha256_file(bundle_path)
        return ArtifactRecord(
            artifact_id="artifact_" + digest[:24],
            mission_id=mission_id,
            task_id=None,
            relative_path=normalized,
            media_type="application/zip",
            bytes=bundle_path.stat().st_size,
            sha256=digest,
            created_at_unix=int(time.time()),
            label="mission artifact bundle",
        )
