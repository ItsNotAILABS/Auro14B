"""AURO constitutional checkpoint substrate.

Turns checkpoint doctrine into executable identity, custody, promotion, continuity,
and rollback gates. The authoritative proof remains content-addressed and external
to the model weights; learned checkpoint awareness may reference this lineage but
cannot self-certify it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROTOCOLS = (
    "canonical_checkpoint_identity",
    "weight_shard_custody",
    "training_resume_determinism",
    "matched_optimization_experiment",
    "capability_promotion_regression",
    "continual_learning_forgetting_control",
    "structured_architecture_introduction",
    "mcp_tool_capability_checkpoint",
    "organism_state_identity_continuity",
    "quarantine_rollback_recovery",
)

AIOPS_DOMAINS = (
    "artificial_identity_operations",
    "verifiable_safe_state_reversion",
    "modular_growth_governance",
    "economic_interface_continuity",
)


class ConstitutionalGateError(RuntimeError):
    """Raised when checkpoint identity, custody, or promotion proof fails."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class ProtocolAssertion:
    protocol: str
    passed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)
    blocker: Optional[str] = None


@dataclass
class ConstitutionalCheckpoint:
    checkpoint_id: str
    checkpoint_class: str
    model_id: str
    parent_checkpoint_id: Optional[str]
    created_at_unix: int
    files: Dict[str, Dict[str, Any]]
    protocols: List[ProtocolAssertion]
    aiops_domains: List[str]
    optimization: Dict[str, Any]
    identity: Dict[str, Any]
    rollback: Dict[str, Any]
    capabilities: Dict[str, Any]
    promotion_status: str = "quarantined"
    schema: str = "auro.substrate.checkpoint.v1"
    manifest_sha256: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data["protocols"] = [asdict(item) for item in self.protocols]
        data["manifest_sha256"] = None
        return data

    def seal(self) -> str:
        self.manifest_sha256 = _sha256_bytes(_canonical_json(self.payload()))
        return self.manifest_sha256

    def to_dict(self) -> Dict[str, Any]:
        if self.manifest_sha256 is None:
            self.seal()
        data = asdict(self)
        data["protocols"] = [asdict(item) for item in self.protocols]
        return data


def inventory_files(
    root: str | Path, relative_paths: Iterable[str]
) -> Dict[str, Dict[str, Any]]:
    root = Path(root)
    inventory: Dict[str, Dict[str, Any]] = {}
    for relative in sorted(set(relative_paths)):
        path = root / relative
        if not path.is_file():
            raise ConstitutionalGateError(f"missing checkpoint artifact: {relative}")
        inventory[relative] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return inventory


def validate_inventory(
    root: str | Path, inventory: Mapping[str, Mapping[str, Any]]
) -> None:
    root = Path(root)
    for relative, expected in inventory.items():
        path = root / relative
        if not path.is_file():
            raise ConstitutionalGateError(
                f"checkpoint artifact disappeared: {relative}"
            )
        if _sha256_file(path) != expected.get("sha256"):
            raise ConstitutionalGateError(
                f"checkpoint artifact hash mismatch: {relative}"
            )


def _assertions(
    checkpoint_class: str, evidence: Mapping[str, Any]
) -> List[ProtocolAssertion]:
    checkpoint_class = checkpoint_class.lower()
    output: List[ProtocolAssertion] = []
    for protocol in PROTOCOLS:
        passed = True
        blocker: Optional[str] = None
        if protocol == "canonical_checkpoint_identity":
            passed = bool(evidence.get("checkpoint_id") and evidence.get("model_id"))
            blocker = None if passed else "checkpoint and model identity are required"
        elif protocol == "weight_shard_custody":
            passed = bool(
                evidence.get("file_count", 0) > 0
                and evidence.get("all_files_hashed")
            )
            blocker = None if passed else "every persisted artifact must be hashed"
        elif protocol == "training_resume_determinism":
            passed = checkpoint_class not in {"training_state", "organism"} or bool(
                evidence.get("resume_state_present")
            )
            blocker = None if passed else "resume state is required"
        elif protocol == "matched_optimization_experiment":
            passed = not evidence.get("optimization_candidate") or bool(
                evidence.get("matched_benchmark")
            )
            blocker = None if passed else "matched benchmark evidence is required"
        elif protocol == "capability_promotion_regression":
            passed = not evidence.get("promotion_requested") or bool(
                evidence.get("protected_capabilities_pass")
            )
            blocker = None if passed else "protected capabilities did not pass"
        elif protocol == "continual_learning_forgetting_control":
            passed = not evidence.get("continual_learning") or bool(
                evidence.get("replay_or_forgetting_eval")
            )
            blocker = None if passed else "forgetting evaluation is required"
        elif protocol == "structured_architecture_introduction":
            passed = not evidence.get("architecture_changed") or bool(
                evidence.get("reversible_module_boundary")
            )
            blocker = None if passed else "architecture growth must be reversible"
        elif protocol == "mcp_tool_capability_checkpoint":
            passed = not evidence.get("tool_capabilities_present") or bool(
                evidence.get("tool_registry_receipt")
            )
            blocker = None if passed else "tool registry receipt is required"
        elif protocol == "organism_state_identity_continuity":
            passed = checkpoint_class != "organism" or bool(
                evidence.get("identity_state_present")
                and evidence.get("continuity_parent_known")
            )
            blocker = None if passed else "organism identity and lineage are required"
        elif protocol == "quarantine_rollback_recovery":
            passed = bool(
                evidence.get("rollback_target")
                and evidence.get("rollback_verified")
            )
            blocker = None if passed else "verified safe rollback target is required"
        output.append(ProtocolAssertion(protocol, passed, dict(evidence), blocker))
    return output


def decide_promotion(
    protocols: Iterable[ProtocolAssertion], requested: bool
) -> str:
    if not requested:
        return "quarantined"
    return "promoted" if all(item.passed for item in protocols) else "quarantined"


def build_constitutional_checkpoint(
    *,
    root: str | Path,
    checkpoint_id: str,
    checkpoint_class: str,
    model_id: str,
    files: Iterable[str],
    parent_checkpoint_id: Optional[str] = None,
    optimization: Optional[Mapping[str, Any]] = None,
    identity: Optional[Mapping[str, Any]] = None,
    rollback: Optional[Mapping[str, Any]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Mapping[str, Any]] = None,
    promotion_requested: bool = False,
) -> ConstitutionalCheckpoint:
    file_inventory = inventory_files(root, files)
    merged_evidence = dict(evidence or {})
    merged_evidence.update(
        {
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "file_count": len(file_inventory),
            "all_files_hashed": True,
            "promotion_requested": promotion_requested,
        }
    )
    protocols = _assertions(checkpoint_class, merged_evidence)
    checkpoint = ConstitutionalCheckpoint(
        checkpoint_id=checkpoint_id,
        checkpoint_class=checkpoint_class,
        model_id=model_id,
        parent_checkpoint_id=parent_checkpoint_id,
        created_at_unix=int(time.time()),
        files=file_inventory,
        protocols=protocols,
        aiops_domains=list(AIOPS_DOMAINS),
        optimization=dict(optimization or {}),
        identity=dict(identity or {}),
        rollback=dict(rollback or {}),
        capabilities=dict(capabilities or {}),
        promotion_status=decide_promotion(protocols, promotion_requested),
    )
    checkpoint.seal()
    return checkpoint


def write_constitutional_manifest(
    root: str | Path,
    checkpoint: ConstitutionalCheckpoint,
    filename: str = "constitutional_manifest.json",
) -> Path:
    path = Path(root) / filename
    path.write_text(
        json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_and_verify_constitutional_manifest(
    root: str | Path, filename: str = "constitutional_manifest.json"
) -> Dict[str, Any]:
    root = Path(root)
    data = json.loads((root / filename).read_text(encoding="utf-8"))
    supplied = data.get("manifest_sha256")
    unsigned = dict(data)
    unsigned["manifest_sha256"] = None
    actual = _sha256_bytes(_canonical_json(unsigned))
    if supplied != actual:
        raise ConstitutionalGateError("constitutional manifest seal mismatch")
    validate_inventory(root, data.get("files") or {})
    failed = [item for item in data.get("protocols", []) if not item.get("passed")]
    data["verified"] = True
    data["failed_protocols"] = [item.get("protocol") for item in failed]
    return data


def require_promotable(manifest: Mapping[str, Any]) -> None:
    if manifest.get("promotion_status") != "promoted":
        failed = manifest.get("failed_protocols") or [
            item.get("protocol")
            for item in manifest.get("protocols", [])
            if not item.get("passed")
        ]
        raise ConstitutionalGateError(
            f"checkpoint remains quarantined; blockers={failed}"
        )
