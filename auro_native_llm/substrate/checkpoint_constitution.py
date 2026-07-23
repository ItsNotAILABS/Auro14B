"""AURO constitutional checkpoint substrate with integrity and authorization."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROTOCOLS = (
    "canonical_checkpoint_identity", "weight_shard_custody", "training_resume_determinism",
    "matched_optimization_experiment", "capability_promotion_regression",
    "continual_learning_forgetting_control", "structured_architecture_introduction",
    "mcp_tool_capability_checkpoint", "organism_state_identity_continuity",
    "quarantine_rollback_recovery",
)
AIOPS_DOMAINS = (
    "artificial_identity_operations", "verifiable_safe_state_reversion",
    "modular_growth_governance", "economic_interface_continuity",
)

class ConstitutionalGateError(RuntimeError):
    pass

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

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
    schema: str = "auro.substrate.checkpoint.v2"
    manifest_sha256: Optional[str] = None
    authorization_hmac_sha256: Optional[str] = None
    authorized_by: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data["protocols"] = [asdict(item) for item in self.protocols]
        data["manifest_sha256"] = None
        data["authorization_hmac_sha256"] = None
        return data

    def seal(self, signing_key: Optional[str] = None, authorized_by: Optional[str] = None) -> str:
        self.authorized_by = authorized_by
        canonical = _canonical_json(self.payload())
        self.manifest_sha256 = _sha256_bytes(canonical)
        if signing_key:
            self.authorization_hmac_sha256 = hmac.new(signing_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        elif self.promotion_status == "promoted":
            raise ConstitutionalGateError("promoted checkpoint requires an authorization signing key")
        return self.manifest_sha256

    def to_dict(self) -> Dict[str, Any]:
        if self.manifest_sha256 is None:
            self.seal()
        data = asdict(self)
        data["protocols"] = [asdict(item) for item in self.protocols]
        return data

def inventory_files(root: str | Path, relative_paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    root = Path(root)
    inventory: Dict[str, Dict[str, Any]] = {}
    for relative in sorted(set(relative_paths)):
        path = root / relative
        if not path.is_file():
            raise ConstitutionalGateError(f"missing checkpoint artifact: {relative}")
        inventory[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return inventory

def validate_inventory(root: str | Path, inventory: Mapping[str, Mapping[str, Any]]) -> None:
    root = Path(root)
    for relative, expected in inventory.items():
        path = root / relative
        if not path.is_file() or _sha256_file(path) != expected.get("sha256"):
            raise ConstitutionalGateError(f"checkpoint artifact hash mismatch: {relative}")

def _assertions(checkpoint_class: str, evidence: Mapping[str, Any]) -> List[ProtocolAssertion]:
    checks = {
        "canonical_checkpoint_identity": bool(evidence.get("checkpoint_id") and evidence.get("model_id")),
        "weight_shard_custody": bool(evidence.get("file_count", 0) > 0 and evidence.get("all_files_hashed")),
        "training_resume_determinism": checkpoint_class.lower() not in {"training_state", "organism"} or bool(evidence.get("resume_state_present")),
        "matched_optimization_experiment": not evidence.get("optimization_candidate") or bool(evidence.get("matched_benchmark")),
        "capability_promotion_regression": not evidence.get("promotion_requested") or bool(evidence.get("protected_capabilities_pass")),
        "continual_learning_forgetting_control": not evidence.get("continual_learning") or bool(evidence.get("replay_or_forgetting_eval")),
        "structured_architecture_introduction": not evidence.get("architecture_changed") or bool(evidence.get("reversible_module_boundary")),
        "mcp_tool_capability_checkpoint": not evidence.get("tool_capabilities_present") or bool(evidence.get("tool_registry_receipt")),
        "organism_state_identity_continuity": checkpoint_class.lower() != "organism" or bool(evidence.get("identity_state_present") and evidence.get("continuity_parent_known")),
        "quarantine_rollback_recovery": bool(evidence.get("rollback_target") and evidence.get("rollback_verified")),
    }
    return [ProtocolAssertion(name, checks[name], dict(evidence), None if checks[name] else f"{name} evidence failed") for name in PROTOCOLS]

def decide_promotion(protocols: Iterable[ProtocolAssertion], requested: bool) -> str:
    return "promoted" if requested and all(item.passed for item in protocols) else "quarantined"

def build_constitutional_checkpoint(*, root: str | Path, checkpoint_id: str, checkpoint_class: str,
    model_id: str, files: Iterable[str], parent_checkpoint_id: Optional[str] = None,
    optimization: Optional[Mapping[str, Any]] = None, identity: Optional[Mapping[str, Any]] = None,
    rollback: Optional[Mapping[str, Any]] = None, capabilities: Optional[Mapping[str, Any]] = None,
    evidence: Optional[Mapping[str, Any]] = None, promotion_requested: bool = False,
    signing_key: Optional[str] = None, authorized_by: Optional[str] = None) -> ConstitutionalCheckpoint:
    file_inventory = inventory_files(root, files)
    merged = dict(evidence or {})
    merged.update({"checkpoint_id": checkpoint_id, "model_id": model_id, "file_count": len(file_inventory), "all_files_hashed": True, "promotion_requested": promotion_requested})
    protocols = _assertions(checkpoint_class, merged)
    checkpoint = ConstitutionalCheckpoint(checkpoint_id, checkpoint_class, model_id, parent_checkpoint_id,
        int(time.time()), file_inventory, protocols, list(AIOPS_DOMAINS), dict(optimization or {}),
        dict(identity or {}), dict(rollback or {}), dict(capabilities or {}), decide_promotion(protocols, promotion_requested))
    checkpoint.seal(signing_key=signing_key, authorized_by=authorized_by)
    return checkpoint

def write_constitutional_manifest(root: str | Path, checkpoint: ConstitutionalCheckpoint, filename: str = "constitutional_manifest.json") -> Path:
    path = Path(root) / filename
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

def load_and_verify_constitutional_manifest(root: str | Path, filename: str = "constitutional_manifest.json",
    signing_key: Optional[str] = None, require_promoted: bool = True) -> Dict[str, Any]:
    root = Path(root)
    data = json.loads((root / filename).read_text(encoding="utf-8"))
    supplied_hash = data.get("manifest_sha256")
    supplied_hmac = data.get("authorization_hmac_sha256")
    unsigned = dict(data)
    unsigned["manifest_sha256"] = None
    unsigned["authorization_hmac_sha256"] = None
    canonical = _canonical_json(unsigned)
    if supplied_hash != _sha256_bytes(canonical):
        raise ConstitutionalGateError("constitutional manifest integrity seal mismatch")
    if data.get("promotion_status") == "promoted":
        if not signing_key or not supplied_hmac:
            raise ConstitutionalGateError("promoted checkpoint lacks verifiable authorization")
        actual_hmac = hmac.new(signing_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(str(supplied_hmac), actual_hmac):
            raise ConstitutionalGateError("constitutional authorization signature mismatch")
    validate_inventory(root, data.get("files") or {})
    failed = [item for item in data.get("protocols", []) if not item.get("passed")]
    data["verified"] = True
    data["authorized"] = bool(supplied_hmac and signing_key)
    data["failed_protocols"] = [item.get("protocol") for item in failed]
    if require_promoted:
        require_promotable(data)
    return data

def require_promotable(manifest: Mapping[str, Any]) -> None:
    if manifest.get("promotion_status") != "promoted":
        failed = manifest.get("failed_protocols") or [item.get("protocol") for item in manifest.get("protocols", []) if not item.get("passed")]
        raise ConstitutionalGateError(f"checkpoint remains quarantined; blockers={failed}")
