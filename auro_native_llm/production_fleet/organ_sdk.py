"""Repository-native SDK joining BRAIN AI, NOVA, MatDaemon, and CAPSULA."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SDKConfig:
    brain_url: str = os.getenv("BRAIN_AI_URL", "http://127.0.0.1:4943")
    nova_url: str = os.getenv("NOVA_URL", "http://127.0.0.1:8090")
    matdaemon_url: str = os.getenv("MATDAEMON_URL", "http://127.0.0.1:8000")
    capsula_url: str = os.getenv("CAPSULA_URL", "http://127.0.0.1:8784")
    timeout: float = float(os.getenv("AURO_SDK_TIMEOUT", "60"))


class HttpJSON:
    def __init__(self, base_url: str, timeout: float = 60): self.base_url = base_url.rstrip("/"); self.timeout = timeout
    def get(self, path: str) -> dict[str, Any]: return self._request("GET", path, None)
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: return self._request("POST", path, payload)
    def _request(self, method: str, path: str, payload: dict[str, Any] | None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(self.base_url + path, data=data, headers={"content-type": "application/json"}, method=method)
        with urlopen(request, timeout=self.timeout) as response: return json.loads(response.read().decode())


class BrainAI:
    def __init__(self, http: HttpJSON): self.http = http
    def health(self): return self.http.get("/health")
    def state(self): return self.http.get("/v1/brain/state")
    def query(self, path="/v1/brain/operator-snapshot"): return self.http.get(path)


class MatDaemon:
    def __init__(self, http: HttpJSON): self.http = http
    def tools(self): return self.http.get("/v1/tools")
    def call(self, name: str, arguments: dict[str, Any]):
        if not name.startswith("matdaemon_"): raise ValueError("MatDaemon tool names must start with matdaemon_")
        return self.http.post(f"/v1/tools/{name}", {"arguments": arguments})
    def rank_text(self, query: str, candidates: list[str], k=5): return self.call("matdaemon_text_similarity_top_k", {"queries": [query], "candidates": candidates, "k": k})


class Capsula:
    def __init__(self, http: HttpJSON): self.http = http
    def runtimes(self): return self.http.get("/api/runtimes")
    def create_session(self, runtime="python", name=None): return self.http.post("/api/session", {"runtime": runtime, "name": name})
    def write_file(self, session_id: str, path: str, content: str): return self.http.post(f"/api/session/{session_id}/file", {"path": path, "content": content})
    def run(self, session_id: str): return self.http.post(f"/api/session/{session_id}/run", {})
    def manifest(self, session_id: str): return self.http.post(f"/api/session/{session_id}/manifest", {})
    def deploy_plan(self, session_id: str): return self.http.post(f"/api/session/{session_id}/deploy-plan", {})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _actions_sha256(actions: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical(actions)).hexdigest()


def _approval_action_key(grant: dict[str, Any], action: dict[str, Any]) -> str:
    material = {
        "approval_id": str(grant.get("approval_id", "")),
        "nonce": str(grant.get("nonce", "")),
        "action": action,
    }
    return hashlib.sha256(_canonical(material)).hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ApprovalReplayStore:
    """Cross-process one-time action consumption using atomic marker creation.

    Each signed approval action can execute once. Markers are append-only safety
    records and may be pruned after the corresponding approval has expired.
    """

    def __init__(self, root: str | Path | None = None):
        configured = root or os.getenv("AURO_APPROVAL_REPLAY_DIR") or "./state/approval-replay"
        self.root = Path(configured)

    def consume(self, grant: dict[str, Any], action: dict[str, Any]) -> bool:
        key = _approval_action_key(grant, action)
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / f"{key}.used"
        payload = {
            "schema": "auro.approval-replay-marker.v1",
            "approval_id": str(grant.get("approval_id", "")),
            "nonce": str(grant.get("nonce", "")),
            "action_sha256": hashlib.sha256(_canonical(action)).hexdigest(),
            "expires_at_ms": int(grant.get("expires_at_ms", 0) or 0),
            "consumed_at_ms": int(time.time() * 1000),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(marker, flags, 0o600)
        except FileExistsError:
            return False
        try:
            os.write(fd, _canonical(payload) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.root)
        return True

    def prune_expired(self, now_ms: int | None = None, limit: int = 1000) -> int:
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        removed = 0
        if not self.root.exists():
            return 0
        for marker in list(self.root.glob("*.used"))[:max(0, int(limit))]:
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if int(payload.get("expires_at_ms", 0)) <= now:
                    marker.unlink(missing_ok=True)
                    removed += 1
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        if removed:
            _fsync_dir(self.root)
        return removed


class AuroOrganSDK:
    def __init__(self, config: SDKConfig | None = None, replay_store: ApprovalReplayStore | None = None):
        self.config = config or SDKConfig(); timeout = self.config.timeout
        self.brain = BrainAI(HttpJSON(self.config.brain_url, timeout)); self.nova = HttpJSON(self.config.nova_url, timeout)
        self.matdaemon = MatDaemon(HttpJSON(self.config.matdaemon_url, timeout)); self.capsula = Capsula(HttpJSON(self.config.capsula_url, timeout))
        self.replay_store = replay_store or ApprovalReplayStore()

    def manifest(self) -> dict[str, Any]:
        return {"schema": "auro.organ_sdk.v2", "approval_authority": "server", "approval_replay_protection": "one-time-per-signed-action", "organs": {
            "brain": {"purpose": "cognitive state, identity, continuity", "operations": ["state", "query"]},
            "nova": {"purpose": "governance, council, arbitration", "operations": ["respond"]},
            "matdaemon": {"purpose": "retrieval, similarity, bounded matrix compute", "operations": ["call", "rank_text"]},
            "capsula": {"purpose": "bounded build sessions", "operations": ["create_session", "write_file", "run", "manifest", "deploy_plan"]}}}

    def health(self) -> dict[str, Any]:
        checks = {"brain": lambda: self.brain.health(), "nova": lambda: self.nova.get("/health"), "matdaemon": lambda: self.matdaemon.tools(), "capsula": lambda: self.capsula.runtimes()}
        result = {}
        for name, check in checks.items():
            started = time.perf_counter()
            try: result[name] = {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "evidence": check()}
            except Exception as exc: result[name] = {"ok": False, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "error": str(exc)[:300]}
        return {"schema": "auro.organ_sdk.health.v1", "organs": result, "ready": all(item["ok"] for item in result.values())}

    def action_contract(self) -> dict[str, Any]:
        return {"matdaemon": {"tool": "matdaemon", "arguments": {"name": "matdaemon_<declared_tool>", "arguments": {}}}, "capsula": {"tool": "capsula", "arguments": {"operation": "create_session|write_file|run|manifest|deploy_plan", "parameters": {}}}}

    def verify_server_approval(self, grant: dict[str, Any], actions: list[dict[str, Any]] | None = None, now_ms: int | None = None) -> bool:
        key = os.getenv("AURO_APPROVAL_HMAC_KEY", "")
        if not key or not isinstance(grant, dict): return False
        signature = str(grant.get("signature", "")); unsigned = dict(grant); unsigned.pop("signature", None)
        if len(signature) != 64: return False
        expected = hmac.new(key.encode("utf-8"), _canonical(unsigned), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected): return False
        schema = unsigned.get("schema")
        if schema not in {"auro.server-approval.v1", "auro.server-approval.v2"}: return False
        if unsigned.get("authority") != "server" or not unsigned.get("approval_id") or not unsigned.get("subject"): return False
        if schema == "auro.server-approval.v2" and not unsigned.get("nonce"): return False
        approved_actions = unsigned.get("actions")
        if not isinstance(approved_actions, list) or not approved_actions: return False
        if not hmac.compare_digest(str(unsigned.get("actions_sha256", "")), _actions_sha256(approved_actions)): return False
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        try: not_before = int(unsigned["not_before_ms"]); expires = int(unsigned["expires_at_ms"])
        except (KeyError, TypeError, ValueError): return False
        if expires <= not_before or now < not_before or now >= expires: return False
        if actions is not None and _canonical(actions) != _canonical(approved_actions): return False
        return True

    def consume_server_approval(self, grant: dict[str, Any], action: dict[str, Any]) -> bool:
        if not self.verify_server_approval(grant): return False
        approved_actions = grant.get("actions", [])
        if _canonical(action) not in {_canonical(item) for item in approved_actions}: return False
        return self.replay_store.consume(grant, action)

    def execute(self, action: dict[str, Any], *, approval_grant: dict[str, Any] | None = None) -> dict[str, Any]:
        if not approval_grant or not self.consume_server_approval(approval_grant, action):
            raise PermissionError("valid unused server-authoritative approval required for execution")
        started = time.perf_counter(); tool = action.get("tool"); args = action.get("arguments") or {}
        if tool == "matdaemon": output = self.matdaemon.call(str(args["name"]), dict(args.get("arguments") or {}))
        elif tool == "capsula":
            operation = str(args.get("operation", "")); allowed = {"create_session", "write_file", "run", "manifest", "deploy_plan"}
            if operation not in allowed: raise ValueError(f"CAPSULA operation not allowed: {operation}")
            output = getattr(self.capsula, operation)(**dict(args.get("parameters") or {}))
        else: raise ValueError(f"Unsupported organ tool: {tool}")
        return {"tool": tool, "ok": True, "output": output, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "approval_id": approval_grant.get("approval_id")}


def build_server_approval(actions: list[dict[str, Any]], subject: str, approval_id: str, *, ttl_ms: int = 60_000, now_ms: int | None = None, key: str | None = None, nonce: str | None = None) -> dict[str, Any]:
    signing_key = key or os.getenv("AURO_APPROVAL_HMAC_KEY", "")
    if not signing_key: raise RuntimeError("AURO_APPROVAL_HMAC_KEY is required to issue server approvals")
    if not actions: raise ValueError("at least one action is required")
    if int(ttl_ms) <= 0: raise ValueError("ttl_ms must be positive")
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    token_nonce = nonce or os.urandom(16).hex()
    grant = {"schema": "auro.server-approval.v2", "approval_id": approval_id, "subject": subject, "nonce": token_nonce, "authority": "server", "actions": actions, "actions_sha256": _actions_sha256(actions), "not_before_ms": now, "expires_at_ms": now + int(ttl_ms)}
    grant["signature"] = hmac.new(signing_key.encode("utf-8"), _canonical(grant), hashlib.sha256).hexdigest()
    return grant
