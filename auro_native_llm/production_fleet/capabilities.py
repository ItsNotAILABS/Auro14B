"""Canonical native capability substrate for the production fleet."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    organ: str
    mode: str
    input_schema: dict[str, Any]
    mutating: bool = False
    approval_required: bool = False
    playbook: tuple[str, ...] = ()


def _obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def capability_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Canonical action signed by the server for one capability invocation."""
    return {"tool": "capability", "arguments": {"name": name, "arguments": arguments}}


BUILTINS = (
    Capability("brain.state", "Read live BRAIN AI cognitive state.", "brain", "tool", _obj({})),
    Capability("brain.operator_snapshot", "Read BRAIN AI operator snapshot.", "brain", "tool", _obj({"path": {"type": "string"}})),
    Capability("brain.fused_snapshot", "Read HIM fused brain state.", "brain", "tool", _obj({})),
    Capability("brain.neuromorphic_snapshot", "Read HIM spiking, synaptic, and CEU telemetry without granting execution authority.", "brain", "tool", _obj({})),
    Capability("brain.cycle", "Run one bounded cognitive cycle.", "brain", "tool", _obj({"observation": {"type": "string"}, "importance": {"type": "number"}, "execute_requested": {"type": "boolean"}}, ("observation",))),
    Capability("brain.migration_status", "Check optional topology parity.", "brain", "tool", _obj({})),
    Capability("memory.rank_text", "Rank memories or documents.", "matdaemon", "tool", _obj({"query": {"type": "string"}, "candidates": {"type": "array", "items": {"type": "string"}}, "k": {"type": "integer"}}, ("query", "candidates"))),
    Capability("compute.matmul", "Run bounded matrix multiplication.", "matdaemon", "tool", _obj({"a": {"type": "array"}, "b": {"type": "array"}, "backend": {"type": "string"}}, ("a", "b"))),
    Capability("compute.engines", "List configured compute planes.", "compute", "tool", _obj({})),
    Capability("cloudflare.runtime", "Describe the optional hosted compatibility plane.", "cloudflare", "tool", _obj({})),
    Capability("cloudflare.plan", "Create a non-executing Cloudflare recipe.", "cloudflare", "tool", _obj({"objective": {"type": "string"}}, ("objective",))),
    Capability("build.create_session", "Create a CAPSULA build session.", "capsula", "tool", _obj({"runtime": {"type": "string"}, "name": {"type": ["string", "null"]}}), True, True),
    Capability("build.write_file", "Write a file inside a CAPSULA session.", "capsula", "tool", _obj({"session_id": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}}, ("session_id", "path", "content")), True, True),
    Capability("build.run", "Run a CAPSULA session.", "capsula", "tool", _obj({"session_id": {"type": "string"}}, ("session_id",)), True, True),
    Capability("build.manifest", "Create a CAPSULA proof manifest.", "capsula", "tool", _obj({"session_id": {"type": "string"}}, ("session_id",))),
    Capability("wallet.balance", "Read a paper-credit balance.", "parallax", "tool", _obj({"account": {"type": "string"}, "asset": {"type": "string"}}, ("account",))),
    Capability("wallet.fund_sandbox", "Issue paper-only test credits.", "parallax", "tool", _obj({"account": {"type": "string"}, "amount": {"type": "string"}, "asset": {"type": "string"}}, ("account", "amount")), True, True),
    Capability("wallet.transfer_paper", "Transfer paper credits.", "parallax", "tool", _obj({"source": {"type": "string"}, "destination": {"type": "string"}, "amount": {"type": "string"}, "asset": {"type": "string"}, "memo": {"type": "string"}}, ("source", "destination", "amount")), True, True),
    Capability("wallet.verify_ledger", "Verify paper-wallet integrity.", "parallax", "tool", _obj({})),
    Capability("office.create_bundle", "Create MD, CSV, DOCX, XLSX, PDF and manifest outputs.", "office", "tool", _obj({"out_dir": {"type": "string"}, "title": {"type": "string"}, "sections": {"type": "array"}, "table": {"type": "array"}, "vault": {"type": "boolean"}}, ("out_dir", "title", "sections")), True, True),
    Capability("browser.task.enqueue", "Queue governed browser work.", "browser-brain", "tool", _obj({"kind": {"type": "string"}, "payload": {"type": "object"}}, ("kind", "payload")), True, True),
    Capability("browser.task.status", "Read browser task state.", "browser-brain", "tool", _obj({"task_id": {"type": "string"}}, ("task_id",))),
    Capability("browser.tasks.list", "List browser tasks.", "browser-brain", "tool", _obj({"limit": {"type": "integer"}})),
    Capability("skill.research", "Internal research workflow.", "nova", "skill", _obj({"objective": {"type": "string"}}), playbook=("retrieve evidence", "compare sources", "red-team", "synthesize")),
    Capability("skill.build", "Internal build workflow.", "nova", "skill", _obj({"objective": {"type": "string"}}), playbook=("specify acceptance tests", "build", "test", "emit receipt")),
    Capability("skill.reason", "Internal reasoning workflow.", "nova", "skill", _obj({"objective": {"type": "string"}}), playbook=("extract premises", "test contradictions", "answer")),
    Capability("skill.memory", "Internal continuity workflow.", "nova", "skill", _obj({"objective": {"type": "string"}}), playbook=("rank memories", "check provenance", "reinject bounded context")),
)


class NativeCapabilities:
    def __init__(self, sdk: Any, capabilities=BUILTINS, ledger=None):
        from .browser_gateway import BrowserTaskBroker
        from .compute import ComputeRegistry
        from .office import NativeOffice
        from .receipts import ReceiptLedger
        from .vault import IntegrityVault
        from .wallet import PaperWallet
        from auro_native_llm.brain import HIMBrain
        from auro_native_llm.cloudflare import CloudflareRuntimeContract

        self.sdk = sdk
        self._items = {item.name: item for item in capabilities}
        self.ledger = ledger or ReceiptLedger()
        self.wallet = PaperWallet(os.getenv("AURO_WALLET_LEDGER") or None)
        self.office = NativeOffice()
        self.vault = IntegrityVault(os.getenv("AURO_VAULT_ROOT", "./state/auro-vault"))
        self.browser = BrowserTaskBroker()
        self.compute = ComputeRegistry()
        self.cloudflare = CloudflareRuntimeContract()
        self.brain = HIMBrain(os.getenv("AURO_BRAIN_STATE") or "./state/him-brain.json")
        self.downloads: dict[str, Path] = {}

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "auro.native_capabilities.v3",
            "protocol": "tool-contract-compatible",
            "approval_authority": "signed-server-grant",
            "approval_replay_protection": "one-time-per-action",
            "capabilities": [asdict(x) for x in self._items.values()],
        }

    def skills_prompt(self) -> str:
        return "\n".join(f"{x.name}: {' -> '.join(x.playbook)}" for x in self._items.values() if x.mode == "skill")

    def requires_approval(self, name: str) -> bool:
        if name not in self._items:
            raise ValueError(f"Unknown capability: {name}")
        return bool(self._items[name].approval_required)

    def call(self, name: str, arguments: dict[str, Any], *, approval_grant: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._items:
            raise ValueError(f"Unknown capability: {name}")
        spec = self._items[name]
        _validate(spec.input_schema, arguments)
        approval_id = None
        if spec.approval_required:
            action = capability_action(name, arguments)
            consumer = getattr(self.sdk, "consume_server_approval", None)
            approved = bool(approval_grant and callable(consumer) and consumer(dict(approval_grant), action))
            if not approved:
                result = {
                    "ok": False,
                    "denied": True,
                    "reason": "valid unused server-authoritative approval required",
                    "capability": name,
                }
                result["receipt"] = asdict(self.ledger.record("capability", name, False, result, {"denied": True, "approval_authority": "server"}))
                return result
            approval_id = str(approval_grant.get("approval_id", ""))
        started = time.perf_counter()
        output = self._dispatch(name, arguments)
        result = {
            "ok": True,
            "capability": name,
            "organ": spec.organ,
            "output": output,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if approval_id:
            result["approval_id"] = approval_id
        result["receipt"] = asdict(self.ledger.record("capability", name, True, result, {"approval_id": approval_id} if approval_id else None))
        return result

    def resolve_download(self, digest: str) -> Path | None:
        return self.downloads.get(digest)

    def _dispatch(self, name: str, a: dict[str, Any]) -> Any:
        if name == "brain.state": return self.sdk.brain.state()
        if name == "brain.operator_snapshot": return self.sdk.brain.query(a.get("path", "/v1/brain/operator-snapshot"))
        if name == "brain.fused_snapshot": return self.brain.snapshot()
        if name == "brain.neuromorphic_snapshot":
            snapshot = self.brain.snapshot()
            return {
                "schema": "auro.brain.neuromorphic-capability.v1",
                "telemetry": snapshot.get("neuromorphic", {}),
                "last_cycle": snapshot.get("last_neuromorphic_cycle"),
                "persistence": snapshot.get("neuromorphic_persistence", {}),
                "authority": "telemetry_only",
                "can_authorize_execution": False,
            }
        if name == "brain.cycle": return asdict(self.brain.cycle(a["observation"], importance=a.get("importance", .5), execute_requested=a.get("execute_requested", False)))
        if name == "brain.migration_status": return self.brain.legacy_parity()
        if name == "memory.rank_text": return self.sdk.matdaemon.rank_text(a["query"], a["candidates"], int(a.get("k", 5)))
        if name == "compute.matmul": return self.sdk.matdaemon.call("matdaemon_matmul", a)
        if name == "compute.engines": return self.compute.manifest()
        if name == "cloudflare.runtime": return self.cloudflare.manifest()
        if name == "cloudflare.plan": return self.cloudflare.recipe(a["objective"])
        if name == "build.create_session": return self.sdk.capsula.create_session(a.get("runtime", "python"), a.get("name"))
        if name == "build.write_file": return self.sdk.capsula.write_file(a["session_id"], a["path"], a["content"])
        if name == "build.run": return self.sdk.capsula.run(a["session_id"])
        if name == "build.manifest": return self.sdk.capsula.manifest(a["session_id"])
        if name == "wallet.balance": return {"account": a["account"], "asset": a.get("asset", "PXCRED"), "balance": self.wallet.balance(a["account"], a.get("asset", "PXCRED")), "mode": "paper"}
        if name == "wallet.fund_sandbox": return self.wallet.fund(a["account"], a["amount"], a.get("asset", "PXCRED"))
        if name == "wallet.transfer_paper": return self.wallet.transfer(a["source"], a["destination"], a["amount"], a.get("asset", "PXCRED"), a.get("memo", "paper transfer"))
        if name == "wallet.verify_ledger": return self.wallet.verify()
        if name == "office.create_bundle":
            bundle = self.office.create_bundle(a["out_dir"], a["title"], a["sections"], a.get("table") or [])
            if a.get("vault"):
                bundle["vault_records"] = [self.vault.put(x["name"], Path(x["path"]).read_bytes()) for x in bundle["files"]]
            return bundle
        if name == "browser.task.enqueue": return self.browser.enqueue(a["kind"], a["payload"])
        if name == "browser.task.status": return self.browser.get(a["task_id"])
        if name == "browser.tasks.list": return {"tasks": self.browser.list(int(a.get("limit", 50)))}
        if name.startswith("skill."): return {"playbook": list(self._items[name].playbook), "arguments": a}
        raise ValueError(f"No dispatcher for {name}")


def _validate(schema: dict[str, Any], args: Any) -> None:
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    allowed = set(schema.get("properties", {}))
    extra = set(args) - allowed
    missing = set(schema.get("required", [])) - set(args)
    if extra:
        raise ValueError("unknown arguments: " + ", ".join(sorted(extra)))
    if missing:
        raise ValueError("missing arguments: " + ", ".join(sorted(missing)))
