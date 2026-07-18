"""Native capability substrate: skills and tool contracts without plugins."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import os, time
from typing import Any, Callable

@dataclass(frozen=True)
class Capability:
    name: str
    description: str
    organ: str
    mode: str
    input_schema: dict[str,Any]
    mutating: bool = False
    approval_required: bool = False
    playbook: tuple[str,...] = ()

def _obj(properties,required=()):
    return {"type":"object","properties":properties,"required":list(required),"additionalProperties":False}

BUILTINS=(
 Capability("brain.state","Read live BRAIN AI cognitive state.","brain","tool",_obj({})),
 Capability("brain.operator_snapshot","Read BRAIN AI operator/ADRE snapshot.","brain","tool",_obj({"path":{"type":"string"}})),
 Capability("memory.rank_text","Rank memories or documents with MatDaemon.","matdaemon","tool",_obj({"query":{"type":"string"},"candidates":{"type":"array","items":{"type":"string"}},"k":{"type":"integer"}},("query","candidates"))),
 Capability("compute.matmul","Run bounded matrix multiplication with MatDaemon.","matdaemon","tool",_obj({"a":{"type":"array"},"b":{"type":"array"},"backend":{"type":"string"}},("a","b"))),
 Capability("compute.engines","List embedded, local, and explicitly configured cloud compute planes.","compute","tool",_obj({})),
 Capability("build.create_session","Create a CAPSULA build session.","capsula","tool",_obj({"runtime":{"type":"string"},"name":{"type":["string","null"]}}),True,True),
 Capability("build.write_file","Write a file inside a CAPSULA session.","capsula","tool",_obj({"session_id":{"type":"string"},"path":{"type":"string"},"content":{"type":"string"}},("session_id","path","content")),True,True),
 Capability("build.run","Run a CAPSULA session.","capsula","tool",_obj({"session_id":{"type":"string"}},("session_id",)),True,True),
 Capability("build.manifest","Create the proof manifest for a CAPSULA session.","capsula","tool",_obj({"session_id":{"type":"string"}},("session_id",))),
 Capability("skill.research","Internal research workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("clarify claim","retrieve evidence","compare sources","red-team","synthesize with uncertainty")),
 Capability("skill.build","Internal governed build workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("specify acceptance tests","design smallest slice","create capsule","run tests","emit receipt")),
 Capability("skill.reason","Internal logic and decision workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("extract premises","test contradictions","quantify uncertainty","compare alternatives","answer")),
 Capability("skill.memory","Internal continuity workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("identify relevant state","rank memories","check recency and provenance","reinject bounded context")),
 Capability("wallet.balance","Read an Auro paper-credit balance.","parallax","tool",_obj({"account":{"type":"string"},"asset":{"type":"string"}},("account",))),
 Capability("wallet.fund_sandbox","Issue paper-only test credits with balanced postings.","parallax","tool",_obj({"account":{"type":"string"},"amount":{"type":"string"},"asset":{"type":"string"}},("account","amount")),True,True),
 Capability("wallet.transfer_paper","Transfer paper credits between internal accounts.","parallax","tool",_obj({"source":{"type":"string"},"destination":{"type":"string"},"amount":{"type":"string"},"asset":{"type":"string"},"memo":{"type":"string"}},("source","destination","amount")),True,True),
 Capability("wallet.verify_ledger","Verify all double-entry postings and transaction hashes.","parallax","tool",_obj({})),
 Capability("office.create_bundle","Create MD, CSV, DOCX, XLSX, PDF, and hash manifest deliverables.","office","tool",_obj({"out_dir":{"type":"string"},"title":{"type":"string"},"sections":{"type":"array"},"table":{"type":"array"},"vault":{"type":"boolean"}},("out_dir","title","sections")),True,True),
 Capability("storage.put_text","Store text in the content-addressed integrity vault.","vault","tool",_obj({"name":{"type":"string"},"text":{"type":"string"},"media_type":{"type":"string"}},("name","text")),True,True),
 Capability("storage.verify","Verify a content-addressed vault record.","vault","tool",_obj({"record":{"type":"object"}},("record",))),
 Capability("security.scan_workspace","Run a bounded static secret and risky-file scan.","security","tool",_obj({"root":{"type":"string"},"max_files":{"type":"integer"}},("root",))),
 Capability("extension.package","Build a content-addressed Manifest V3 extension ZIP.","browser","tool",_obj({"out_dir":{"type":"string"},"name":{"type":"string"},"api_base":{"type":"string"}},("out_dir",)),True,True),
)

class NativeCapabilities:
    def __init__(self,sdk,capabilities=BUILTINS,ledger=None):
        from .receipts import ReceiptLedger
        from .wallet import PaperWallet
        from .office import NativeOffice
        from .vault import IntegrityVault
        self.sdk=sdk; self._items={x.name:x for x in capabilities}; self.ledger=ledger or ReceiptLedger()
        self.wallet=PaperWallet(os.getenv("AURO_WALLET_LEDGER") or None); self.office=NativeOffice()
        self.vault=IntegrityVault(os.getenv("AURO_VAULT_ROOT","./state/auro-vault"))
        from .compute import ComputeRegistry
        self.compute=ComputeRegistry()
        self.downloads={}
    def manifest(self): return {"schema":"auro.native_capabilities.v1","protocol":"tool-contract-compatible","capabilities":[asdict(x) for x in self._items.values()]}
    def skills_prompt(self):
        return "\n".join(f"{x.name}: {' -> '.join(x.playbook)}" for x in self._items.values() if x.mode=="skill")
    def call(self,name:str,arguments:dict[str,Any],approved=False):
        if name not in self._items: raise ValueError(f"Unknown capability: {name}")
        spec=self._items[name]; _validate(spec.input_schema,arguments)
        if spec.approval_required and not approved:
            result={"ok":False,"denied":True,"reason":"explicit approval required","capability":name}
            result["receipt"]=asdict(self.ledger.record("capability",name,False,result,{"denied":True})); return result
        started=time.perf_counter(); output=self._dispatch(name,arguments)
        result={"ok":True,"capability":name,"organ":spec.organ,"output":output,"latency_ms":round((time.perf_counter()-started)*1000,3)}
        result["receipt"]=asdict(self.ledger.record("capability",name,True,result)); return result
    def _dispatch(self,name,a):
        if name=="brain.state": return self.sdk.brain.state()
        if name=="brain.operator_snapshot": return self.sdk.brain.query(a.get("path","/v1/brain/operator-snapshot"))
        if name=="memory.rank_text": return self.sdk.matdaemon.rank_text(a["query"],a["candidates"],int(a.get("k",5)))
        if name=="compute.matmul": return self.sdk.matdaemon.call("matdaemon_matmul",a)
        if name=="compute.engines": return self.compute.manifest()
        if name=="build.create_session": return self.sdk.capsula.create_session(a.get("runtime","python"),a.get("name"))
        if name=="build.write_file": return self.sdk.capsula.write_file(a["session_id"],a["path"],a["content"])
        if name=="build.run": return self.sdk.capsula.run(a["session_id"])
        if name=="build.manifest": return self.sdk.capsula.manifest(a["session_id"])
        if name=="wallet.balance": return {"account":a["account"],"asset":a.get("asset","PXCRED"),"balance":self.wallet.balance(a["account"],a.get("asset","PXCRED")),"mode":"paper"}
        if name=="wallet.fund_sandbox": return self.wallet.fund(a["account"],a["amount"],a.get("asset","PXCRED"))
        if name=="wallet.transfer_paper": return self.wallet.transfer(a["source"],a["destination"],a["amount"],a.get("asset","PXCRED"),a.get("memo","paper transfer"))
        if name=="wallet.verify_ledger": return self.wallet.verify()
        if name=="office.create_bundle":
            bundle=self.office.create_bundle(a["out_dir"],a["title"],a["sections"],a.get("table") or [])
            if a.get("vault"):
                from pathlib import Path
                bundle["vault_records"]=[self.vault.put(x["name"],Path(x["path"]).read_bytes()) for x in bundle["files"]]
            return bundle
        if name=="storage.put_text": return self.vault.put(a["name"],a["text"].encode("utf-8"),a.get("media_type","text/plain; charset=utf-8"))
        if name=="storage.verify": return self.vault.verify(a["record"])
        if name=="security.scan_workspace":
            from .security import scan_workspace
            return scan_workspace(a["root"],max_files=int(a.get("max_files",2000)))
        if name=="extension.package":
            from .extensions import package_extension
            result=package_extension(a["out_dir"],a.get("name","Auro Sovereign Workspace"),a.get("api_base","http://127.0.0.1:8090"))
            self.downloads[result["sha256"]]=result["path"]
            result["download_url"]="/v1/downloads/"+result["sha256"]+".zip"
            return result
        if name.startswith("skill."): return {"playbook":list(self._items[name].playbook),"arguments":a}
        raise ValueError(f"No dispatcher for {name}")
    def resolve_download(self,digest):
        from pathlib import Path
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest): return None
        path=self.downloads.get(digest)
        return Path(path) if path and Path(path).is_file() else None

def _validate(schema,args):
    if not isinstance(args,dict): raise ValueError("arguments must be an object")
    allowed=set(schema.get("properties",{})); extra=set(args)-allowed
    missing=set(schema.get("required",[]))-set(args)
    if extra: raise ValueError("unknown arguments: "+", ".join(sorted(extra)))
    if missing: raise ValueError("missing arguments: "+", ".join(sorted(missing)))
