"""Tamper-evident receipt chain for model, capability, and organ activity."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, os, threading, time
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class Receipt:
    sequence: int
    timestamp_ns: int
    kind: str
    subject: str
    ok: bool
    previous_hash: str
    payload_hash: str
    receipt_hash: str
    metadata: dict[str,Any]

class ReceiptLedger:
    def __init__(self,path: str|Path|None=None):
        configured=path if path is not None else os.getenv("AURO_RECEIPT_LEDGER")
        self.path=Path(configured) if configured else None
        self._lock=threading.Lock(); self._receipts:list[Receipt]=[]
        if self.path and self.path.exists(): self._load()

    def record(self,kind:str,subject:str,ok:bool,payload:Any,metadata:dict[str,Any]|None=None)->Receipt:
        with self._lock:
            sequence=len(self._receipts)+1; previous=self._receipts[-1].receipt_hash if self._receipts else "GENESIS"
            payload_hash=_hash(payload); timestamp=time.time_ns(); meta=dict(metadata or {})
            material={"sequence":sequence,"timestamp_ns":timestamp,"kind":kind,"subject":subject,"ok":bool(ok),"previous_hash":previous,"payload_hash":payload_hash,"metadata":meta}
            receipt=Receipt(**material,receipt_hash=_hash(material)); self._receipts.append(receipt)
            if self.path:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(asdict(receipt),sort_keys=True,separators=(",",":"))+"\n")
            return receipt

    def verify(self)->dict[str,Any]:
        previous="GENESIS"
        for index,r in enumerate(self._receipts,1):
            material={k:v for k,v in asdict(r).items() if k!="receipt_hash"}
            if r.sequence!=index or r.previous_hash!=previous or _hash(material)!=r.receipt_hash:
                return {"valid":False,"failed_sequence":index,"head":previous}
            previous=r.receipt_hash
        return {"valid":True,"count":len(self._receipts),"head":previous}

    def tail(self,limit=20): return [asdict(x) for x in self._receipts[-max(0,int(limit)):]]
    def _load(self):
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip(): self._receipts.append(Receipt(**json.loads(line)))
        status=self.verify()
        if not status["valid"]: raise ValueError(f"invalid receipt ledger at sequence {status['failed_sequence']}")

def _hash(value:Any)->str:
    encoded=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str).encode()
    return hashlib.sha256(encoded).hexdigest()

