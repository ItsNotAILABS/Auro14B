"""PARALLAX-aligned sandbox wallet: paper credits and double-entry only."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib, json, threading, time, uuid
from pathlib import Path

@dataclass(frozen=True)
class Posting: account:str; asset:str; amount:str
@dataclass(frozen=True)
class Transaction: id:str; timestamp_ns:int; memo:str; postings:list[Posting]; hash:str

class PaperWallet:
    SYSTEM_EQUITY="SYSTEM:EQUITY"
    def __init__(self,path=None):
        self.path=Path(path) if path else None; self._lock=threading.Lock(); self.transactions=[]
        if self.path and self.path.exists():
            self.transactions=[Transaction(**{**x,"postings":[Posting(**p) for p in x["postings"]]}) for x in map(json.loads,self.path.read_text().splitlines()) if x]
            self.verify()
    def balance(self,account,asset="PXCRED"):
        return str(sum((Decimal(p.amount) for t in self.transactions for p in t.postings if p.account==account and p.asset==asset),Decimal("0")))
    def fund(self,account,amount,asset="PXCRED",memo="sandbox funding"):
        value=_amount(amount); return self._post(memo,[Posting(account,asset,str(value)),Posting(self.SYSTEM_EQUITY,asset,str(-value))])
    def transfer(self,source,destination,amount,asset="PXCRED",memo="paper transfer"):
        if source==destination: raise ValueError("source and destination must differ")
        value=_amount(amount)
        if Decimal(self.balance(source,asset))<value: raise ValueError("insufficient paper balance")
        return self._post(memo,[Posting(source,asset,str(-value)),Posting(destination,asset,str(value))])
    def verify(self):
        for t in self.transactions:
            totals={}
            for p in t.postings: totals[p.asset]=totals.get(p.asset,Decimal("0"))+Decimal(p.amount)
            if any(v!=0 for v in totals.values()): raise ValueError(f"unbalanced transaction {t.id}")
            raw={"id":t.id,"timestamp_ns":t.timestamp_ns,"memo":t.memo,"postings":[asdict(p) for p in t.postings]}
            if _hash(raw)!=t.hash: raise ValueError(f"invalid transaction hash {t.id}")
        return {"valid":True,"transactions":len(self.transactions),"mode":"paper","live_custody":False}
    def _post(self,memo,postings):
        with self._lock:
            raw={"id":str(uuid.uuid4()),"timestamp_ns":time.time_ns(),"memo":str(memo),"postings":[asdict(p) for p in postings]}
            t=Transaction(id=raw["id"],timestamp_ns=raw["timestamp_ns"],memo=raw["memo"],postings=postings,hash=_hash(raw)); self.transactions.append(t)
            if self.path:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(asdict(t),sort_keys=True)+"\n")
            return asdict(t)
def _amount(value):
    try: d=Decimal(str(value))
    except InvalidOperation: raise ValueError("invalid decimal amount")
    if not d.is_finite() or d<=0: raise ValueError("amount must be positive and finite")
    return d.quantize(Decimal("0.00000001"))
def _hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
