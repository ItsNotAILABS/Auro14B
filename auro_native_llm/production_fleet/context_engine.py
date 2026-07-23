"""Persistent Python context virtualization for HIM.

The logical context store can contain millions of tokens. Each model call gets
only a ranked, provenance-preserving working set under an explicit token budget.
This is retrieval and context engineering, not a false transformer-window claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Iterable
from functools import wraps

CHARS_PER_TOKEN = 4
TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

def synchronized(function):
    @wraps(function)
    def locked(self,*args,**kwargs):
        with self._lock:return function(self,*args,**kwargs)
    return locked


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class ContextHit:
    chunk_id: str
    document_id: str
    source: str
    text: str
    tokens: int
    score: float
    importance: float
    created_at: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ContextPack:
    query: str
    context: str
    hits: tuple[ContextHit, ...]
    logical_tokens: int
    injected_tokens: int
    token_budget: int
    receipt_hash: str

    def public(self) -> dict[str, Any]:
        value=asdict(self)
        value["hits"]=[{k:v for k,v in asdict(x).items() if k!="text"} for x in self.hits]
        return value


class ContextEngine:
    schema="him.context.virtualization.v1"

    def __init__(self,path: str|Path="state/him-context.sqlite",default_budget:int=32_000):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.default_budget=max(512,int(default_budget))
        self._lock=threading.RLock()
        self.db=sqlite3.connect(self.path,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self._schema()

    def _schema(self):
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA busy_timeout=30000;
        CREATE TABLE IF NOT EXISTS documents(
          id TEXT PRIMARY KEY,source TEXT NOT NULL,content_sha256 TEXT NOT NULL,
          tokens INTEGER NOT NULL,importance REAL NOT NULL,created_at REAL NOT NULL,
          metadata_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(
          id TEXT PRIMARY KEY,document_id TEXT NOT NULL,ordinal INTEGER NOT NULL,
          text TEXT NOT NULL,tokens INTEGER NOT NULL,importance REAL NOT NULL,
          created_at REAL NOT NULL,metadata_json TEXT NOT NULL,
          FOREIGN KEY(document_id) REFERENCES documents(id));
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED,text,tokenize='unicode61');
        CREATE TABLE IF NOT EXISTS context_receipts(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,payload_json TEXT NOT NULL,
          previous_hash TEXT NOT NULL,hash TEXT NOT NULL,created_at REAL NOT NULL);
        """);self.db.commit()

    @synchronized
    def ingest(self,text:str,*,source:str="input",kind:str="document",importance:float=.5,
               metadata:dict[str,Any]|None=None,chunk_tokens:int=900,overlap_tokens:int=120,
               allow_sensitive:bool=False)->dict[str,Any]:
        text=str(text or "")
        if not text.strip(): raise ValueError("context text must not be empty")
        if not allow_sensitive and any(p.search(text) for p in SECRET_PATTERNS):
            raise ValueError("possible secret detected; context was not stored")
        importance=max(0.0,min(1.0,float(importance)));now=time.time()
        digest=hashlib.sha256(text.encode()).hexdigest()
        doc_id="doc-"+digest[:20]
        existing=self.db.execute("SELECT id FROM documents WHERE id=?",(doc_id,)).fetchone()
        if existing:return {"ok":True,"deduplicated":True,"document_id":doc_id,"content_sha256":digest}
        meta={"kind":kind,**(metadata or {})}
        pieces=list(self._chunks(text,chunk_tokens,overlap_tokens))
        self.db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?)",
            (doc_id,source,digest,estimate_tokens(text),importance,now,json.dumps(meta,sort_keys=True)))
        for ordinal,piece in enumerate(pieces):
            cid=f"{doc_id}-c{ordinal:06d}"
            self.db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                (cid,doc_id,ordinal,piece,estimate_tokens(piece),importance,now,json.dumps(meta,sort_keys=True)))
            self.db.execute("INSERT INTO chunks_fts(chunk_id,text) VALUES(?,?)",(cid,piece))
        self.db.commit()
        receipt=self._receipt("ingest",{"document_id":doc_id,"source":source,"tokens":estimate_tokens(text),
                                        "chunks":len(pieces),"content_sha256":digest})
        return {"ok":True,"deduplicated":False,"document_id":doc_id,"content_sha256":digest,
                "tokens":estimate_tokens(text),"chunks":len(pieces),"receipt":receipt}

    @synchronized
    def retrieve(self,query:str,*,token_budget:int|None=None,top_k:int=24)->ContextPack:
        query=str(query or "").strip();budget=max(256,int(token_budget or self.default_budget))
        terms=list(dict.fromkeys(x.lower() for x in TOKEN_RE.findall(query)))[:32]
        rows=[]
        if terms:
            expression=" OR ".join(f'"{x}"' for x in terms)
            rows=self.db.execute("""
              SELECT c.*,d.source,bm25(chunks_fts) AS lexical
              FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.chunk_id
              JOIN documents d ON d.id=c.document_id
              WHERE chunks_fts MATCH ? ORDER BY lexical LIMIT ?
            """,(expression,max(top_k*4,40))).fetchall()
        now=time.time();ranked=[]
        for row in rows:
            age_days=max(0,(now-float(row["created_at"]))/86400)
            recency=1/(1+age_days/30)
            lexical=1/(1+abs(float(row["lexical"])))
            score=.65*lexical+.25*float(row["importance"])+.10*recency
            ranked.append((score,row))
        ranked.sort(key=lambda x:(-x[0],-float(x[1]["created_at"])))
        hits=[];parts=[];used=0
        for score,row in ranked:
            tokens=int(row["tokens"])
            if used+tokens>budget:continue
            meta=json.loads(row["metadata_json"])
            hit=ContextHit(row["id"],row["document_id"],row["source"],row["text"],tokens,
                           round(score,6),float(row["importance"]),float(row["created_at"]),meta)
            hits.append(hit);used+=tokens
            parts.append(f'[SOURCE id={hit.chunk_id} origin="{hit.source}" score={hit.score}]\n{hit.text}\n[/SOURCE]')
            if len(hits)>=top_k:break
        logical=self.stats()["logical_tokens"]
        header=f"[HIM_CONTEXT logical_tokens={logical} injected_tokens={used} budget={budget} sources={len(hits)}]"
        context=header+"\n"+"\n\n".join(parts)+"\n[/HIM_CONTEXT]"
        payload={"query_sha256":hashlib.sha256(query.encode()).hexdigest(),"hit_ids":[x.chunk_id for x in hits],
                 "logical_tokens":logical,"injected_tokens":used,"token_budget":budget}
        receipt=self._receipt("retrieve",payload)["hash"]
        return ContextPack(query,context,tuple(hits),logical,used,budget,receipt)

    @synchronized
    def stats(self)->dict[str,Any]:
        row=self.db.execute("SELECT COUNT(*) documents,COALESCE(SUM(tokens),0) tokens FROM documents").fetchone()
        chunks=self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        head=self.db.execute("SELECT hash FROM context_receipts ORDER BY sequence DESC LIMIT 1").fetchone()
        return {"schema":self.schema,"documents":int(row["documents"]),"chunks":int(chunks),
                "logical_tokens":int(row["tokens"]),"default_injection_budget":self.default_budget,
                "receipt_head":head["hash"] if head else "0"*64,
                "window_claim":"virtualized retrieval; not single-pass transformer attention"}

    def _chunks(self,text:str,chunk_tokens:int,overlap_tokens:int)->Iterable[str]:
        size=max(256,int(chunk_tokens))*CHARS_PER_TOKEN
        overlap=min(max(0,int(overlap_tokens))*CHARS_PER_TOKEN,size//2)
        start=0
        while start<len(text):
            end=min(len(text),start+size)
            if end<len(text):
                boundary=max(text.rfind("\n",start+size//2,end),text.rfind(". ",start+size//2,end))
                if boundary>start:end=boundary+1
            piece=text[start:end].strip()
            if piece:yield piece
            if end>=len(text):break
            start=max(start+1,end-overlap)

    @synchronized
    def _receipt(self,kind:str,payload:dict[str,Any])->dict[str,Any]:
        prior=self.db.execute("SELECT hash FROM context_receipts ORDER BY sequence DESC LIMIT 1").fetchone()
        previous=prior["hash"] if prior else "0"*64;created=time.time()
        canonical=json.dumps({"kind":kind,"payload":payload,"previous":previous,"created_at":created},
                             sort_keys=True,separators=(",",":"))
        digest=hashlib.sha256(canonical.encode()).hexdigest()
        cur=self.db.execute("INSERT INTO context_receipts(kind,payload_json,previous_hash,hash,created_at) VALUES(?,?,?,?,?)",
                            (kind,json.dumps(payload,sort_keys=True),previous,digest,created));self.db.commit()
        return {"sequence":cur.lastrowid,"kind":kind,"previous_hash":previous,"hash":digest,"created_at":created}

    @synchronized
    def close(self):self.db.close()
