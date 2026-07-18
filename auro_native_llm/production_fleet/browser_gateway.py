"""Persistent task broker between HIM's Python runtime and Chrome brain."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any
import hashlib, json, os, time, uuid

ALLOWED_BROWSER_TASKS={"think","recall","research","graph","document","workflow","training-export","mesh-status"}

@dataclass
class BrowserTask:
    id:str; kind:str; payload:dict[str,Any]; status:str; created_at:float
    updated_at:float; result:Any=None; error:str|None=None; receipt_hash:str|None=None

class BrowserTaskBroker:
    def __init__(self,path:str|Path|None=None):
        self.path=Path(path or os.getenv("AURO_BROWSER_TASKS","state/browser-tasks.json"));self._lock=RLock();self._tasks={};self._load()
    def enqueue(self,kind:str,payload:dict[str,Any]):
        if kind not in ALLOWED_BROWSER_TASKS: raise ValueError(f"unsupported browser task: {kind}")
        if not isinstance(payload,dict): raise ValueError("browser task payload must be an object")
        now=time.time();task=BrowserTask("bt_"+uuid.uuid4().hex,kind,payload,"queued",now,now);self._tasks[task.id]=task;self._save();return asdict(task)
    def claim(self,worker_id:str):
        with self._lock:
            queued=sorted((x for x in self._tasks.values() if x.status=="queued"),key=lambda x:x.created_at)
            if not queued:return None
            task=queued[0];task.status="claimed";task.updated_at=time.time();task.payload={**task.payload,"_worker_id":worker_id};self._save();return asdict(task)
    def complete(self,task_id:str,result:Any=None,error:str|None=None):
        with self._lock:
            if task_id not in self._tasks:raise ValueError("browser task not found")
            task=self._tasks[task_id];task.status="failed" if error else "completed";task.result=result;task.error=error;task.updated_at=time.time()
            core=json.dumps({"id":task.id,"kind":task.kind,"status":task.status,"result":result,"error":error},sort_keys=True,ensure_ascii=False,separators=(",",":"));task.receipt_hash=hashlib.sha256(core.encode()).hexdigest();self._save();return asdict(task)
    def get(self,task_id:str):return asdict(self._tasks[task_id]) if task_id in self._tasks else None
    def list(self,limit=50):return [asdict(x) for x in sorted(self._tasks.values(),key=lambda x:x.created_at,reverse=True)[:limit]]
    def _load(self):
        if self.path.exists():
            try:self._tasks={x["id"]:BrowserTask(**x) for x in json.loads(self.path.read_text())}
            except Exception:self._tasks={}
    def _save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);tmp=self.path.with_suffix(".tmp");tmp.write_text(json.dumps([asdict(x) for x in self._tasks.values()],indent=2),encoding="utf-8");tmp.replace(self.path)

