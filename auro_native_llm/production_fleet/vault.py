"""Local content-addressed integrity vault. Not a secret-key custody system."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path

class IntegrityVault:
    def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def put(self,name,data:bytes,media_type="application/octet-stream"):
        digest=hashlib.sha256(data).hexdigest(); target=self.root/digest[:2]/digest
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        try: os.chmod(target,0o600)
        except OSError: pass
        record={"name":Path(name).name,"sha256":digest,"size":len(data),"media_type":media_type,"stored_at_ns":time.time_ns(),"path":str(target),"encrypted":False,"secret_custody":False}
        (target.with_suffix(".json")).write_text(json.dumps(record,indent=2),encoding="utf-8"); return record
    def verify(self,record):
        path=Path(record["path"]); actual=hashlib.sha256(path.read_bytes()).hexdigest()
        return {"valid":actual==record["sha256"],"sha256":actual,"secret_custody":False}

