"""Repository-native SDK joining BRAIN AI, NOVA, MatDaemon, and CAPSULA."""
from __future__ import annotations
from dataclasses import dataclass
import json, os, time
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
    def __init__(self,base_url:str,timeout:float=60): self.base_url=base_url.rstrip("/"); self.timeout=timeout
    def get(self,path:str)->dict[str,Any]: return self._request("GET",path,None)
    def post(self,path:str,payload:dict[str,Any])->dict[str,Any]: return self._request("POST",path,payload)
    def _request(self,method,path,payload):
        data=None if payload is None else json.dumps(payload).encode()
        req=Request(self.base_url+path,data=data,headers={"content-type":"application/json"},method=method)
        with urlopen(req,timeout=self.timeout) as r: return json.loads(r.read().decode())

class BrainAI:
    """Client for the BRAIN AI HTTP/ICP bridge; paths remain explicit."""
    def __init__(self,http:HttpJSON): self.http=http
    def health(self): return self.http.get("/health")
    def state(self): return self.http.get("/v1/brain/state")
    def query(self,path="/v1/brain/operator-snapshot"): return self.http.get(path)

class MatDaemon:
    def __init__(self,http:HttpJSON): self.http=http
    def tools(self): return self.http.get("/v1/tools")
    def call(self,name:str,arguments:dict[str,Any]):
        if not name.startswith("matdaemon_"): raise ValueError("MatDaemon tool names must start with matdaemon_")
        return self.http.post(f"/v1/tools/{name}",{"arguments":arguments})
    def rank_text(self,query:str,candidates:list[str],k=5):
        return self.call("matdaemon_text_similarity_top_k",{"queries":[query],"candidates":candidates,"k":k})

class Capsula:
    def __init__(self,http:HttpJSON): self.http=http
    def runtimes(self): return self.http.get("/api/runtimes")
    def create_session(self,runtime="python",name=None):
        return self.http.post("/api/session",{"runtime":runtime,"name":name})
    def write_file(self,session_id:str,path:str,content:str):
        return self.http.post(f"/api/session/{session_id}/file",{"path":path,"content":content})
    def run(self,session_id:str): return self.http.post(f"/api/session/{session_id}/run",{})
    def manifest(self,session_id:str): return self.http.post(f"/api/session/{session_id}/manifest",{})
    def deploy_plan(self,session_id:str): return self.http.post(f"/api/session/{session_id}/deploy-plan",{})

class AuroOrganSDK:
    def __init__(self,config:SDKConfig|None=None):
        self.config=config or SDKConfig(); t=self.config.timeout
        self.brain=BrainAI(HttpJSON(self.config.brain_url,t))
        self.nova=HttpJSON(self.config.nova_url,t)
        self.matdaemon=MatDaemon(HttpJSON(self.config.matdaemon_url,t))
        self.capsula=Capsula(HttpJSON(self.config.capsula_url,t))

    def manifest(self)->dict[str,Any]:
        return {"schema":"auro.organ_sdk.v1","organs":{
          "brain":{"purpose":"cognitive state, identity, continuity","operations":["state","query"]},
          "nova":{"purpose":"governance, council, arbitration","operations":["respond"]},
          "matdaemon":{"purpose":"retrieval, similarity, bounded matrix compute","operations":["call","rank_text"]},
          "capsula":{"purpose":"bounded build sessions","operations":["create_session","write_file","run","manifest","deploy_plan"]}}}

    def health(self)->dict[str,Any]:
        checks={"brain":lambda:self.brain.health(),"nova":lambda:self.nova.get("/health"),
                "matdaemon":lambda:self.matdaemon.tools(),"capsula":lambda:self.capsula.runtimes()}
        result={}
        for name,check in checks.items():
            started=time.perf_counter()
            try:
                payload=check(); result[name]={"ok":True,"latency_ms":round((time.perf_counter()-started)*1000,3),"evidence":payload}
            except Exception as exc:
                result[name]={"ok":False,"latency_ms":round((time.perf_counter()-started)*1000,3),"error":str(exc)[:300]}
        return {"schema":"auro.organ_sdk.health.v1","organs":result,"ready":all(x["ok"] for x in result.values())}

    def action_contract(self)->dict[str,Any]:
        return {"matdaemon":{"tool":"matdaemon","arguments":{"name":"matdaemon_<declared_tool>","arguments":{}}},
                "capsula":{"tool":"capsula","arguments":{"operation":"create_session|write_file|run|manifest|deploy_plan","parameters":{}}}}

    def execute(self,action:dict[str,Any])->dict[str,Any]:
        started=time.perf_counter(); tool=action.get("tool"); args=action.get("arguments") or {}
        if tool=="matdaemon":
            output=self.matdaemon.call(str(args["name"]),dict(args.get("arguments") or {}))
        elif tool=="capsula":
            op=str(args.get("operation",""))
            allowed={"create_session","write_file","run","manifest","deploy_plan"}
            if op not in allowed: raise ValueError(f"CAPSULA operation not allowed: {op}")
            output=getattr(self.capsula,op)(**dict(args.get("parameters") or {}))
        else: raise ValueError(f"Unsupported organ tool: {tool}")
        return {"tool":tool,"ok":True,"output":output,"latency_ms":round((time.perf_counter()-started)*1000,3)}
