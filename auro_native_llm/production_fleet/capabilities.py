"""Native capability substrate: skills and tool contracts without plugins."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import time
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
 Capability("build.create_session","Create a CAPSULA build session.","capsula","tool",_obj({"runtime":{"type":"string"},"name":{"type":["string","null"]}}),True,True),
 Capability("build.write_file","Write a file inside a CAPSULA session.","capsula","tool",_obj({"session_id":{"type":"string"},"path":{"type":"string"},"content":{"type":"string"}},("session_id","path","content")),True,True),
 Capability("build.run","Run a CAPSULA session.","capsula","tool",_obj({"session_id":{"type":"string"}},("session_id",)),True,True),
 Capability("build.manifest","Create the proof manifest for a CAPSULA session.","capsula","tool",_obj({"session_id":{"type":"string"}},("session_id",))),
 Capability("skill.research","Internal research workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("clarify claim","retrieve evidence","compare sources","red-team","synthesize with uncertainty")),
 Capability("skill.build","Internal governed build workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("specify acceptance tests","design smallest slice","create capsule","run tests","emit receipt")),
 Capability("skill.reason","Internal logic and decision workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("extract premises","test contradictions","quantify uncertainty","compare alternatives","answer")),
 Capability("skill.memory","Internal continuity workflow.","nova","skill",_obj({"objective":{"type":"string"}}),playbook=("identify relevant state","rank memories","check recency and provenance","reinject bounded context")),
)

class NativeCapabilities:
    def __init__(self,sdk,capabilities=BUILTINS): self.sdk=sdk; self._items={x.name:x for x in capabilities}
    def manifest(self): return {"schema":"auro.native_capabilities.v1","protocol":"tool-contract-compatible","capabilities":[asdict(x) for x in self._items.values()]}
    def skills_prompt(self):
        return "\n".join(f"{x.name}: {' -> '.join(x.playbook)}" for x in self._items.values() if x.mode=="skill")
    def call(self,name:str,arguments:dict[str,Any],approved=False):
        if name not in self._items: raise ValueError(f"Unknown capability: {name}")
        spec=self._items[name]; _validate(spec.input_schema,arguments)
        if spec.approval_required and not approved: return {"ok":False,"denied":True,"reason":"explicit approval required","capability":name}
        started=time.perf_counter(); output=self._dispatch(name,arguments)
        return {"ok":True,"capability":name,"organ":spec.organ,"output":output,"latency_ms":round((time.perf_counter()-started)*1000,3)}
    def _dispatch(self,name,a):
        if name=="brain.state": return self.sdk.brain.state()
        if name=="brain.operator_snapshot": return self.sdk.brain.query(a.get("path","/v1/brain/operator-snapshot"))
        if name=="memory.rank_text": return self.sdk.matdaemon.rank_text(a["query"],a["candidates"],int(a.get("k",5)))
        if name=="compute.matmul": return self.sdk.matdaemon.call("matdaemon_matmul",a)
        if name=="build.create_session": return self.sdk.capsula.create_session(a.get("runtime","python"),a.get("name"))
        if name=="build.write_file": return self.sdk.capsula.write_file(a["session_id"],a["path"],a["content"])
        if name=="build.run": return self.sdk.capsula.run(a["session_id"])
        if name=="build.manifest": return self.sdk.capsula.manifest(a["session_id"])
        if name.startswith("skill."): return {"playbook":list(self._items[name].playbook),"arguments":a}
        raise ValueError(f"No dispatcher for {name}")

def _validate(schema,args):
    if not isinstance(args,dict): raise ValueError("arguments must be an object")
    allowed=set(schema.get("properties",{})); extra=set(args)-allowed
    missing=set(schema.get("required",[]))-set(args)
    if extra: raise ValueError("unknown arguments: "+", ".join(sorted(extra)))
    if missing: raise ValueError("missing arguments: "+", ".join(sorted(missing)))

