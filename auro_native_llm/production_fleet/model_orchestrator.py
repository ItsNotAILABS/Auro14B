"""Verified multi-model routing for HIM/NOVA.

A model lane is an actual generator plus identity metadata. Agents do not count
as models and parameter totals are never summed unless weights are truly loaded.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib, json, time
from typing import Any, Callable, Iterable

Generator = Callable[[list[dict[str,str]],dict[str,Any]],dict[str,Any]]

@dataclass(frozen=True)
class ModelLane:
    id: str
    model: str
    role: str
    provider: str
    generator: Generator = field(repr=False,compare=False)
    parameter_count: int|None = None
    capabilities: tuple[str,...] = ("general",)
    priority: int = 100
    local: bool = False
    enabled: bool = True
    checkpoint_hash: str|None = None
    def public(self):
        value=asdict(self);value.pop("generator",None)
        value["parameter_count_verified"]=self.parameter_count is not None
        value["identity_verified"]=bool(self.checkpoint_hash or (self.model and self.provider))
        return value

@dataclass(frozen=True)
class RouteDecision:
    task: str
    strategy: str
    selected_lane: str
    candidates: tuple[str,...]
    reason: str

class MultiModelOrchestrator:
    """Task router with observable attempts, bounded failover and no identity blur."""
    def __init__(self,lanes:Iterable[ModelLane],allow_hosted_fallback:bool=False):
        self.lanes={x.id:x for x in lanes if x.enabled}
        if not self.lanes: raise ValueError("at least one enabled model lane is required")
        self.allow_hosted_fallback=allow_hosted_fallback
        self.traces:list[dict[str,Any]]=[]

    def manifest(self):
        return {"schema":"him.model_fleet.v1","model_count":len(self.lanes),
                "allow_hosted_fallback":self.allow_hosted_fallback,
                "parameter_accounting":"per-lane weights; never agents or token counts",
                "models":[x.public() for x in self.lanes.values()]}

    def classify(self,text:str)->str:
        lower=text.lower()
        scores={
          "code":sum(x in lower for x in ("code","python","typescript","function","debug","test","contract","solidity")),
          "math":sum(x in lower for x in ("calculate","equation","proof","risk","monte carlo","optimize","math")),
          "research":sum(x in lower for x in ("research","source","evidence","compare","investigate","latest")),
          "tool":sum(x in lower for x in ("deploy","execute","run","build","manage","worker","agent")),
        }
        best=max(scores,key=scores.get)
        return best if scores[best] else "general"

    def route(self,messages:list[dict[str,str]],strategy:str="single")->RouteDecision:
        text="\n".join(str(x.get("content","")) for x in messages);task=self.classify(text)
        ranked=sorted(self.lanes.values(),key=lambda x:(task not in x.capabilities and "general" not in x.capabilities,
                    not x.local,x.priority,x.id))
        selected=ranked[0]
        return RouteDecision(task,strategy,selected.id,tuple(x.id for x in ranked),
            f"selected {selected.id}: capability={task}; local lanes preferred; priority breaks ties")

    def __call__(self,messages:list[dict[str,str]],options:dict[str,Any])->dict[str,Any]:
        strategy=str(options.pop("auro_strategy","single"));decision=self.route(messages,strategy)
        ordered=[self.lanes[x] for x in decision.candidates];attempts=[];started=time.perf_counter()
        last:Exception|None=None
        for lane in ordered:
            if lane.id!=decision.selected_lane and lane.provider!="repository-native-open-weights" and not self.allow_hosted_fallback:
                continue
            t0=time.perf_counter()
            try:
                output=lane.generator(messages,dict(options))
                trace={"lane_id":lane.id,"model":lane.model,"provider":lane.provider,"task":decision.task,
                       "ok":True,"latency_ms":round((time.perf_counter()-t0)*1000,3)}
                attempts.append(trace);output=dict(output);output["routed_model"]=lane.public();output["route"]=asdict(decision)
                self._record(decision,attempts,started);return output
            except Exception as exc:
                last=exc;attempts.append({"lane_id":lane.id,"model":lane.model,"provider":lane.provider,
                    "task":decision.task,"ok":False,"error":type(exc).__name__,"latency_ms":round((time.perf_counter()-t0)*1000,3)})
        self._record(decision,attempts,started)
        raise RuntimeError(f"all authorized model lanes failed ({len(attempts)} attempts)") from last

    def _record(self,decision,attempts,started):
        body={"decision":asdict(decision),"attempts":attempts,"elapsed_ms":round((time.perf_counter()-started)*1000,3)}
        body["receipt_hash"]=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.traces.append(body)

    def drain_traces(self):
        value=self.traces[:];self.traces.clear();return value
