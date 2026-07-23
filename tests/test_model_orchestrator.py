import json
from auro_native_llm.production_fleet.model_orchestrator import ModelLane,MultiModelOrchestrator

class Generator:
    def __init__(self,name,fail=False): self.name=name;self.fail=fail;self.calls=0
    def __call__(self,messages,options):
        self.calls+=1
        if self.fail: raise RuntimeError("failed")
        return {"text":json.dumps({"answer":self.name}),"raw_model":self.name}

def lane(id,capabilities=("general",),local=False,priority=10,fail=False,provider=None):
    return ModelLane(id,id,"specialist",provider or ("repository-native-open-weights" if local else "explicit-hosted"),
                     Generator(id,fail),123,capabilities,priority,local,True,"a"*64 if local else None)

def test_routes_code_to_capable_local_lane_and_receipts_it():
    fleet=MultiModelOrchestrator([lane("general",local=True,priority=5),lane("coder",("code",),local=True)])
    result=fleet([{"role":"user","content":"debug this Python function and write tests"}],{})
    assert result["routed_model"]["id"]=="coder"
    trace=fleet.drain_traces()[0]
    assert trace["decision"]["task"]=="code"
    assert len(trace["receipt_hash"])==64
    assert trace["attempts"][0]["model"]=="coder"

def test_hosted_fallback_is_fail_closed_by_default():
    local=lane("him-local",local=True,fail=True)
    hosted=lane("cloud",provider="workers-ai-explicit")
    fleet=MultiModelOrchestrator([local,hosted])
    try: fleet([{"role":"user","content":"answer"}],{})
    except RuntimeError: pass
    else: raise AssertionError("hosted fallback must not run")
    assert hosted.generator.calls==0

def test_explicit_hosted_fallback_preserves_model_identity():
    local=lane("him-local",local=True,fail=True)
    hosted=lane("cloud",provider="workers-ai-explicit")
    fleet=MultiModelOrchestrator([local,hosted],allow_hosted_fallback=True)
    result=fleet([{"role":"user","content":"answer"}],{})
    assert result["routed_model"]["id"]=="cloud"
    assert result["routed_model"]["provider"]=="workers-ai-explicit"
    assert [x["ok"] for x in fleet.drain_traces()[0]["attempts"]]==[False,True]

def test_manifest_never_counts_agents_or_sums_parameters():
    fleet=MultiModelOrchestrator([lane("a"),lane("b")])
    manifest=fleet.manifest()
    assert manifest["model_count"]==2
    assert manifest["parameter_accounting"]=="per-lane weights; never agents or token counts"
    assert [x["parameter_count"] for x in manifest["models"]]==[123,123]
