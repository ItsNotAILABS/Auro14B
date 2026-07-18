import json
from auro_native_llm.production_fleet.runtime import ModelEndpoint, NovaRuntime

class FakeGenerator:
    def __init__(self): self.calls=0
    def __call__(self,messages,options):
        self.calls += 1
        if self.calls <= 5:
            return {"text":json.dumps({"summary":f"agent-{self.calls}","confidence":0.8,"evidence":["fixture"],"proposed_actions":[]})}
        return {"text":json.dumps({"answer":"usable answer","reasoning_summary":["verified"],"confidence":0.9,"actions":[{"tool":"capsula","arguments":{"task":"build"},"reason":"approved proposal"}]})}

class FakeSDK:
    def manifest(self): return {"schema":"test.sdk"}
    def action_contract(self): return {"capsula":{"tool":"capsula"}}
    def execute(self,action): return {"tool":action["tool"],"ok":True}

def test_council_answers_without_executing():
    endpoint=ModelEndpoint("test","http://127.0.0.1:1/v1","test",8_200_000_000)
    result=NovaRuntime(endpoint,generator=FakeGenerator(),sdk=FakeSDK()).respond("Build it")
    assert result["answer"] == "usable answer"
    assert len(result["agents"]) == 5
    assert result["approved_actions"] == []
    assert result["model"]["parameter_count"] == 8_200_000_000
    assert result["model"]["agent_count_is_not_parameter_count"] is True

def test_explicit_execute_only_approves_bounded_tools():
    endpoint=ModelEndpoint("test","http://127.0.0.1:1/v1","test")
    result=NovaRuntime(endpoint,generator=FakeGenerator(),sdk=FakeSDK()).respond("Build it",execute=True)
    assert result["approved_actions"][0]["tool"] == "capsula"
    assert result["executions"][0]["ok"] is True

def test_runtime_virtualizes_and_reuses_persistent_context(tmp_path,monkeypatch):
    monkeypatch.setenv("AURO_CONTEXT_DB",str(tmp_path/"context.sqlite"))
    endpoint=ModelEndpoint("test","http://127.0.0.1:1/v1","test",100)
    runtime=NovaRuntime(endpoint,generator=FakeGenerator(),sdk=FakeSDK())
    first=runtime.respond("Remember project codename HELIOS")
    assert first["context"]["logical_tokens"]==0
    second=runtime.respond("What is the project codename?")
    assert second["context"]["logical_tokens"]>0
    assert "HELIOS" in second["context"]["context"]
    assert second["context"]["injected_tokens"]<=second["context"]["token_budget"]
