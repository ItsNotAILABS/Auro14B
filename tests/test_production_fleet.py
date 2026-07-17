import json
from auro_native_llm.production_fleet.runtime import ModelEndpoint, NovaRuntime

class FakeGenerator:
    def __init__(self): self.calls=0
    def __call__(self,messages,options):
        self.calls += 1
        if self.calls <= 5:
            return {"text":json.dumps({"summary":f"agent-{self.calls}","confidence":0.8,"evidence":["fixture"],"proposed_actions":[]})}
        return {"text":json.dumps({"answer":"usable answer","reasoning_summary":["verified"],"confidence":0.9,"actions":[{"tool":"capsula","arguments":{"task":"build"},"reason":"approved proposal"}]})}

def test_council_answers_without_executing():
    endpoint=ModelEndpoint("test","http://127.0.0.1:1/v1","test",8_200_000_000)
    result=NovaRuntime(endpoint,generator=FakeGenerator()).respond("Build it")
    assert result["answer"] == "usable answer"
    assert len(result["agents"]) == 5
    assert result["approved_actions"] == []
    assert result["model"]["parameter_count"] == 8_200_000_000
    assert result["model"]["agent_count_is_not_parameter_count"] is True

def test_explicit_execute_only_approves_bounded_tools():
    endpoint=ModelEndpoint("test","http://127.0.0.1:1/v1","test")
    result=NovaRuntime(endpoint,generator=FakeGenerator()).respond("Build it",execute=True)
    assert result["approved_actions"][0]["tool"] == "capsula"

