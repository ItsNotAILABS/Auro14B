import json
from auro_native_llm.production_fleet.runtime import ModelEndpoint,NovaRuntime

class SDK:
 def manifest(self):return {"schema":"test"}
 def action_contract(self):return {}
 def execute(self,action):return action

class InvalidGenerator:
 def __call__(self,messages,options):return {"text":"raw undertrained token noise"}

class ValidGenerator:
 def __call__(self,messages,options):
  system=messages[0]["content"]
  if "governing a council" in system:return {"text":json.dumps({"answer":"fluent model answer","reasoning_summary":[],"confidence":.9,"actions":[]})}
  return {"text":json.dumps({"summary":"checked","confidence":.8,"evidence":[],"proposed_actions":[]})}

def test_invalid_model_contract_uses_labeled_local_fallback(tmp_path,monkeypatch):
 monkeypatch.setenv("AURO_CONTEXT_DB",str(tmp_path/"c.sqlite"))
 result=NovaRuntime(ModelEndpoint("m","local://m","m",10),generator=InvalidGenerator(),sdk=SDK()).respond("What is HIM?")
 assert result["answer_origin"]=="local_orchestration_fallback"
 assert result["generation_quality"]["contract_valid"] is False
 assert "I am HIM" in result["answer"]

def test_valid_model_contract_remains_model_generation(tmp_path,monkeypatch):
 monkeypatch.setenv("AURO_CONTEXT_DB",str(tmp_path/"c.sqlite"))
 result=NovaRuntime(ModelEndpoint("m","local://m","m",10),generator=ValidGenerator(),sdk=SDK()).respond("hello")
 assert result["answer"]=="fluent model answer"
 assert result["answer_origin"]=="model_generation"
 assert result["generation_quality"]["fallback"] is None

def test_invalid_model_uses_retrieved_fact_before_generic_fallback(tmp_path,monkeypatch):
 monkeypatch.setenv("AURO_CONTEXT_DB",str(tmp_path/"c.sqlite"))
 runtime=NovaRuntime(ModelEndpoint("m","local://m","m",10),generator=InvalidGenerator(),sdk=SDK())
 runtime.context.ingest("The private project codename is ORPHEUS-77.",source="brief.md")
 result=runtime.respond("What is the private project codename?")
 assert "ORPHEUS-77" in result["answer"]
 assert "brief.md" in result["answer"]
 assert result["generation_quality"]["fallback"]["method"]=="grounded_context_extractive"
