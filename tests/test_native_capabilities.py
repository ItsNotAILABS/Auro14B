from auro_native_llm.production_fleet.capabilities import NativeCapabilities

class Brain:
 def state(self): return {"beat":7}
 def query(self,path): return {"path":path}
class Mat:
 def rank_text(self,q,c,k): return {"top":[0]}
 def call(self,n,a): return {"name":n}
class Caps:
 def create_session(self,runtime,name): return {"id":"s1"}
 def write_file(self,*a): return {"path":a[1]}
 def run(self,s): return {"exit_code":0}
 def manifest(self,s): return {"session":s}
class SDK: brain=Brain(); matdaemon=Mat(); capsula=Caps()

def test_manifest_includes_skills_and_tools():
 names={x["name"] for x in NativeCapabilities(SDK()).manifest()["capabilities"]}
 assert {"skill.research","skill.build","brain.state","memory.rank_text"} <= names

def test_mutation_denied_without_approval():
 out=NativeCapabilities(SDK()).call("build.create_session",{"runtime":"python","name":None})
 assert out["denied"] is True

def test_approved_build_and_read_tool():
 caps=NativeCapabilities(SDK())
 assert caps.call("build.create_session",{"runtime":"python","name":None},approved=True)["ok"]
 assert caps.call("brain.state",{})["output"]["beat"] == 7

def test_unknown_arguments_rejected():
 try: NativeCapabilities(SDK()).call("brain.state",{"surprise":1})
 except ValueError as exc: assert "unknown arguments" in str(exc)
 else: raise AssertionError("validation should reject unknown arguments")
