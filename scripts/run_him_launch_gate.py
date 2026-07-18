"""Emit an honest, reproducible HIM public-alpha readiness receipt."""
from __future__ import annotations
import hashlib,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from auro_native_llm.open_weights import ByteTokenizer,OpenHIM
from auro_native_llm.production_fleet.context_engine import ContextEngine
from auro_native_llm.production_fleet.model_orchestrator import ModelLane,MultiModelOrchestrator

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def generator(name):
 def call(messages,options):return {"text":json.dumps({"answer":name,"actions":[]})}
 return call

def main():
 started=time.time();checks=[]
 checkpoint=ROOT/"checkpoints/open/HIM-native-v0"
 expected=json.loads((checkpoint/"SHA256SUMS.json").read_text())
 hashes={name:sha(checkpoint/name) for name in expected}
 checks.append({"id":"checkpoint_integrity","pass":hashes==expected,"evidence":hashes})
 tokenizer=ByteTokenizer();sample="HIM φ 日本語\n  def x():\treturn '✓'"
 checks.append({"id":"tokenizer_lossless","pass":tokenizer.decode(tokenizer.encode(sample))==sample,
                "evidence":tokenizer.manifest()})
 model=OpenHIM.load(checkpoint)
 checks.append({"id":"parameter_accounting","pass":model.num_parameters==146576,
                "evidence":{"measured_parameters":model.num_parameters,"family_label_not_parameter_count":True}})
 with tempfile.TemporaryDirectory(prefix="him-launch-gate-") as root:
  engine=ContextEngine(Path(root)/"context.sqlite",1200)
  ingest=engine.ingest("alpha receipt owner-signing context. "*120_000,source="million.txt",
                       chunk_tokens=700,overlap_tokens=60)
  pack=engine.retrieve("owner signing receipt",token_budget=1200)
  checks.append({"id":"million_token_context","pass":ingest["tokens"]>=1_000_000 and 0<pack.injected_tokens<=1200,
                 "evidence":{"logical_tokens":pack.logical_tokens,"injected_tokens":pack.injected_tokens,
                             "budget":pack.token_budget,"receipt_hash":pack.receipt_hash}})
 local=ModelLane("him-general","HIM-General","general","repository-native-open-weights",
                 generator("general"),100,("general",),5,True,True,"a"*64)
 coder=ModelLane("him-coder","HIM-Coder","coding","repository-native-open-weights",
                 generator("coder"),100,("code",),10,True,True,"b"*64)
 fleet=MultiModelOrchestrator([local,coder])
 routed=fleet([{"role":"user","content":"write Python code and tests"}],{})
 checks.append({"id":"specialist_routing","pass":routed["routed_model"]["id"]=="him-coder",
                "evidence":fleet.drain_traces()[0]})
 raw=model.generate("<user> What is HIM?\n<assistant>",max_new_tokens=80,temperature=.25,top_k=8)
 from auro_native_llm.model.usable import is_usable_text
 raw_quality=is_usable_text(raw,min_len=40)
 checks.append({"id":"bundled_general_chat_quality","pass":False,
                "critical_for":"general_model_release",
                "evidence":{"heuristic_printable":raw_quality,"sample":raw[-300:],
                            "assessment":"reference checkpoint remains undertrained; production contract gate required"}})
 alpha_ids={"checkpoint_integrity","tokenizer_lossless","parameter_accounting","million_token_context","specialist_routing"}
 alpha_ready=all(x["pass"] for x in checks if x["id"] in alpha_ids)
 result={"schema":"him.public_alpha.readiness.v1","generated_at":time.time(),
         "public_alpha_runtime_ready":alpha_ready,"general_model_release_ready":False,
         "marketing_posture":"public alpha / hackathon-ready runtime" if alpha_ready else "development",
         "checks":checks,"critical_blockers":[
          "Bundled HIM-native-v0 is not a fluent general-purpose checkpoint.",
          "PyTorch portability suites require the ML test environment.",
          "A promoted Auro checkpoint needs official benchmark and safety evaluation."
         ],"elapsed_seconds":round(time.time()-started,3)}
 out=ROOT/"evidence/him-public-alpha-readiness.json";out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(result,indent=2),encoding="utf-8")
 print(json.dumps(result,indent=2));return 0 if alpha_ready else 1

if __name__=="__main__":raise SystemExit(main())
