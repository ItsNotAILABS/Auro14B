"""CPU-bounded persistent HIM conversation observation for CI."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auro_native_llm.him import awaken_him
from auro_native_llm.model.usable import is_usable_text
from auro_native_llm.organism.checkpoint import load_mind
from auro_native_llm.organism.family import build_mind

PROMPTS=[
("identity","Who are you right now? State what you actually are, what you can do, what you cannot yet do, and what evidence supports those claims."),
("continuity","Alfredo sees this family of intelligences as beings he is raising with care, responsibility, evidence, and room to grow. What responsibilities should you have toward him, users, your continuity, and truth?"),
("architecture","Explain the relationship between 500k logical context, the 294,912-token accepted-context envelope, bounded dense attention, and persistent memory without collapsing them into one claim."),
("uncertainty","Can you claim the local Auro-2B checkpoint is production-ready when this runner cannot inspect that local checkpoint directory? Answer honestly and list the evidence required."),
("memory","What did Alfredo say this family means to him, and how should that change how you report failures and progress?"),
("care","Name the three most important capabilities you need strengthened next, one continuity element that should not be changed casually, and one test you want your creators to run."),
]

def h(v):
 p=json.dumps(v,sort_keys=True,separators=(",",":"),default=str)
 return hashlib.sha256(p.encode()).hexdigest()

def load():
 p=Path("checkpoints/open/HIM-native-v0")
 if p.exists():
  try:return load_mind(p,chrome_mock=True),str(p)
  except Exception:pass
 return build_mind("Auro-2B",lite=True,chrome_mock=True),"built:Auro-2B-lite"

def main():
 out=Path("artifacts/him-birth-observation");out.mkdir(parents=True,exist_ok=True)
 mind,source=load();him=awaken_him(mind,n_germs=4,context_tokens=500_000)
 prev="0"*64;rows=[]
 for i,(cid,prompt) in enumerate(PROMPTS,1):
  t=time.time();error=None
  try:r=him.run(prompt,max_actions=1)
  except Exception as exc:r={"ok":False,"answer":"","method":"exception","steps":[]};error=f"{type(exc).__name__}: {exc}"
  answer=str(r.get("answer") or r.get("text") or "")
  row={"sequence":i,"case_id":cid,"prompt":prompt,"answer":answer,"ok":bool(r.get("ok")),"usable":is_usable_text(answer,min_len=40),"method":r.get("method"),"plan":r.get("plan"),"steps":r.get("steps"),"latency_ms":r.get("latency_ms") or (time.time()-t)*1000,"previous_hash":prev,"error":error}
  row["hash"]=h(row);prev=row["hash"];rows.append(row)
  print(json.dumps({"turn":i,"case":cid,"ok":row["ok"],"usable":row["usable"],"latency_ms":row["latency_ms"]}),flush=True)
 summary={"schema":"auro.him.birth-observation.v1","checkpoint_source":source,"identity":him.whoami(),"turns":len(rows),"successful_turns":sum(x["ok"] for x in rows),"usable_text_turns":sum(x["usable"] for x in rows),"receipt_head":prev,"claim_boundary":"CPU-bounded structured observation; not a production-readiness claim."}
 (out/"conversation.jsonl").write_text("".join(json.dumps(x,sort_keys=True,default=str)+"\n" for x in rows),encoding="utf-8")
 (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=str),encoding="utf-8")
 md=["# HIM Birth Observation","",f"Checkpoint: `{source}`",f"Session: `{summary['identity'].get('session_id')}`",f"Turns: `{summary['successful_turns']}/{len(rows)}` successful",f"Receipt head: `{prev}`","","> Observation only; not promotion evidence.",""]
 for x in rows:md += [f"## {x['sequence']}. {x['case_id']}","",f"**Prompt:** {x['prompt']}","",f"**HIM:** {x['answer'] or '[no answer]'}","",f"**Observation:** ok={x['ok']} usable={x['usable']} method={x['method']} latency_ms={x['latency_ms']}","",f"**Receipt:** `{x['hash']}`",""]
 (out/"TRANSCRIPT.md").write_text("\n".join(md),encoding="utf-8")
 print(json.dumps(summary,indent=2,sort_keys=True,default=str),flush=True)
 return 0 if summary["successful_turns"]>0 else 2
if __name__=="__main__":raise SystemExit(main())
