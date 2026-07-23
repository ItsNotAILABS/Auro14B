"""Four dialogue turns, then autonomous SignalLens development with full logs."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from typing import Any

from auro_native_llm.him.mature import awaken_mature_him
from auro_native_llm.model.usable import is_usable_text
from auro_native_llm.organism.checkpoint import load_mind
from auro_native_llm.organism.family import build_mind

MISSION="""NEXUS Relay / SignalLens watches LangChain SSRF hardening, LiteLLM proxy controls, MCP SDK authorization patterns, and Qdrant/Milvus retrieval features. Use primary sources, separate verified facts from inference, preserve uncertainty, and produce actionable signals rather than marketing summaries."""
SNAPSHOT="""Observer-supplied source snapshot, 2026-07-23: LangChain advisories and releases include SSRF hardening themes such as URL handling, redirects, and DNS rebinding. LiteLLM documents centralized proxy authentication/authorization, virtual keys, budgets, spend tracking, hooks, logging, and rate limits. MCP authorization is optional; HTTP transports follow OAuth-style patterns when supported while STDIO commonly obtains credentials from the environment. Qdrant and Milvus are monitored for dense/sparse hybrid retrieval, filtering, reranking, multivector or equivalent advanced retrieval, and operational changes. Exact current claims require source verification before alerts."""
DIALOGUE=[
("mission",f"Read this mission and tell me what work you understand.\n\n{MISSION}"),
("evidence","What evidence rules will you follow so security, authorization, proxy-control, and retrieval claims are not overstated?"),
("priorities","Which changes are urgent security signals, which are architectural signals, and which are normal feature tracking?"),
("handoff","After this answer, stop treating this as conversation. State the autonomous sequence you will execute: generate, read, challenge, deepen, revise, report."),
]


def h(v:Any)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def load():
 for p in (Path("checkpoints/auro_minds/Auro-2B_continual"),Path("checkpoints/auro_minds/Auro-2B_physics"),Path("checkpoints/open/HIM-native-v0")):
  if p.exists():
   try:return load_mind(p,chrome_mock=True),str(p)
   except Exception:pass
 return build_mind("Auro-2B",lite=True,chrome_mock=True),"built:Auro-2B-lite"

def run_record(him,kind,phase,instruction,seq,prev):
 t=time.time();err=None
 try:r=him.run(instruction,max_actions=5)
 except Exception as exc:r={"ok":False,"answer":"","method":"exception","steps":[]};err=f"{type(exc).__name__}: {exc}"
 out=str(r.get("answer") or r.get("text") or "")
 row={"sequence":seq,"kind":kind,"phase":phase,"instruction":instruction,"output":out,"ok":bool(r.get("ok")),"usable":is_usable_text(out,min_len=40),"method":r.get("method"),"plan":r.get("plan"),"steps":r.get("steps"),"latency_ms":r.get("latency_ms") or (time.time()-t)*1000,"language_receipt":r.get("language_receipt"),"previous_hash":prev,"error":err}
 row["hash"]=h(row);return row

def main():
 root=Path("artifacts/him-birth-observation");root.mkdir(parents=True,exist_ok=True)
 mind,source=load();him=awaken_mature_him(mind,n_germs=20,context_tokens=500_000)
 him.colony.context.ingest(MISSION,kind="system",meta={"program":"NEXUS Relay / SignalLens"})
 him.colony.context.ingest(SNAPSHOT,kind="evidence",meta={"as_of":"2026-07-23"})
 him.lexicon.ingest(MISSION,source="mission");him.lexicon.ingest(SNAPSHOT,source="source-snapshot")
 rows=[];prev="0"*64;seq=0
 for phase,prompt in DIALOGUE:
  seq+=1;row=run_record(him,"dialogue",phase,prompt,seq,prev);prev=row["hash"];rows.append(row);print(json.dumps({"kind":"dialogue","phase":phase,"ok":row["ok"]}),flush=True)
 task=f"{MISSION}\n\n{SNAPSHOT}\n\nProduce a giant evidence-aware research report and operational watch design."
 development=him.develop(task,cycles=4)
 for stage in development["stages"]:
  seq+=1;r=stage["report"];out=str(r.get("answer") or r.get("text") or "")
  row={"sequence":seq,"kind":"autonomous_work","phase":f"development_{stage['stage']}","instruction":stage["instruction"],"output":out,"ok":bool(r.get("ok")),"usable":is_usable_text(out,min_len=40),"method":r.get("method"),"plan":r.get("plan"),"steps":r.get("steps"),"latency_ms":r.get("latency_ms"),"language_receipt":r.get("language_receipt"),"previous_hash":prev,"error":None};row["hash"]=h(row);prev=row["hash"];rows.append(row)
  (root/f"{seq:02d}-{row['phase']}.md").write_text(out,encoding="utf-8")
 seq+=1;final=development["final"]
 row={"sequence":seq,"kind":"autonomous_work","phase":"final_report","instruction":"Seal final SignalLens research report after generation, readback, red-team, and revision.","output":final,"ok":bool(final),"usable":is_usable_text(final,min_len=40),"method":"mature_him_development_final","plan":None,"steps":[],"latency_ms":None,"language_receipt":{"lexicon":development["lexicon"]},"previous_hash":prev,"error":None};row["hash"]=h(row);prev=row["hash"];rows.append(row)
 (root/"FINAL_SIGNALLENS_REPORT.md").write_text(final,encoding="utf-8")
 summary={"schema":"auro.him.signallens-development.v1","checkpoint_source":source,"identity":him.whoami(),"dialogue_turns":4,"autonomous_work_records":len(rows)-4,"successful_records":sum(x["ok"] for x in rows),"total_records":len(rows),"receipt_head":prev,"lexicon":him.lexicon.manifest(),"claim_boundary":"Observed repository runtime and supplied snapshot; competitive superiority requires reproducible external benchmarks."}
 (root/"cycle.jsonl").write_text("".join(json.dumps(x,sort_keys=True,default=str)+"\n" for x in rows),encoding="utf-8");(root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=str),encoding="utf-8")
 md=["# HIM SignalLens Development Cycle","",f"Checkpoint: `{source}`",f"Session: `{summary['identity'].get('session_id')}`",""]
 for x in rows:md += [f"## {x['sequence']}. {x['kind']}: {x['phase']}","",f"**Instruction:** {x['instruction']}","",f"**HIM:** {x['output'] or '[no output]'}","",f"**Receipt:** `{x['hash']}`",""]
 (root/"TRANSCRIPT.md").write_text("\n".join(md),encoding="utf-8");print(json.dumps(summary,indent=2,sort_keys=True),flush=True);return 0 if summary["successful_records"] else 2
if __name__=="__main__":raise SystemExit(main())
