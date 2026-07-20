"""Ingest large local corpora into HIM's virtualized Python context store."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from auro_native_llm.production_fleet.context_engine import ContextEngine

TEXT_EXT={".txt",".md",".rst",".py",".js",".ts",".tsx",".jsx",".json",".jsonl",".csv",".toml",".yaml",".yml",".sol",".rs",".jl",".mo",".ps1"}

def files(paths):
 for raw in paths:
  path=Path(raw)
  if path.is_file():yield path
  elif path.is_dir():
   for child in path.rglob("*"):
    if child.is_file() and child.suffix.lower() in TEXT_EXT and ".git" not in child.parts:yield child

def main():
 p=argparse.ArgumentParser(description="HIM million-token logical context utility")
 p.add_argument("--db",default="state/him-context.sqlite")
 sub=p.add_subparsers(dest="command",required=True)
 ingest=sub.add_parser("ingest");ingest.add_argument("paths",nargs="+");ingest.add_argument("--importance",type=float,default=.6)
 query=sub.add_parser("query");query.add_argument("text");query.add_argument("--tokens",type=int,default=8000);query.add_argument("--top-k",type=int,default=12)
 sub.add_parser("stats");a=p.parse_args();engine=ContextEngine(a.db)
 if a.command=="ingest":
  results=[];failed=[]
  for path in files(a.paths):
   try:results.append(engine.ingest(path.read_text(encoding="utf-8"),source=str(path),importance=a.importance))
   except (UnicodeDecodeError,ValueError,OSError) as exc:failed.append({"path":str(path),"error":str(exc)})
  print(json.dumps({"ingested":len(results),"failed":failed,"stats":engine.stats()},indent=2))
 elif a.command=="query":print(json.dumps(engine.retrieve(a.text,token_budget=a.tokens,top_k=a.top_k).public(),indent=2))
 else:print(json.dumps(engine.stats(),indent=2))

if __name__=="__main__":main()
