"""One-command HIM launch for local evaluation and real endpoint fleets."""
from __future__ import annotations
import argparse,json,os,secrets,sys,threading,webbrowser
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(description="Launch HIM chat, context, models, tools and receipts")
 p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8090)
 p.add_argument("--checkpoint",default="checkpoints/open/HIM-native-v0")
 p.add_argument("--base-url",help="Explicit OpenAI-compatible local/remote endpoint ending in /v1")
 p.add_argument("--model",help="Model ID served by --base-url")
 p.add_argument("--parameter-count",type=int)
 p.add_argument("--fleet",type=Path,help="JSON array of additional real model lanes")
 p.add_argument("--context-db",default="state/him-context.sqlite")
 p.add_argument("--context-tokens",type=int,default=32000,help="Per-turn retrieved injection budget (max 300000)")
 p.add_argument("--secure",action="store_true",help="require a generated read token, even on loopback")
 p.add_argument("--no-browser",action="store_true")
 a=p.parse_args()
 if a.host not in {"127.0.0.1","localhost","::1"} and not a.secure:
  p.error("non-loopback binding requires --secure")
 os.environ["AURO_CONTEXT_DB"]=a.context_db;os.environ["AURO_CONTEXT_INJECTION_TOKENS"]=str(a.context_tokens)
 if a.base_url:
  if not a.model:p.error("--base-url requires --model")
  os.environ.pop("AURO_NATIVE_CHECKPOINT",None);os.environ["AURO_BASE_URL"]=a.base_url;os.environ["AURO_MODEL"]=a.model
  if a.parameter_count:os.environ["AURO_PARAMETER_COUNT"]=str(a.parameter_count)
  mode=f"explicit endpoint · {a.model}"
 else:
  checkpoint=Path(a.checkpoint)
  if not checkpoint.exists():p.error(f"checkpoint not found: {checkpoint}")
  os.environ["AURO_NATIVE_CHECKPOINT"]=str(checkpoint);mode=f"repository-native · {checkpoint}"
 if a.fleet:os.environ["AURO_MODEL_FLEET_JSON"]=a.fleet.read_text(encoding="utf-8")
 execution=os.environ.setdefault("AURO_EXECUTION_TOKEN",secrets.token_urlsafe(24))
 read=None
 if a.secure:read=os.environ.setdefault("AURO_API_TOKEN",secrets.token_urlsafe(24))
 from auro_native_llm.production_fleet.server import Handler,ThreadingHTTPServer
 url=f"http://{a.host}:{a.port}/"
 print("\nHIM IS READY")
 print(f"Console: {url}\nMode: {mode}\nContext: {a.context_db} · injection budget {a.context_tokens:,}")
 print(f"Execution token (paste only when approving writes): {execution}")
 if read:print(f"Read/API token: {read}")
 if "HIM-native-v0" in str(a.checkpoint) and not a.base_url:
  print("Quality boundary: bundled HIM-native-v0 is a 146,576-parameter reference checkpoint; "
        "quality-gated local orchestration handles known domains. Use --base-url for broad fluent chat.")
 server=ThreadingHTTPServer((a.host,a.port),Handler)
 if not a.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()

if __name__=="__main__":main()
