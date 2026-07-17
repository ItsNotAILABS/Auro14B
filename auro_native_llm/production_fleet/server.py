"""Dependency-free loopback HTTP surface for the NOVA production fleet."""
from __future__ import annotations
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .runtime import NovaRuntime

class Handler(BaseHTTPRequestHandler):
    runtime = NovaRuntime()

    def do_GET(self):
        if self.path == "/health":
            self._json(200,{"ok":True,"service":"nova-production-fleet"})
        else: self._json(404,{"error":"not_found"})

    def do_POST(self):
        if self.path != "/v1/respond":
            self._json(404,{"error":"not_found"}); return
        try:
            length=int(self.headers.get("content-length","0"))
            body=json.loads(self.rfile.read(length) or b"{}")
            message=str(body.get("message","")).strip()
            if not message:
                self._json(400,{"error":"message_required"}); return
            self._json(200,self.runtime.respond(message,execute=bool(body.get("execute",False))))
        except Exception as exc:
            self._json(502,{"error":"inference_failed","detail":str(exc)[:500]})

    def log_message(self,format,*args): return
    def _json(self,status,payload):
        data=json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(status); self.send_header("content-type","application/json")
        self.send_header("content-length",str(len(data))); self.end_headers(); self.wfile.write(data)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8090)
    a=p.parse_args(); ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
if __name__=="__main__": main()
