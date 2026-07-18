"""Dependency-free loopback HTTP surface for the NOVA production fleet."""
from __future__ import annotations
import argparse, hmac, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .runtime import NovaRuntime
from .console import ASSETS

MAX_REQUEST_BYTES = 1_048_576

def token_authorized(header: str, expected: str) -> bool:
    if not expected or not header.startswith("Bearer "): return False
    return hmac.compare_digest(header[7:],expected)

class Handler(BaseHTTPRequestHandler):
    runtime = NovaRuntime()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ASSETS:
            content_type, data = ASSETS[path]
            self._bytes(200, content_type, data)
        elif path == "/health":
            self._json(200,{"ok":True,"service":"nova-production-fleet"})
        elif path == "/v1/capabilities":
            self._json(200,self.runtime.capabilities.manifest())
        elif path == "/v1/receipts/verify":
            self._json(200,self.runtime.capabilities.ledger.verify())
        elif path == "/v1/receipts":
            self._json(200,{"receipts":self.runtime.capabilities.ledger.tail(20)})
        else: self._json(404,{"error":"not_found"})

    def do_POST(self):
        if self.path == "/v1/capabilities/call":
            try:
                body=self._body()
                approved=bool(body.get("approved",False))
                if approved and not self._authorized(): self._json(403,{"error":"operator_token_required"}); return
                self._json(200,self.runtime.capabilities.call(str(body.get("name","")),dict(body.get("arguments") or {}),approved=approved))
            except Exception as exc: self._json(400,{"error":"capability_call_failed","detail":str(exc)[:500]})
            return
        if self.path != "/v1/respond":
            self._json(404,{"error":"not_found"}); return
        try:
            body=self._body()
            message=str(body.get("message","")).strip()
            if not message:
                self._json(400,{"error":"message_required"}); return
            execute=bool(body.get("execute",False))
            if execute and not self._authorized(): self._json(403,{"error":"operator_token_required"}); return
            self._json(200,self.runtime.respond(message,execute=execute))
        except Exception as exc:
            self._json(502,{"error":"inference_failed","detail":str(exc)[:500]})

    def log_message(self,format,*args): return
    def _authorized(self):
        return token_authorized(self.headers.get("authorization",""),os.getenv("AURO_EXECUTION_TOKEN",""))
    def _body(self):
        length=int(self.headers.get("content-length","0"))
        if length < 0 or length > MAX_REQUEST_BYTES: raise ValueError("request_body_too_large")
        return json.loads(self.rfile.read(length) or b"{}")
    def _json(self,status,payload):
        data=json.dumps(payload,ensure_ascii=False).encode()
        self._bytes(status,"application/json; charset=utf-8",data)
    def _bytes(self,status,content_type,data):
        self.send_response(status); self.send_header("content-type",content_type)
        self.send_header("content-length",str(len(data))); self.send_header("cache-control","no-store")
        self.send_header("x-content-type-options","nosniff"); self.send_header("x-frame-options","DENY")
        self.send_header("referrer-policy","no-referrer")
        self.send_header("content-security-policy","default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers(); self.wfile.write(data)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8090)
    a=p.parse_args(); ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
if __name__=="__main__": main()
