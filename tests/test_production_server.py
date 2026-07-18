import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from auro_native_llm.production_fleet.server import Handler, ThreadingHTTPServer, token_authorized

def test_execution_token_is_fail_closed():
 assert token_authorized("Bearer secret","") is False
 assert token_authorized("","secret") is False
 assert token_authorized("secret","secret") is False

def test_execution_token_accepts_exact_bearer_only():
 assert token_authorized("Bearer secret","secret") is True
 assert token_authorized("Bearer wrong","secret") is False

def test_console_assets_are_real_and_security_headers_present():
 server=ThreadingHTTPServer(("127.0.0.1",0),Handler); thread=Thread(target=server.serve_forever,daemon=True); thread.start()
 try:
  for path,marker in (("/","Auro operator console"),("/console.css","--signal"),("/console.js","function boot")):
   with urlopen(f"http://127.0.0.1:{server.server_port}{path}") as response:
    body=response.read().decode(); assert marker in body
    assert response.headers["x-content-type-options"]=="nosniff"
    assert "default-src 'self'" in response.headers["content-security-policy"]
 finally: server.shutdown(); server.server_close()

def test_console_api_health_and_unknown_route():
 server=ThreadingHTTPServer(("127.0.0.1",0),Handler); thread=Thread(target=server.serve_forever,daemon=True); thread.start()
 try:
  with urlopen(f"http://127.0.0.1:{server.server_port}/health") as response:
   assert json.loads(response.read())["ok"] is True
  try: urlopen(f"http://127.0.0.1:{server.server_port}/not-real")
  except HTTPError as exc: assert exc.code==404
  else: raise AssertionError("unknown route must be 404")
 finally: server.shutdown(); server.server_close()
