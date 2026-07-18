import json
import os
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from auro_native_llm.production_fleet.client import AuroAPIError, AuroClient
from auro_native_llm.production_fleet.server import Handler, ThreadingHTTPServer, extract_user_message, openai_completion


class _Ledger:
 def verify(self): return {"ok":True,"count":1}
 def tail(self,n): return [{"kind":"model_response","hash":"abc","ok":True}]

class _Capabilities:
 def __init__(self): self.ledger=_Ledger()
 def manifest(self): return {"schema":"test","capabilities":[{"name":"brain.state"}]}
 def call(self,name,arguments,approved=False): return {"ok":True,"capability":name,"approved":approved,"arguments":arguments}

class _Runtime:
 def __init__(self):
  self.endpoint=SimpleNamespace(id="auro-test",model="auro-test",role="orchestrator",parameter_count=123)
  self.capabilities=_Capabilities()
 def respond(self,message,execute=False):
  return {"schema":"nova.production.response.v1","answer":"HIM: "+message,"confidence":.9,"reasoning_summary":["checked"],"agents":[],"proposed_actions":[],"executions":[],"model":{"model":"auro-test","parameter_count_verified":True},"receipt":{"hash":"abc"}}

def _server():
 old=Handler.runtime; Handler.runtime=_Runtime()
 server=ThreadingHTTPServer(("127.0.0.1",0),Handler); Thread(target=server.serve_forever,daemon=True).start()
 return old,server,f"http://127.0.0.1:{server.server_port}"

def _stop(old,server):
 server.shutdown(); server.server_close(); Handler.runtime=old

def test_extract_user_message_uses_final_user_turn():
 assert extract_user_message([{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"user","content":"three"}])=="three"

def test_openai_adapter_keeps_auro_evidence():
 payload=openai_completion(_Runtime().respond("hello"),"req_test")
 assert payload["object"]=="chat.completion"
 assert payload["choices"][0]["message"]["content"]=="HIM: hello"
 assert payload["auro"]["receipt"]["hash"]=="abc"

def test_native_and_openai_api_contracts():
 old,server,base=_server()
 try:
  client=AuroClient(base)
  assert client.models()["data"][0]["parameter_count_verified"] is True
  native=client.respond("native hello"); assert native["answer"]=="HIM: native hello"; assert native["request_id"]=="sdk_request"
  chat=client.chat([{"role":"user","content":"compatible hello"}],model="auro-test")
  assert chat["choices"][0]["message"]["content"]=="HIM: compatible hello"
  assert chat["auro"]["parameter_count_verified"] is True
  with urlopen(base+"/openapi.json") as response: assert json.loads(response.read())["openapi"]=="3.1.0"
 finally: _stop(old,server)

def test_api_auth_and_execution_are_separate():
 old,server,base=_server(); prior_api=os.environ.get("AURO_API_TOKEN"); prior_exec=os.environ.get("AURO_EXECUTION_TOKEN")
 os.environ["AURO_API_TOKEN"]="read-secret"; os.environ["AURO_EXECUTION_TOKEN"]="exec-secret"
 try:
  try: AuroClient(base).models()
  except AuroAPIError as exc: assert exc.status==401 and exc.code=="api_token_required"
  else: raise AssertionError("API must reject missing bearer token")
  client=AuroClient(base,api_token="read-secret")
  assert client.respond("read only")["answer"]=="HIM: read only"
  try: client.respond("execute",execute=True)
  except AuroAPIError as exc: assert exc.status==403 and exc.code=="operator_token_required"
  else: raise AssertionError("execution must require a separate token")
  executing=AuroClient(base,api_token="read-secret",execution_token="exec-secret")
  assert executing.respond("execute",execute=True)["answer"]=="HIM: execute"
 finally:
  if prior_api is None: os.environ.pop("AURO_API_TOKEN",None)
  else: os.environ["AURO_API_TOKEN"]=prior_api
  if prior_exec is None: os.environ.pop("AURO_EXECUTION_TOKEN",None)
  else: os.environ["AURO_EXECUTION_TOKEN"]=prior_exec
  _stop(old,server)

def test_streaming_is_explicitly_rejected_with_structured_error():
 old,server,base=_server()
 try:
  req=Request(base+"/v1/chat/completions",data=json.dumps({"model":"auro-test","stream":True,"messages":[{"role":"user","content":"hello"}]}).encode(),headers={"content-type":"application/json"},method="POST")
  try: urlopen(req)
  except HTTPError as exc:
   payload=json.loads(exc.read()); assert exc.code==400; assert payload["error"]["code"]=="streaming_not_supported"; assert payload["error"]["request_id"]
  else: raise AssertionError("unsupported streaming must fail explicitly")
 finally: _stop(old,server)

