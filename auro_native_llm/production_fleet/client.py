"""Dependency-free Python SDK for the Auro14B · HIM API."""
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class AuroAPIError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, request_id: str | None = None):
        self.status, self.code, self.request_id = status, code, request_id
        super().__init__(f"{code}: {message}")


class AuroClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8090", *, api_token: str | None = None,
                 execution_token: str | None = None, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.execution_token = execution_token
        self.timeout = timeout

    def discovery(self) -> dict[str, Any]: return self._request("GET", "/v1")
    def models(self) -> dict[str, Any]: return self._request("GET", "/v1/models")
    def capabilities(self) -> dict[str, Any]: return self._request("GET", "/v1/capabilities")
    def receipts(self) -> dict[str, Any]: return self._request("GET", "/v1/receipts")
    def verify_receipts(self) -> dict[str, Any]: return self._request("GET", "/v1/receipts/verify")
    def context_stats(self) -> dict[str, Any]: return self._request("GET","/v1/context")
    def query_context(self, query: str, *, token_budget: int = 32_000, top_k: int = 24) -> dict[str, Any]:
        return self._request("POST","/v1/context/query",{"query":query,"token_budget":token_budget,"top_k":top_k})
    def ingest_context(self,text:str,*,source:str="sdk",kind:str="document",importance:float=.5) -> dict[str, Any]:
        return self._request("POST","/v1/context/ingest",
            {"text":text,"source":source,"kind":kind,"importance":importance},execution=True)

    def respond(self, message: str, *, execute: bool = False) -> dict[str, Any]:
        return self._request("POST", "/v1/him/respond", {"message": message, "execute": execute}, execution=execute)

    def chat(self, messages: list[dict[str, str]], *, model: str = "auro-him", execute: bool = False) -> dict[str, Any]:
        return self._request("POST", "/v1/chat/completions", {
            "model": model, "messages": messages, "stream": False, "auro_execute": execute
        }, execution=execute)

    def call(self, name: str, arguments: dict[str, Any], *, approved: bool = False) -> dict[str, Any]:
        return self._request("POST", "/v1/capabilities/call", {
            "name": name, "arguments": arguments, "approved": approved
        }, execution=approved)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None, *, execution: bool = False):
        headers = {"accept": "application/json", "x-request-id": "sdk_request"}
        if self.api_token:
            headers["authorization"] = "Bearer " + self.api_token
        if execution and self.execution_token:
            headers["x-auro-execution-token"] = self.execution_token
        data = None
        if body is not None:
            headers["content-type"] = "application/json"
            data = json.dumps(body).encode()
        try:
            with urlopen(Request(self.base_url + path, data=data, headers=headers, method=method), timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            try: payload = json.loads(exc.read())
            except Exception: payload = {}
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            raise AuroAPIError(exc.code, error.get("code", "http_error"), error.get("message", str(exc)), error.get("request_id")) from exc
