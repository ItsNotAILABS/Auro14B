"""Cloudflare outside-plane contract without implicit remote execution."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path


class CloudflareRuntimeContract:
    def __init__(self, path=None):
        self.path=Path(path or Path(__file__).resolve().parents[2]/"configs"/"cloudflare_runtime.json")
        self.data=json.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("schema")!="auro.cloudflare.runtime.v1": raise ValueError("unsupported Cloudflare runtime contract")
        if self.data["mcp"].get("tools") != ["search","execute"]: raise ValueError("Cloudflare API MCP must expose search then execute")
    def manifest(self):
        raw=self.path.read_bytes(); configured=bool(os.getenv("CLOUDFLARE_API_TOKEN"))
        return {**self.data,"contract_sha256":hashlib.sha256(raw).hexdigest(),"configured":configured,"credential_exposed":False,"execution_available":configured and bool(os.getenv("AURO_ENABLE_CLOUDFLARE_EXECUTE"))}
    def mcp_client_config(self):
        return {"mcpServers":{"cloudflare-api":{"url":self.data["mcp"]["url"]}}}
    def recipe(self, objective: str):
        return {"schema":"auro.cloudflare.recipe.v1","objective":objective,"steps":[{"tool":"search","query":objective,"mutating":False},{"tool":"execute","endpoint":"selected-from-search","params":"operator-reviewed","mutating":True,"approval_required":True}],"executed":False}
