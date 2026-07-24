"""Explicit embedded/local/cloud compute-plane registry."""
from __future__ import annotations
import json, os


class ComputeRegistry:
    def __init__(self, raw=None):
        raw = os.getenv("AURO_CLOUD_ENGINES_JSON", "[]") if raw is None else raw
        try: configured = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc: raise ValueError("AURO_CLOUD_ENGINES_JSON must be JSON") from exc
        if not isinstance(configured, list): raise ValueError("cloud engines must be a list")
        self.cloud=[]
        for item in configured:
            if not isinstance(item,dict) or not item.get("id") or not item.get("base_url") or not item.get("model"):
                raise ValueError("each cloud engine requires id, base_url, and model")
            self.cloud.append({"id":str(item["id"]),"base_url":str(item["base_url"]),"model":str(item["model"]),"api_key_env":str(item.get("api_key_env") or ""),"credential_configured":bool(item.get("api_key_env") and os.getenv(str(item["api_key_env"])))})
    def manifest(self):
        return {"schema":"auro.compute.engines.v1","default":"embedded-browser","remote_fallback":False,"engines":[{"id":"embedded-browser","plane":"browser","backend":"onnxruntime-web","network_model_load":False},{"id":"local-auro","plane":"local","backend":"openai-compatible","base_url":os.getenv("AURO_BASE_URL","http://127.0.0.1:8088/v1")},*[{**x,"plane":"cloud","api_key_env":x["api_key_env"] or None} for x in self.cloud]]}
