from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from typing import Any
from urllib import request as urlrequest


@dataclass(frozen=True)
class ServingProfile:
    name: str
    transport: str
    checkpoint_format: str
    command_template: tuple[str, ...]
    openai_compatible: bool
    status: str = "PROTOTYPE"


PROFILES: dict[str, ServingProfile] = {
    "transformers": ServingProfile(
        "transformers",
        "in-process",
        "huggingface-safetensors",
        ("python", "-m", "auro_native_llm.serve.st14b_transformers", "--model", "{model_path}"),
        False,
    ),
    "vllm": ServingProfile(
        "vllm",
        "http-openai",
        "huggingface-safetensors",
        ("vllm", "serve", "{model_path}", "--tensor-parallel-size", "{tp}", "--dtype", "{dtype}", "--max-model-len", "8192"),
        True,
    ),
    "tensorrt-llm": ServingProfile(
        "tensorrt-llm",
        "http-openai",
        "tensorrt-engine",
        ("trtllm-serve", "{engine_path}", "--host", "127.0.0.1", "--port", "8000"),
        True,
    ),
    "llama.cpp": ServingProfile(
        "llama.cpp",
        "http-openai",
        "gguf",
        ("llama-server", "-m", "{gguf_path}", "-c", "8192", "--host", "127.0.0.1", "--port", "8080"),
        True,
    ),
}


def serving_manifest() -> dict[str, Any]:
    return {
        "schema": "auro.st14b.serving.v1",
        "profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
        "promotion_requirements": [
            "checkpoint_sha256",
            "tokenizer_sha256",
            "exact_parameter_count",
            "runtime_version",
            "hardware_identity",
            "quality_receipt",
            "latency_throughput_receipt",
        ],
        "status": "PROTOTYPE",
    }


def render_command(name: str, **values: Any) -> list[str]:
    if name not in PROFILES:
        raise ValueError(f"unknown runtime profile: {name}")
    rendered = []
    defaults = {"tp": 1, "dtype": "bfloat16"}
    merged = {**defaults, **values}
    for token in PROFILES[name].command_template:
        try:
            rendered.append(token.format(**merged))
        except KeyError as exc:
            raise ValueError(f"missing runtime argument: {exc.args[0]}") from exc
    return rendered


class OpenAICompatibleAdapter:
    """Minimal fail-closed client for vLLM, TensorRT-LLM and llama.cpp servers."""

    def __init__(self, runtime: str, base_url: str, model: str, token: str | None = None) -> None:
        profile = PROFILES.get(runtime)
        if profile is None or not profile.openai_compatible:
            raise ValueError("runtime is not an OpenAI-compatible serving profile")
        if not base_url.startswith(("http://127.0.0.1", "http://localhost", "https://")):
            raise ValueError("remote plaintext serving endpoints are not permitted")
        self.runtime = runtime
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token = token or ""

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int = 128, temperature: float = 0.0, timeout: int = 60) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages are required")
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max(1, min(max_tokens, 4096)),
            "temperature": temperature,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urlrequest.Request(f"{self.base_url}/v1/chat/completions", data=body, method="POST", headers=headers)
        with urlrequest.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"runtime": self.runtime, "model": self.model, "response": payload}

    def health(self, timeout: int = 10) -> dict[str, Any]:
        req = urlrequest.Request(f"{self.base_url}/v1/models", method="GET")
        with urlrequest.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"ok": True, "runtime": self.runtime, "payload": payload}


def write_serving_manifest(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(serving_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    return target
