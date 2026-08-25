"""Production adapter for the Auro-2B hierarchical council runtime.

The service is disabled until an operator supplies an explicit endpoint
manifest. It does not silently reuse one endpoint under four model names, and it
never treats a configured endpoint as proof of an exact promoted checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from auro_native_llm.model.council_runtime import (
    Auro2BCouncilRuntime,
    CouncilTurnResult,
    ModelExecutor,
    ModelIdentity,
)


CONFIG_SCHEMA = "auro.2b-council.config.v1"


class CouncilUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    model_id: str
    parameter_target: int
    base_url: str
    model: str
    api_key_env: str | None = None
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    adapter_id: str | None = None
    adapter_sha256: str | None = None
    measured_parameters: int | None = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EndpointConfig":
        model_id = str(value.get("model_id") or "").strip()
        base_url = str(value.get("base_url") or "").strip()
        model = str(value.get("model") or "").strip()
        if not model_id or not base_url or not model:
            raise ValueError("model_id, base_url, and model are required for each council endpoint")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"unsupported council endpoint URL for {model_id}")
        return cls(
            model_id=model_id,
            parameter_target=int(value.get("parameter_target") or 0),
            base_url=base_url,
            model=model,
            api_key_env=str(value.get("api_key_env") or "").strip() or None,
            checkpoint_id=str(value.get("checkpoint_id") or "").strip() or None,
            checkpoint_sha256=str(value.get("checkpoint_sha256") or "").strip() or None,
            adapter_id=str(value.get("adapter_id") or "").strip() or None,
            adapter_sha256=str(value.get("adapter_sha256") or "").strip() or None,
            measured_parameters=(
                int(value["measured_parameters"])
                if value.get("measured_parameters") is not None
                else None
            ),
            timeout_seconds=max(1.0, min(float(value.get("timeout_seconds", 120.0)), 600.0)),
        )

    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            model_id=self.model_id,
            parameter_target=self.parameter_target,
            checkpoint_id=self.checkpoint_id,
            checkpoint_sha256=self.checkpoint_sha256,
            adapter_id=self.adapter_id,
            adapter_sha256=self.adapter_sha256,
            measured_parameters=self.measured_parameters,
            provider="openai-compatible-explicit",
        )

    def public(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_target": self.parameter_target,
            "origin": f"{urlsplit(self.base_url).scheme}://{urlsplit(self.base_url).netloc}",
            "model": self.model,
            "api_key_configured": bool(self.api_key_env and os.getenv(self.api_key_env)),
            "identity": self.identity().public(),
        }


class OpenAIEndpointCallable:
    def __init__(self, endpoint: EndpointConfig):
        self.endpoint = endpoint

    def __call__(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.endpoint.model,
            "messages": messages,
            "stream": False,
            "temperature": float(options.get("temperature", 0.2)),
            "max_tokens": int(options.get("max_tokens", 256)),
        }
        headers = {"content-type": "application/json"}
        if self.endpoint.api_key_env:
            value = os.getenv(self.endpoint.api_key_env, "")
            if value:
                headers["authorization"] = "Bearer " + value
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.endpoint.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], Mapping):
            raise RuntimeError(f"{self.endpoint.model_id} endpoint returned no choices")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "")
        return {
            "text": text,
            "usage": dict(payload.get("usage") or {}),
            "raw_model": payload.get("model"),
        }


class CouncilService:
    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        runtime: Auro2BCouncilRuntime | None = None,
        source: str = "disabled",
    ) -> None:
        self.source = source
        self.config = dict(config or {})
        self.config_sha256 = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        self.runtime = runtime
        self.blockers: list[str] = []
        if self.runtime is None:
            self.blockers.append("AURO_COUNCIL_CONFIG_JSON or AURO_COUNCIL_CONFIG_PATH is not configured")

    @classmethod
    def from_env(cls) -> "CouncilService":
        raw = os.getenv("AURO_COUNCIL_CONFIG_JSON", "").strip()
        path = os.getenv("AURO_COUNCIL_CONFIG_PATH", "").strip()
        if raw and path:
            raise ValueError("configure only one of AURO_COUNCIL_CONFIG_JSON or AURO_COUNCIL_CONFIG_PATH")
        if path:
            config_path = Path(path).expanduser().resolve()
            if not config_path.is_file():
                raise ValueError(f"council config file does not exist: {config_path}")
            value = json.loads(config_path.read_text(encoding="utf-8"))
            source = str(config_path)
        elif raw:
            value = json.loads(raw)
            source = "environment"
        else:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("council configuration must be a JSON object")
        runtime = cls._build_runtime(value)
        return cls(config=value, runtime=runtime, source=source)

    @staticmethod
    def _build_runtime(value: Mapping[str, Any]) -> Auro2BCouncilRuntime:
        if value.get("schema") != CONFIG_SCHEMA:
            raise ValueError(f"council config schema must be {CONFIG_SCHEMA}")
        main_cfg = EndpointConfig.from_mapping(dict(value.get("main") or {}))
        specialist_values = list(value.get("specialists") or [])
        atomic_values = list(value.get("atomic") or [])
        if len(specialist_values) != 3:
            raise ValueError("council config requires exactly three specialists")
        specialists = [EndpointConfig.from_mapping(dict(item)) for item in specialist_values]
        atomics = [EndpointConfig.from_mapping(dict(item)) for item in atomic_values]

        main = ModelExecutor(main_cfg.identity(), OpenAIEndpointCallable(main_cfg))
        specialist_executors = [
            ModelExecutor(item.identity(), OpenAIEndpointCallable(item)) for item in specialists
        ]
        atomic_executors = {
            item.model_id: ModelExecutor(item.identity(), OpenAIEndpointCallable(item))
            for item in atomics
        }
        signing_key = os.getenv("AURO_COUNCIL_RECEIPT_HMAC_KEY", "") or None
        signer_id = os.getenv("AURO_COUNCIL_RECEIPT_SIGNER", "auro-council-local")
        return Auro2BCouncilRuntime(
            main_2b=main,
            specialists=specialist_executors,
            atomic_executors=atomic_executors,
            max_workers=int(value.get("max_workers", 12)),
            signing_key=signing_key,
            signer_id=signer_id,
        )

    @property
    def configured(self) -> bool:
        return self.runtime is not None

    def status(self) -> dict[str, Any]:
        return {
            "schema": "auro.2b-council.service-status.v1",
            "configured": self.configured,
            "source": self.source,
            "config_sha256": self.config_sha256 if self.config else None,
            "blockers": list(self.blockers),
            "runtime": self.runtime.manifest() if self.runtime else None,
            "claim_boundary": (
                "configuration proves endpoint wiring only; checkpoint identity, model quality, "
                "latency, and promotion require separate evidence"
            ),
        }

    def respond(
        self,
        message: str,
        *,
        full_parent_context: str | None = None,
    ) -> dict[str, Any]:
        if self.runtime is None:
            raise CouncilUnavailable("Auro-2B council is not configured")
        result: CouncilTurnResult = self.runtime.run_turn(
            message,
            full_parent_context=full_parent_context,
        )
        return result.to_dict()


def example_config() -> dict[str, Any]:
    """Return a non-secret operator template for deployment documentation."""
    return {
        "schema": CONFIG_SCHEMA,
        "main": {
            "model_id": "Auro-2B",
            "parameter_target": 2_000_000_000,
            "base_url": "http://127.0.0.1:8088/v1",
            "model": "auro-2b",
            "api_key_env": "AURO_2B_API_KEY",
            "checkpoint_id": "replace-with-exact-checkpoint-id",
            "checkpoint_sha256": "replace-with-64-character-sha256",
        },
        "specialists": [
            {
                "model_id": model_id,
                "parameter_target": 500_000_000,
                "base_url": "http://127.0.0.1:8088/v1",
                "model": model_id.lower(),
                "api_key_env": "AURO_500M_API_KEY",
                "checkpoint_id": "replace-with-exact-checkpoint-or-base-id",
                "checkpoint_sha256": "replace-with-64-character-sha256",
                "adapter_id": "replace-with-specialist-adapter-id",
                "adapter_sha256": "replace-with-64-character-sha256",
            }
            for model_id in (
                "Auro-500M-SENSUS",
                "Auro-500M-PRAXIS",
                "Auro-500M-VERBUM",
            )
        ],
        "atomic": [
            {
                "model_id": model_id,
                "parameter_target": parameter_target,
                "base_url": "http://127.0.0.1:8088/v1",
                "model": model_id.lower(),
                "api_key_env": "AURO_ATOMIC_API_KEY",
                "checkpoint_id": "replace-with-exact-checkpoint-id",
                "checkpoint_sha256": "replace-with-64-character-sha256",
            }
            for model_id, parameter_target in (
                ("Auro-156K", 156_000),
                ("Auro-250M", 250_000_000),
            )
        ],
        "max_workers": 12,
    }
