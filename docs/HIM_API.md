# Auro14B · HIM API

HIM now exposes two stable response contracts from the same governed runtime:

- `POST /v1/him/respond` — native, receipt-rich Auro response
- `POST /v1/chat/completions` — OpenAI-compatible non-streaming chat response with evidence under `auro`

## Start

```bash
export AURO_BASE_URL=http://127.0.0.1:8088/v1
export AURO_MODEL=medina-native-8b
export AURO_API_TOKEN=replace-with-a-long-random-token
export AURO_EXECUTION_TOKEN=use-a-different-long-random-token
python -m auro_native_llm.production_fleet.server --host 127.0.0.1 --port 8090
```

`AURO_API_TOKEN` is optional for loopback development. Set it for any shared environment. `AURO_EXECUTION_TOKEN` is required for execution and approved mutation. Never bind publicly without a TLS reverse proxy, network access controls, and both tokens.

## Python SDK

```python
from auro_native_llm.production_fleet.client import AuroClient

auro = AuroClient(api_token="…", execution_token="…")
print(auro.models())
reply = auro.respond("Explain the current MESIE proof boundary")
print(reply["answer"])

compatible = auro.chat([
    {"role": "system", "content": "Answer with evidence."},
    {"role": "user", "content": "What can HIM actually execute?"},
])
print(compatible["choices"][0]["message"]["content"])
```

## HTTP

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer $AURO_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"auro-him","messages":[{"role":"user","content":"What is HIM?"}]}'
```

Approved execution additionally uses `X-Auro-Execution-Token`. The API token and execution token are intentionally separate.

## Discovery and operations

| Route | Purpose |
|---|---|
| `GET /health` | unauthenticated process liveness |
| `GET /v1/health/ready` | endpoint configuration and receipt-chain readiness |
| `GET /v1` | API discovery |
| `GET /v1/models` | honest configured-model metadata |
| `GET /v1/capabilities` | machine-readable internal capability contracts |
| `POST /v1/capabilities/call` | governed capability call |
| `GET /v1/receipts` | recent evidence |
| `GET /v1/receipts/verify` | hash-chain verification |
| `GET /openapi.json` | OpenAPI 3.1 discovery document |

Every API response includes `X-Request-ID` and `X-Auro-API-Version`. Client-supplied request IDs are accepted only when short and restricted to alphanumeric, hyphen, and underscore characters.

## Compatibility boundary

The chat endpoint implements non-streaming text chat, not the entire OpenAI API. Token usage remains `null` unless the underlying runtime reports exact tokenizer accounting. Evidence extensions remain available under `auro`; private chain-of-thought is never returned.

